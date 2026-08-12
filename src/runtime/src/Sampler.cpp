#include "tf/runtime/Sampler.h"

#include <algorithm>
#include <chrono>
#include <cmath>

namespace tf::runtime {
namespace {

/// SplitMix64. Chosen because it needs no warm-up and one 64-bit word of state,
/// so a seed maps to a stream directly and a run is reproducible from the seed
/// alone.
[[nodiscard]] u64 splitMix64(u64& state) {
    state += 0x9E3779B97F4A7C15ull;
    u64 z = state;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

}  // namespace

Status SamplingParams::validate() const {
    if (temperature < 0.0f || !std::isfinite(temperature)) {
        return makeError(ErrorCode::InvalidArgument,
                         "temperature {} is not a finite, non-negative value", temperature);
    }
    if (topP <= 0.0f || !std::isfinite(topP)) {
        return makeError(ErrorCode::InvalidArgument, "top-p {} must be above zero", topP);
    }
    if (repetitionPenalty <= 0.0f || !std::isfinite(repetitionPenalty)) {
        return makeError(ErrorCode::InvalidArgument, "repetition penalty {} must be above zero",
                         repetitionPenalty);
    }
    return {};
}

Sampler::Sampler(SamplingParams params) : params_(params) {
    seed_ = params_.seed;
    if (seed_ == 0) {
        seed_ = static_cast<u64>(
                std::chrono::steady_clock::now().time_since_epoch().count());
        // Never resolve to zero, so seed() always round-trips into a run that
        // reproduces this one.
        if (seed_ == 0) {
            seed_ = 1;
        }
    }
    state_ = seed_;
}

float Sampler::nextUniform() {
    // The top 24 bits, scaled: enough precision for a distribution of at most a
    // few hundred candidates, and it never returns exactly 1.0.
    const u64 bits = splitMix64(state_) >> 40;
    return static_cast<float>(bits) / static_cast<float>(1u << 24);
}

u32 Sampler::sample(std::span<float> logits, std::span<const u32> history) {
    if (logits.empty()) {
        return 0;
    }

    // ---- Repetition penalty ---------------------------------------------
    //
    // Dividing a positive logit and multiplying a negative one both push the
    // token down. Applying only the division would *raise* an already unlikely
    // token, which is the classic bug in this operation.
    if (params_.repetitionPenalty != 1.0f && !history.empty()) {
        const usize window =
                std::min(static_cast<usize>(params_.repetitionWindow), history.size());
        for (const u32 id : history.last(window)) {
            if (id >= logits.size()) {
                continue;
            }
            float& value = logits[id];
            value = value > 0.0f ? value / params_.repetitionPenalty
                                 : value * params_.repetitionPenalty;
        }
    }

    // ---- Greedy ----------------------------------------------------------
    if (params_.isGreedy()) {
        u32 best = 0;
        float bestScore = logits[0];
        for (u32 i = 1; i < logits.size(); ++i) {
            // Strictly greater keeps the lowest index on a tie, matching the
            // GPU argmax so the two paths cannot disagree.
            if (logits[i] > bestScore) {
                bestScore = logits[i];
                best = i;
            }
        }
        return best;
    }

    // ---- Candidate set ---------------------------------------------------
    //
    // Top-k first, because it bounds everything that follows: the softmax, the
    // top-p scan and the sort all run over k entries rather than the whole
    // 262144-token vocabulary.
    const usize keep = params_.topK == 0
                               ? logits.size()
                               : std::min(static_cast<usize>(params_.topK), logits.size());

    candidates_.resize(logits.size());
    for (u32 i = 0; i < logits.size(); ++i) {
        candidates_[i] = Candidate{.id = i, .score = logits[i]};
    }

    const auto byScore = [](const Candidate& a, const Candidate& b) {
        // Descending, with the lower id winning a tie so sampling is
        // reproducible across standard-library implementations.
        return a.score != b.score ? a.score > b.score : a.id < b.id;
    };

    if (keep < candidates_.size()) {
        std::nth_element(candidates_.begin(),
                         candidates_.begin() + static_cast<isize>(keep) - 1,
                         candidates_.end(), byScore);
        candidates_.resize(keep);
    }
    std::sort(candidates_.begin(), candidates_.end(), byScore);

    // ---- Temperature and softmax ----------------------------------------
    const float inverseTemperature = 1.0f / params_.temperature;
    const float maximum = candidates_.front().score;
    double total = 0.0;
    for (Candidate& candidate : candidates_) {
        // Subtracting the maximum before the exponential: the raw logits reach
        // +-30 after the softcap, and dividing by a temperature of 0.05 puts
        // them well past the range of expf.
        const auto weight = static_cast<float>(
                std::exp(static_cast<double>(candidate.score - maximum) * inverseTemperature));
        candidate.score = weight;
        total += weight;
    }

    // ---- Top-p -----------------------------------------------------------
    //
    // The prefix is kept inclusive of the token that crosses the threshold, so
    // a top-p below the single most likely token's probability still leaves one
    // candidate rather than none.
    if (params_.topP < 1.0f) {
        const double limit = static_cast<double>(params_.topP) * total;
        double running = 0.0;
        usize kept = 0;
        for (const Candidate& candidate : candidates_) {
            running += candidate.score;
            ++kept;
            if (running >= limit) {
                break;
            }
        }
        candidates_.resize(kept);
        total = running;
    }

    // ---- Draw ------------------------------------------------------------
    const double target = static_cast<double>(nextUniform()) * total;
    double running = 0.0;
    for (const Candidate& candidate : candidates_) {
        running += candidate.score;
        if (running >= target) {
            return candidate.id;
        }
    }
    // Only reachable through floating-point drift in the running sum.
    return candidates_.back().id;
}

}  // namespace tf::runtime
