#pragma once

#include <span>
#include <vector>

#include "tf/core/base/Error.h"
#include "tf/core/base/Types.h"

namespace tf::runtime {

/// How the next token is chosen from the logits.
///
/// Defaults match upstream's CLI. They are low-temperature on purpose: this is
/// an instruction-tuned model, and the failure mode users notice is not
/// blandness but drift into confident nonsense.
struct SamplingParams {
    /// Zero means greedy. Anything else divides the logits before the softmax,
    /// so smaller is more deterministic.
    float temperature = 0.2f;

    /// Keep only the `topK` highest-scoring tokens. Zero disables it.
    u32 topK = 64;

    /// Keep the shortest prefix of the sorted distribution whose probabilities
    /// reach `topP`. One or above disables it.
    float topP = 0.95f;

    /// Divides the logit of any token seen in the recent history, or multiplies
    /// it when the logit is negative - which is the detail that makes this
    /// behave as a penalty in both directions. One disables it.
    float repetitionPenalty = 1.0f;

    /// How far back the penalty looks.
    u64 repetitionWindow = 64;

    /// Zero seeds from the system entropy source; anything else makes a run
    /// reproducible.
    u64 seed = 0;

    [[nodiscard]] bool isGreedy() const noexcept { return temperature <= 0.0f; }

    [[nodiscard]] Status validate() const;
};

/// Chooses tokens from a logit vector.
///
/// Sampling runs on the host. The logits have to cross PCIe for any non-greedy
/// choice anyway, and at 262144 entries that is about a millisecond against a
/// 25 ms token - visible but not worth a kernel until the rest of M13's tuning
/// is done. Greedy decoding does not download anything: ForwardRunner's argmax
/// kernel handles it on the device.
class Sampler {
public:
    explicit Sampler(SamplingParams params);

    /// Picks a token, modifying `logits` in place as it applies the penalty and
    /// the temperature.
    ///
    /// `history` is the tokens generated so far, most recent last; only the
    /// last `repetitionWindow` of them are considered.
    [[nodiscard]] u32 sample(std::span<float> logits, std::span<const u32> history);

    [[nodiscard]] const SamplingParams& params() const noexcept { return params_; }

    /// The seed actually in use, which is the resolved one when the caller
    /// asked for zero. Worth reporting: it is what makes a run repeatable.
    [[nodiscard]] u64 seed() const noexcept { return seed_; }

private:
    /// A token and its score, kept together through the sort.
    struct Candidate {
        u32 id = 0;
        float score = 0.0f;
    };

    [[nodiscard]] float nextUniform();

    SamplingParams params_;
    u64 seed_ = 0;
    u64 state_ = 0;
    std::vector<Candidate> candidates_;
};

}  // namespace tf::runtime
