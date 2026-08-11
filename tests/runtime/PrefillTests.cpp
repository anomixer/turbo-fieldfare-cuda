// Prefill against decode, on real weights.
//
// The decode path is the one already checked against the CPU reference stage by
// stage, so it is the reference here. A chunk of N tokens must leave the same
// residual stream, the same KV cache and the same logits as running those N
// tokens through decodeStep one at a time - if it does not, the batching is
// wrong somewhere, and this says so without a 30-layer bisection.
//
// The two paths are genuinely different code below the layer sequence: batched
// GEMMs instead of GEMVs, per-token causal masking inside one attention launch,
// and routed experts grouped by expert rather than looped per token. Agreement
// to fp32 rounding across all of that is a strong signal.
//
// Skipped when no install is present.

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <format>
#include <span>
#include <string>
#include <vector>

#include "tf/gpu/Backend.h"
#include "tf/runtime/ForwardRunner.h"
#include "tf/runtime/KVCache.h"
#include "tf/runtime/Model.h"

using namespace tf;

namespace {

std::filesystem::path installDir() {
    if (const char* override = std::getenv("TF_GTURBO_DIR")) {
        return std::filesystem::path{override};
    }
    if (const char* home = std::getenv("USERPROFILE")) {
        return std::filesystem::path{home} / "model-data" / "gemma4.gturbo";
    }
    return {};
}

/// How far apart two tensors are, both in the worst element and on average.
///
/// Both numbers are needed to tell a structural disagreement from accumulation
/// rounding. Rounding shows up as a worst-element ratio well above the RMS
/// ratio - one unlucky element out of thousands - whereas a mis-grouped expert
/// or a bad mask moves the whole tensor and drives both up together.
struct Deviation {
    double worstOverRms = 0.0;
    double rmsOverRms = 0.0;

    [[nodiscard]] std::string describe() const {
        return std::format("worst/rms {:.5f}, rms/rms {:.6f}", worstOverRms, rmsOverRms);
    }
};

/// A non-finite value on either side returns infinity rather than being quietly
/// skipped by std::max, which is how a NaN once hid behind a passing check.
[[nodiscard]] Deviation deviationVersusRms(std::span<const float> reference,
                                           std::span<const float> candidate) {
    REQUIRE(reference.size() == candidate.size());
    double sumSquares = 0.0;
    double errorSquares = 0.0;
    double worst = 0.0;
    for (usize i = 0; i < reference.size(); ++i) {
        if (!std::isfinite(reference[i]) || !std::isfinite(candidate[i])) {
            return Deviation{.worstOverRms = std::numeric_limits<double>::infinity(),
                             .rmsOverRms = std::numeric_limits<double>::infinity()};
        }
        const double error = static_cast<double>(reference[i]) - candidate[i];
        sumSquares += static_cast<double>(reference[i]) * reference[i];
        errorSquares += error * error;
        worst = std::max(worst, std::abs(error));
    }
    const auto count = static_cast<double>(reference.size());
    const double rms = std::sqrt(sumSquares / count);
    const double errorRms = std::sqrt(errorSquares / count);
    if (rms <= 0.0) {
        return Deviation{.worstOverRms = worst, .rmsOverRms = errorRms};
    }
    return Deviation{.worstOverRms = worst / rms, .rmsOverRms = errorRms / rms};
}

/// A model, a cache and a runner, built fresh so each path starts from an empty
/// KV cache and an empty expert slot cache.
struct Engine {
    gpu::BackendPtr backend;
    runtime::Model model;
    runtime::KVCacheManager cache;
    runtime::ForwardRunner runner;
};

/// Tokens with no particular meaning, spread across the vocabulary. Prefill
/// correctness does not depend on them being a real sentence, and arbitrary ids
/// exercise more of the router than a repeated token would.
[[nodiscard]] std::vector<u32> sampleTokens(u32 count, u64 vocabSize) {
    std::vector<u32> ids;
    ids.reserve(count);
    u64 state = 12345;
    for (u32 i = 0; i < count; ++i) {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        ids.push_back(static_cast<u32>((state >> 33) % vocabSize));
    }
    return ids;
}

}  // namespace

TEST_CASE("a prefill chunk matches the same tokens run one at a time",
          "[prefill]") {
    const auto dir = installDir();
    if (dir.empty() || !std::filesystem::exists(dir / "manifest.json")) {
        SKIP("no .gturbo install at " << dir.string());
    }
    if (gpu::compiledBackends().empty()) {
        SKIP("no GPU backend");
    }

    // 17 crosses the GEMM's 16-token tile, so the last tile is partial and the
    // out-of-range rows in it have to be handled rather than merely not crash.
    const u32 tokenCount = GENERATE(u32{1}, u32{2}, u32{16}, u32{17}, u32{40});
    CAPTURE(tokenCount);

    auto backendResult = gpu::createBackend(gpu::compiledBackends().front());
    if (!backendResult) {
        SKIP("GPU unavailable: " << backendResult.error().message());
    }
    gpu::IGpuBackend& backend = **backendResult;

    constexpr u64 kContext = 256;
    auto modelResult = runtime::Model::load(
            backend, dir, runtime::Model::LoadOptions{.budget = {.contextLength = kContext}});
    REQUIRE(modelResult.has_value());
    runtime::Model& model = *modelResult;
    const ArchInfo& arch = model.arch();

    const auto tokens = sampleTokens(tokenCount, arch.vocabSize);
    const auto hiddenSize = static_cast<usize>(arch.hiddenSize);

    // ---- Reference: one token at a time ---------------------------------
    std::vector<float> decodeHidden;
    std::vector<float> decodeLogits;
    {
        auto cacheResult = runtime::KVCacheManager::create(backend, arch, kContext);
        REQUIRE(cacheResult.has_value());
        auto runnerResult = runtime::ForwardRunner::create(backend, model, *cacheResult);
        REQUIRE(runnerResult.has_value());

        for (u32 i = 0; i < tokenCount; ++i) {
            const auto status =
                    runnerResult->decodeStep(tokens[i], cacheResult->position(),
                                             /*computeLogits=*/i + 1 == tokenCount);
            INFO("decode step " << i << ": "
                                << (status ? std::string{} : status.error().toString()));
            REQUIRE(status.has_value());
            cacheResult->advance(1);
        }

        auto hidden = runnerResult->readHiddenState();
        REQUIRE(hidden.has_value());
        decodeHidden = std::move(*hidden);

        auto logits = runnerResult->readLogits();
        REQUIRE(logits.has_value());
        decodeLogits = std::move(*logits);
    }

    // ---- Candidate: the whole chunk at once ------------------------------
    std::vector<float> prefillHidden;
    std::vector<float> prefillLogits;
    {
        auto cacheResult = runtime::KVCacheManager::create(backend, arch, kContext);
        REQUIRE(cacheResult.has_value());
        auto runnerResult = runtime::ForwardRunner::create(backend, model, *cacheResult, 64);
        REQUIRE(runnerResult.has_value());

        const auto status = runnerResult->prefillChunk(tokens, 0);
        INFO("prefill: " << (status ? std::string{} : status.error().toString()));
        REQUIRE(status.has_value());
        cacheResult->advance(tokenCount);

        REQUIRE(runnerResult->runHeadOnly().has_value());

        // readHiddenState returns one token; after a chunk that is the first,
        // so read the whole stream and take the last, which is the one the
        // decode path ended on.
        auto hidden = runnerResult->readScratch("hidden", hiddenSize * tokenCount);
        REQUIRE(hidden.has_value());
        prefillHidden.assign(hidden->end() - static_cast<isize>(hiddenSize), hidden->end());

        auto logits = runnerResult->readLogits();
        REQUIRE(logits.has_value());
        prefillLogits = std::move(*logits);
    }

    // The two paths sum the same products in a different order, so they differ
    // by accumulation rounding but nothing else. A real disagreement - a
    // mis-grouped expert, an off-by-one in the causal mask - lands far above
    // this, and would move the RMS as well as the worst element.
    constexpr double kWorstTolerance = 0.05;
    constexpr double kRmsTolerance = 0.005;

    const auto hiddenDeviation = deviationVersusRms(decodeHidden, prefillHidden);
    WARN("hidden: " << hiddenDeviation.describe());
    CHECK(hiddenDeviation.worstOverRms < kWorstTolerance);
    CHECK(hiddenDeviation.rmsOverRms < kRmsTolerance);

    const auto logitDeviation = deviationVersusRms(decodeLogits, prefillLogits);
    WARN("logits: " << logitDeviation.describe());
    CHECK(logitDeviation.worstOverRms < kWorstTolerance);
    CHECK(logitDeviation.rmsOverRms < kRmsTolerance);

    // The argmax is what actually reaches the user, so check it survives even
    // if the tolerance above were ever loosened.
    const auto decodeBest = static_cast<usize>(
            std::max_element(decodeLogits.begin(), decodeLogits.end()) - decodeLogits.begin());
    const auto prefillBest =
            static_cast<usize>(std::max_element(prefillLogits.begin(), prefillLogits.end()) -
                               prefillLogits.begin());
    CHECK(decodeBest == prefillBest);
}

TEST_CASE("a chunk wider than the KV rings were sized for is refused",
          "[prefill]") {
    const auto dir = installDir();
    if (dir.empty() || !std::filesystem::exists(dir / "manifest.json")) {
        SKIP("no .gturbo install at " << dir.string());
    }
    if (gpu::compiledBackends().empty()) {
        SKIP("no GPU backend");
    }
    auto backendResult = gpu::createBackend(gpu::compiledBackends().front());
    if (!backendResult) {
        SKIP("GPU unavailable: " << backendResult.error().message());
    }
    gpu::IGpuBackend& backend = **backendResult;

    constexpr u64 kContext = 256;
    auto modelResult = runtime::Model::load(
            backend, dir, runtime::Model::LoadOptions{.budget = {.contextLength = kContext}});
    REQUIRE(modelResult.has_value());
    const ArchInfo& arch = modelResult->arch();

    // Rings sized for 16, a runner willing to take 64. The chunk has to be
    // rejected rather than silently overwriting the sliding history its own
    // early tokens read - a failure that produces fluent, wrong output.
    auto cacheResult = runtime::KVCacheManager::create(backend, arch, kContext,
                                                       /*maxPrefillTokens=*/16);
    REQUIRE(cacheResult.has_value());
    auto runnerResult = runtime::ForwardRunner::create(backend, *modelResult, *cacheResult, 64);
    REQUIRE(runnerResult.has_value());

    const auto tokens = sampleTokens(40, arch.vocabSize);
    const auto status = runnerResult->prefillChunk(tokens, 0);
    REQUIRE_FALSE(status.has_value());
    CHECK(status.error().code() == ErrorCode::InvalidArgument);
    CHECK(status.error().message().find("sliding window") != std::string::npos);

    // Within the ring's headroom it goes through.
    CHECK(runnerResult->prefillChunk(std::span<const u32>{tokens}.first(16), 0).has_value());
}

TEST_CASE("the scratch estimate covers what the runner actually allocates",
          "[prefill]") {
    const auto dir = installDir();
    if (dir.empty() || !std::filesystem::exists(dir / "manifest.json")) {
        SKIP("no .gturbo install at " << dir.string());
    }
    if (gpu::compiledBackends().empty()) {
        SKIP("no GPU backend");
    }
    auto backendResult = gpu::createBackend(gpu::compiledBackends().front());
    if (!backendResult) {
        SKIP("GPU unavailable: " << backendResult.error().message());
    }
    gpu::IGpuBackend& backend = **backendResult;

    constexpr u64 kContext = 256;
    auto modelResult = runtime::Model::load(
            backend, dir, runtime::Model::LoadOptions{.budget = {.contextLength = kContext}});
    REQUIRE(modelResult.has_value());
    const ArchInfo& arch = modelResult->arch();

    // The estimate is what the residency planner reserves before any of this is
    // allocated. If it ever falls below the real figure the model loads and
    // then the runner fails, which is the worst order for that to happen in.
    const u32 chunk = GENERATE(u32{1}, u32{32}, u32{128});
    CAPTURE(chunk);

    auto cacheResult = runtime::KVCacheManager::create(backend, arch, kContext);
    REQUIRE(cacheResult.has_value());
    auto runnerResult =
            runtime::ForwardRunner::create(backend, *modelResult, *cacheResult, chunk);
    REQUIRE(runnerResult.has_value());

    const u64 estimated = runtime::ForwardRunner::estimateScratchBytes(arch, chunk);
    const u64 actual = runnerResult->scratchBytes();
    WARN("chunk " << chunk << ": estimated " << estimated << " actual " << actual);
    CHECK(estimated >= actual);
    // And not so generous that it starves the residency plan.
    CHECK(estimated <= actual * 2);
}

TEST_CASE("prefill leaves a cache that decode can continue from", "[prefill]") {
    const auto dir = installDir();
    if (dir.empty() || !std::filesystem::exists(dir / "manifest.json")) {
        SKIP("no .gturbo install at " << dir.string());
    }
    if (gpu::compiledBackends().empty()) {
        SKIP("no GPU backend");
    }
    auto backendResult = gpu::createBackend(gpu::compiledBackends().front());
    if (!backendResult) {
        SKIP("GPU unavailable: " << backendResult.error().message());
    }
    gpu::IGpuBackend& backend = **backendResult;

    constexpr u64 kContext = 256;
    constexpr u32 kPrompt = 24;
    constexpr u32 kGenerated = 4;

    auto modelResult = runtime::Model::load(
            backend, dir, runtime::Model::LoadOptions{.budget = {.contextLength = kContext}});
    REQUIRE(modelResult.has_value());
    runtime::Model& model = *modelResult;
    const ArchInfo& arch = model.arch();

    const auto tokens = sampleTokens(kPrompt, arch.vocabSize);

    // What matters is not just that prefill is internally consistent but that a
    // decode step reading the cache it wrote sees the same history a decode-only
    // run would have written. A chunk that filled the wrong ring slots would
    // pass the equivalence test above and fail here.
    const auto continueFrom = [&](bool usePrefill) {
        auto cacheResult = runtime::KVCacheManager::create(backend, arch, kContext);
        REQUIRE(cacheResult.has_value());
        auto runnerResult = runtime::ForwardRunner::create(backend, model, *cacheResult, 64);
        REQUIRE(runnerResult.has_value());

        if (usePrefill) {
            REQUIRE(runnerResult->prefillChunk(tokens, 0).has_value());
            cacheResult->advance(kPrompt);
            REQUIRE(runnerResult->runHeadOnly().has_value());
        } else {
            for (u32 i = 0; i < kPrompt; ++i) {
                REQUIRE(runnerResult->decodeStep(tokens[i], cacheResult->position(),
                                                 /*computeLogits=*/i + 1 == kPrompt)
                                .has_value());
                cacheResult->advance(1);
            }
        }

        std::vector<u32> produced;
        for (u32 i = 0; i < kGenerated; ++i) {
            auto next = runnerResult->greedyToken();
            REQUIRE(next.has_value());
            produced.push_back(*next);
            REQUIRE(runnerResult->decodeStep(*next, cacheResult->position()).has_value());
            cacheResult->advance(1);
        }
        return produced;
    };

    CHECK(continueFrom(/*usePrefill=*/true) == continueFrom(/*usePrefill=*/false));
}
