// One real layer, CPU reference against GPU, stage by stage.
//
// This is what M3's reference implementations were built for. Comparing whole
// generations tells you only that something is wrong somewhere in 30 layers;
// comparing each intermediate against a scalar implementation driven by the
// same real weights says which operation disagrees.
//
// Skipped when no install is present.

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <cmath>
#include <limits>
#include <cstdlib>
#include <filesystem>
#include <span>
#include <vector>

#include "tf/core/format/GTurbo.h"
#include "tf/core/io/File.h"
#include "tf/core/math/Float.h"
#include "tf/gpu/Backend.h"
#include "tf/runtime/ForwardRunner.h"
#include "tf/runtime/KVCache.h"
#include "tf/runtime/Model.h"
#include "tf/reference/Ops.h"
#include "tf/reference/Prng.h"
#include "tf/reference/Tolerance.h"

using namespace tf;
using namespace tf::reference;

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

/// Host-side view of the resident weight file, so the reference can read the
/// same bytes the GPU was given.
class HostWeights {
public:
    HostWeights(const gturbo::Manifest& manifest, const io::MappedFile& resident)
        : manifest_(manifest), resident_(resident) {}

    [[nodiscard]] std::vector<float> bf16Tensor(const std::string& name) const {
        const auto* tensor = manifest_.resident.find(name);
        REQUIRE(tensor != nullptr);
        const auto bytes = resident_.range(tensor->range);
        REQUIRE(bytes.has_value());

        const auto halves = asBf16(*bytes);
        std::vector<float> values(halves.size());
        for (usize i = 0; i < halves.size(); ++i) {
            values[i] = toFloat(halves[i]);
        }
        return values;
    }

    /// Dequantizes a quantized linear into a dense row-major matrix.
    [[nodiscard]] std::vector<float> dense(const std::string& base,
                                           const QuantizedLinearLayout& layout) const {
        const auto* packedTensor = manifest_.resident.find(base + ".weight");
        const auto* scaleTensor = manifest_.resident.find(base + ".scales");
        const auto* biasTensor = manifest_.resident.find(base + ".biases");
        REQUIRE(packedTensor != nullptr);
        REQUIRE(scaleTensor != nullptr);
        REQUIRE(biasTensor != nullptr);

        const auto packedBytes = resident_.range(packedTensor->range);
        const auto scaleBytes = resident_.range(scaleTensor->range);
        const auto biasBytes = resident_.range(biasTensor->range);
        REQUIRE(packedBytes.has_value());

        return dequantizeMatrix(asU32(*packedBytes), asBf16(*scaleBytes), asBf16(*biasBytes),
                                layout);
    }

    /// Matrix-vector product against an already-dequantized matrix.
    [[nodiscard]] static std::vector<float> matvec(std::span<const float> matrix,
                                                   std::span<const float> x, u64 outFeatures,
                                                   u64 inFeatures) {
        std::vector<float> out(static_cast<usize>(outFeatures));
        for (u64 row = 0; row < outFeatures; ++row) {
            double sum = 0.0;
            for (u64 column = 0; column < inFeatures; ++column) {
                sum += static_cast<double>(matrix[static_cast<usize>(row * inFeatures +
                                                                     column)]) *
                       static_cast<double>(x[static_cast<usize>(column)]);
            }
            out[static_cast<usize>(row)] = static_cast<float>(sum);
        }
        return out;
    }

    /// Dequantizes one routed expert's projection from a packed layer file.
    [[nodiscard]] static std::vector<float> expertDense(const io::MappedFile& layerFile,
                                                        const gturbo::ExpertLayout& layout,
                                                        u64 layer, u32 expert,
                                                        std::string_view role,
                                                        const QuantizedLinearLayout& shape) {
        const auto packed = layout.componentRange(layer, expert,
                                                   std::string{role} + ".weight");
        const auto scales = layout.componentRange(layer, expert,
                                                   std::string{role} + ".scales");
        const auto biases = layout.componentRange(layer, expert,
                                                   std::string{role} + ".biases");
        REQUIRE(packed.has_value());
        REQUIRE(scales.has_value());
        REQUIRE(biases.has_value());

        const auto packedBytes = layerFile.range(*packed);
        const auto scaleBytes = layerFile.range(*scales);
        const auto biasBytes = layerFile.range(*biases);
        REQUIRE(packedBytes.has_value());

        return dequantizeMatrix(asU32(*packedBytes), asBf16(*scaleBytes), asBf16(*biasBytes),
                                shape);
    }

private:
    const gturbo::Manifest& manifest_;
    const io::MappedFile& resident_;
};

/// Applies a per-head RMS norm across a flat [heads * headDim] buffer.
std::vector<float> perHeadNorm(std::span<const float> values, std::span<const float> weight,
                               u64 heads, u64 headDim, double eps) {
    std::vector<float> out(values.size());
    for (u64 head = 0; head < heads; ++head) {
        const auto begin = static_cast<usize>(head * headDim);
        const auto slice = values.subspan(begin, static_cast<usize>(headDim));
        const auto normed = weight.empty() ? rmsNormNoScale(slice, eps)
                                           : rmsNorm(slice, weight, eps);
        std::ranges::copy(normed, out.begin() + static_cast<isize>(begin));
    }
    return out;
}

void ropeAllHeads(std::span<float> values, u64 heads, u64 headDim, u64 position,
                  const RopeParams& params) {
    for (u64 head = 0; head < heads; ++head) {
        applyRope(values.subspan(static_cast<usize>(head * headDim),
                                 static_cast<usize>(headDim)),
                  position, params);
    }
}

/// Activations are fp32 on both sides, so this is an identity. Kept as a named
/// step so the "feed both sides the same representation" rule stays visible.
std::vector<float> throughHalf(std::span<const float> values) {
    return {values.begin(), values.end()};
}

/// Largest absolute deviation as a fraction of the reference vector's RMS.
///
/// Per-element relative error is the wrong measure for these tensors. A 2816
/// term dot product whose terms are order 1 can cancel to order 0.001, and a
/// perfectly good kernel then shows 30% "relative error" on that element while
/// being numerically indistinguishable from the reference. Measuring against
/// the vector's own scale is what actually distinguishes a broken kernel from
/// fp16 rounding.
[[nodiscard]] double deviationVersusRms(std::span<const float> reference,
                                        std::span<const float> candidate) {
    REQUIRE(reference.size() == candidate.size());

    double sumSquares = 0.0;
    double worst = 0.0;
    for (usize i = 0; i < reference.size(); ++i) {
        // A non-finite value on either side is a failure outright. Letting it
        // flow into the arithmetic makes the result NaN, and std::max against
        // NaN silently returns the other operand - which reported a deviation
        // of zero and hid five NaN experts behind a passing check.
        if (!std::isfinite(reference[i]) || !std::isfinite(candidate[i])) {
            return std::numeric_limits<double>::infinity();
        }
        sumSquares += static_cast<double>(reference[i]) * reference[i];
        worst = std::max(worst, std::abs(static_cast<double>(reference[i]) - candidate[i]));
    }
    const double rms = std::sqrt(sumSquares / static_cast<double>(reference.size()));
    return rms > 0.0 ? worst / rms : worst;
}

/// fp16 storage plus a long accumulation; anything above this is a real
/// disagreement, not rounding.
constexpr double kRmsTolerance = 0.05;

struct Fixture {
    gpu::BackendPtr backend;
    runtime::Model model;
    runtime::KVCacheManager cache;
    runtime::ForwardRunner runner;
    io::MappedFile resident;
};

}  // namespace

TEST_CASE("the output head matches the reference on real weights",
          "[layer-reference]") {
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
    auto modelResult = runtime::Model::load(backend, dir,
                                            runtime::Model::LoadOptions{
                                                    .budget = {.contextLength = kContext}});
    REQUIRE(modelResult.has_value());
    runtime::Model& model = *modelResult;
    const ArchInfo& arch = model.arch();

    auto cacheResult = runtime::KVCacheManager::create(backend, arch, kContext);
    REQUIRE(cacheResult.has_value());
    auto runnerResult = runtime::ForwardRunner::create(backend, model, *cacheResult);
    REQUIRE(runnerResult.has_value());
    runtime::ForwardRunner& runner = *runnerResult;

    auto residentResult = io::MappedFile::open(dir / gturbo::kResidentFile);
    REQUIRE(residentResult.has_value());
    const HostWeights host{model.manifest(), *residentResult};

    // The head is the last untested surface: everything upstream of it is
    // covered by the per-layer comparison, and a wrong head produces fluent
    // nonsense no matter how correct the layers are.
    SplitMix64 rng{4242};
    const auto hidden =
            throughHalf(randomGaussians(rng, static_cast<usize>(arch.hiddenSize), 2.0f));

    REQUIRE(runner.setHiddenState(hidden).has_value());
    REQUIRE(runner.runHeadOnly().has_value());

    const auto finalNorm = host.bf16Tensor("norm.weight");
    const auto normed = throughHalf(rmsNorm(hidden, finalNorm, arch.rmsNormEps));

    // Tied weights: the embedding table transposed. No sqrt(hiddenSize) scale
    // here - that applies only on the way in.
    const auto embedding = host.dense("embed_tokens", arch.embeddingLayout());
    auto expectedLogits = HostWeights::matvec(embedding, normed, arch.vocabSize,
                                              arch.hiddenSize);
    if (arch.finalLogitSoftcap > 0.0) {
        expectedLogits =
                logitSoftcap(expectedLogits, static_cast<float>(arch.finalLogitSoftcap));
    }
    expectedLogits = throughHalf(expectedLogits);

    const auto actualLogits = runner.readLogits();
    REQUIRE(actualLogits.has_value());

    const double versusRms = deviationVersusRms(expectedLogits, *actualLogits);
    INFO("logits: vs RMS " << versusRms);
    CHECK(versusRms < kRmsTolerance);

    // Greedy decoding depends only on the argmax, so check it directly too.
    CHECK(reference::argmax(expectedLogits) == reference::argmax(*actualLogits));
}

TEST_CASE("a full layer matches the reference on real weights",
          "[layer-reference]") {
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

    // A short context keeps the KV allocation small; only position 0 is used.
    constexpr u64 kContext = 256;

    auto modelResult = runtime::Model::load(backend, dir,
                                            runtime::Model::LoadOptions{
                                                    .budget = {.contextLength = kContext}});
    REQUIRE(modelResult.has_value());
    runtime::Model& model = *modelResult;
    const ArchInfo& arch = model.arch();

    auto cacheResult = runtime::KVCacheManager::create(backend, arch, kContext);
    REQUIRE(cacheResult.has_value());

    auto runnerResult = runtime::ForwardRunner::create(backend, model, *cacheResult);
    REQUIRE(runnerResult.has_value());
    runtime::ForwardRunner& runner = *runnerResult;

    auto residentResult = io::MappedFile::open(dir / gturbo::kResidentFile);
    REQUIRE(residentResult.has_value());
    const HostWeights host{model.manifest(), *residentResult};

    // ---- Drive both sides from the same hidden state ---------------------
    //
    // Layer 0 is sliding window; layer 5 is full attention, which differs in
    // head dimension (512 against 256), KV head count (2 against 8), RoPE type
    // (proportional against default) and has no v_proj at all. Both need
    // covering: a bug in one is invisible in the other.
    const u64 kLayer = GENERATE(u64{0}, u64{5});
    constexpr u64 kPosition = 0;
    INFO("layer " << kLayer << (kLayer % 6 == 5 ? " (full attention)" : " (sliding)"));
    const auto hiddenSize = arch.hiddenSize;

    SplitMix64 rng{9001};
    // Magnitudes near what the embedding actually produces.
    const auto hidden = throughHalf(randomGaussians(rng, static_cast<usize>(hiddenSize), 2.0f));

    REQUIRE(runner.setHiddenState(hidden).has_value());
    REQUIRE(runner.runSingleLayer(kLayer, kPosition).has_value());

    const double eps = arch.rmsNormEps;
    const u64 heads = arch.numHeads;
    const u64 kvHeads = arch.kvHeadsFor(kLayer);
    const u64 headDim = arch.headDimFor(kLayer);
    const u64 queryWidth = arch.qProjOutFeatures(kLayer);
    const u64 kvWidth = arch.kvProjOutFeatures(kLayer);

    const std::string prefix = std::format("layers.{}", kLayer);

    // ---- Stage 1: input norm --------------------------------------------
    const auto inputNorm = host.bf16Tensor(prefix + ".input_layernorm.weight");
    // Rounded through fp16 because that is what the GPU stored and fed to the
    // projections. Without this the reference consumes a more precise input and
    // the comparison measures representation error rather than kernel error -
    // which these dot products amplify badly, since they cancel down to
    // magnitudes far below their individual terms.
    const auto expectedNormed = throughHalf(rmsNorm(hidden, inputNorm, eps));

    // The GPU has since overwritten `normed` with the pre-feedforward norm, so
    // this stage is verified through its consumers below rather than directly.

    // ---- Stage 2: Q projection, per-head norm, RoPE ----------------------
    const auto qDense = host.dense(prefix + ".self_attn.q_proj", arch.qProjLayout(kLayer));
    auto expectedQuery = throughHalf(
            HostWeights::matvec(qDense, expectedNormed, queryWidth, hiddenSize));

    const auto qNorm = host.bf16Tensor(prefix + ".self_attn.q_norm.weight");
    expectedQuery = throughHalf(perHeadNorm(expectedQuery, qNorm, heads, headDim, eps));

    const RopeParams ropeParams{.headDim = headDim,
                                .theta = arch.ropeThetaFor(kLayer),
                                .partialRotaryFactor =
                                        arch.isFullAttention(kLayer)
                                                ? arch.partialRotaryFactor
                                                : 1.0};
    ropeAllHeads(expectedQuery, heads, headDim, kPosition, ropeParams);

    const auto actualQuery = runner.readScratch("query", static_cast<usize>(queryWidth));
    REQUIRE(actualQuery.has_value());

    {
        const auto deviation = compare(expectedQuery, *actualQuery);
        const double versusRms = deviationVersusRms(expectedQuery, *actualQuery);
        INFO("query: : " << deviation.describe() << "  |  vs RMS " << versusRms);
        CHECK(versusRms < kRmsTolerance);
        }

    // ---- Stage 3: K projection, per-head norm, RoPE ----------------------
    const auto kDense = host.dense(prefix + ".self_attn.k_proj", arch.kvProjLayout(kLayer));
    const auto expectedKeyRaw =
            throughHalf(HostWeights::matvec(kDense, expectedNormed, kvWidth, hiddenSize));

    {
        const auto actualKeyRaw = runner.readScratch("keyRaw", static_cast<usize>(kvWidth));
        REQUIRE(actualKeyRaw.has_value());
        const auto deviation = compare(expectedKeyRaw, *actualKeyRaw);
        const double versusRms = deviationVersusRms(expectedKeyRaw, *actualKeyRaw);
        INFO("keyRaw: " << deviation.describe() << "  |  vs RMS " << versusRms);
        CHECK(versusRms < kRmsTolerance);
    }

    const auto kNorm = host.bf16Tensor(prefix + ".self_attn.k_norm.weight");
    auto expectedKey = throughHalf(perHeadNorm(expectedKeyRaw, kNorm, kvHeads, headDim, eps));
    ropeAllHeads(expectedKey, kvHeads, headDim, kPosition, ropeParams);

    {
        const auto actualKey = runner.readScratch("key", static_cast<usize>(kvWidth));
        REQUIRE(actualKey.has_value());
        const auto deviation = compare(expectedKey, *actualKey);
        const double versusRms = deviationVersusRms(expectedKey, *actualKey);
        INFO("key: " << deviation.describe() << "  |  vs RMS " << versusRms);
        CHECK(versusRms < kRmsTolerance);
    }

    // ---- Stage 4: V, which branches off the raw K projection -------------
    std::vector<float> expectedValueSource;
    if (arch.hasSeparateVProj(kLayer)) {
        const auto vDense =
                host.dense(prefix + ".self_attn.v_proj", arch.kvProjLayout(kLayer));
        expectedValueSource =
                throughHalf(HostWeights::matvec(vDense, expectedNormed, kvWidth, hiddenSize));
    } else {
        expectedValueSource = expectedKeyRaw;
    }
    // v_norm carries no learnable weight.
    const auto expectedValue =
            throughHalf(perHeadNorm(expectedValueSource, {}, kvHeads, headDim, eps));

    {
        const auto actualValue = runner.readScratch("value", static_cast<usize>(kvWidth));
        REQUIRE(actualValue.has_value());
        const auto deviation = compare(expectedValue, *actualValue);
        const double versusRms = deviationVersusRms(expectedValue, *actualValue);
        INFO("value: " << deviation.describe() << "  |  vs RMS " << versusRms);
        CHECK(versusRms < kRmsTolerance);
    }

    // ---- Stage 5: attention over a single cached position ----------------
    const AttentionParams attentionParams{.numHeads = heads,
                                          .numKVHeads = kvHeads,
                                          .headDim = headDim,
                                          .scale = 1.0f,
                                          .slidingWindow = arch.isFullAttention(kLayer)
                                                                   ? 0
                                                                   : arch.slidingWindow,
                                          .capacity = 1,
                                          .circular = false};
    const auto expectedAttention =
            decodeAttention(expectedQuery, expectedKey, expectedValue, 1, kPosition,
                            attentionParams);

    {
        const auto actualAttention =
                runner.readScratch("attention", static_cast<usize>(queryWidth));
        REQUIRE(actualAttention.has_value());
        const auto deviation = compare(expectedAttention, *actualAttention);
        const double versusRms = deviationVersusRms(expectedAttention, *actualAttention);
        INFO("attention: " << deviation.describe() << "  |  vs RMS " << versusRms);
        CHECK(versusRms < kRmsTolerance);
    }

    // ---- Stage 6: output projection and the post-attention norm ----------
    const auto oDense = host.dense(prefix + ".self_attn.o_proj", arch.oProjLayout(kLayer));
    auto expectedOut = throughHalf(
            HostWeights::matvec(oDense, expectedAttention, hiddenSize, queryWidth));

    const auto postAttentionNorm =
            host.bf16Tensor(prefix + ".post_attention_layernorm.weight");
    expectedOut = throughHalf(rmsNorm(expectedOut, postAttentionNorm, eps));

    {
        const auto actualOut =
                runner.readScratch("attentionOut", static_cast<usize>(hiddenSize));
        REQUIRE(actualOut.has_value());
        const auto deviation = compare(expectedOut, *actualOut);
        const double versusRms = deviationVersusRms(expectedOut, *actualOut);
        INFO("attentionOut after post-attention norm: " << deviation.describe() << "  |  vs RMS " << versusRms);
        CHECK(versusRms < kRmsTolerance);
    }

    // ---- Stage 7: the residual join, which starts the feed-forward half ---
    std::vector<float> afterAttention(static_cast<usize>(hiddenSize));
    for (usize i = 0; i < afterAttention.size(); ++i) {
        afterAttention[i] = hidden[i] + expectedOut[i];
    }
    afterAttention = throughHalf(afterAttention);

    // ---- Stage 8: the router ---------------------------------------------
    //
    // The router reads the post-attention hidden state directly, not the
    // normed branch input, and its norm weight is router.scale premultiplied
    // by hiddenSize^-0.5.
    const auto routerScale = host.bf16Tensor(prefix + ".router.scale");
    auto foldedScale = foldRouterScale(routerScale, hiddenSize);
    // The loader stores the folded weight as bf16, so the reference has to
    // round it the same way or the comparison charges the kernel for the fold.
    for (auto& value : foldedScale) {
        value = toFloat(toBf16(value));
    }
    const auto expectedRouterNormed =
            throughHalf(rmsNorm(afterAttention, foldedScale, eps));

    {
        const auto actual =
                runner.readScratch("routerNormed", static_cast<usize>(hiddenSize));
        REQUIRE(actual.has_value());
        const double versusRms = deviationVersusRms(expectedRouterNormed, *actual);
        INFO("routerNormed: vs RMS " << versusRms);
        CHECK(versusRms < kRmsTolerance);
    }

    const auto routerDense = host.dense(prefix + ".router.proj", arch.routerLayout());
    const auto expectedScores = throughHalf(HostWeights::matvec(
            routerDense, expectedRouterNormed, arch.numExperts, hiddenSize));

    {
        const auto actual =
                runner.readScratch("routerScores", static_cast<usize>(arch.numExperts));
        REQUIRE(actual.has_value());
        const double versusRms = deviationVersusRms(expectedScores, *actual);
        INFO("routerScores: vs RMS " << versusRms);
        CHECK(versusRms < kRmsTolerance);
    }

    const auto perExpertScale = host.bf16Tensor(prefix + ".router.per_expert_scale");
    const auto expectedRouting =
            reference::routerTopK(expectedScores, perExpertScale, arch.topKExperts);

    {
        // Expert selection must match exactly: routing to different experts is
        // not a rounding difference, it is a different computation.
        const auto actualIndices = runner.lastRouterIndices();
        REQUIRE(actualIndices.size() == expectedRouting.indices.size());
        for (usize k = 0; k < actualIndices.size(); ++k) {
            INFO("selected expert " << k);
            CHECK(actualIndices[k] == expectedRouting.indices[k]);
        }

        const auto actualWeights =
                runner.readScratch("routerWeights", static_cast<usize>(arch.topKExperts));
        REQUIRE(actualWeights.has_value());
        const double versusRms =
                deviationVersusRms(expectedRouting.weights, *actualWeights);
        INFO("routerWeights: vs RMS " << versusRms);
        CHECK(versusRms < kRmsTolerance);
    }

    // ---- Stage 9: the two branch inputs ----------------------------------
    const auto preFeedforward = host.bf16Tensor(prefix + ".pre_feedforward_layernorm.weight");
    const auto preFeedforward2 =
            host.bf16Tensor(prefix + ".pre_feedforward_layernorm_2.weight");

    const auto expectedDenseInput =
            throughHalf(rmsNorm(afterAttention, preFeedforward, eps));
    const auto expectedExpertInput =
            throughHalf(rmsNorm(afterAttention, preFeedforward2, eps));

    {
        const auto actual = runner.readScratch("normed", static_cast<usize>(hiddenSize));
        REQUIRE(actual.has_value());
        const double versusRms = deviationVersusRms(expectedDenseInput, *actual);
        INFO("dense branch input (pre_feedforward_layernorm): vs RMS " << versusRms);
        CHECK(versusRms < kRmsTolerance);
    }
    {
        const auto actual = runner.readScratch("branchInput", static_cast<usize>(hiddenSize));
        REQUIRE(actual.has_value());
        const double versusRms = deviationVersusRms(expectedExpertInput, *actual);
        INFO("expert branch input (pre_feedforward_layernorm_2): vs RMS " << versusRms);
        CHECK(versusRms < kRmsTolerance);
    }

    // ---- Stage 10: the dense expert --------------------------------------
    const auto gateDense = host.dense(prefix + ".mlp.gate_proj", arch.sharedGateUpLayout());
    const auto upDense = host.dense(prefix + ".mlp.up_proj", arch.sharedGateUpLayout());
    const auto downDense = host.dense(prefix + ".mlp.down_proj", arch.sharedDownLayout());

    const auto expectedGate = throughHalf(HostWeights::matvec(
            gateDense, expectedDenseInput, arch.intermediateSize, hiddenSize));
    const auto expectedUp = throughHalf(HostWeights::matvec(
            upDense, expectedDenseInput, arch.intermediateSize, hiddenSize));

    {
        const auto actualGate =
                runner.readScratch("denseGate", static_cast<usize>(arch.intermediateSize));
        REQUIRE(actualGate.has_value());
        const double versusRms = deviationVersusRms(expectedGate, *actualGate);
        INFO("dense gate projection: vs RMS " << versusRms);
        CHECK(versusRms < kRmsTolerance);
    }
    {
        const auto actualUp =
                runner.readScratch("denseUp", static_cast<usize>(arch.intermediateSize));
        REQUIRE(actualUp.has_value());
        const double versusRms = deviationVersusRms(expectedUp, *actualUp);
        INFO("dense up projection: vs RMS " << versusRms);
        CHECK(versusRms < kRmsTolerance);
    }

    // The gate is what passes through GELU.
    const auto expectedActivated = throughHalf(geglu(expectedGate, expectedUp));

    {
        const auto actual =
                runner.readScratch("denseAct", static_cast<usize>(arch.intermediateSize));
        REQUIRE(actual.has_value());
        const double versusRms = deviationVersusRms(expectedActivated, *actual);
        INFO("dense GeGLU: vs RMS " << versusRms);
        CHECK(versusRms < kRmsTolerance);
    }

    auto expectedShared = throughHalf(HostWeights::matvec(downDense, expectedActivated,
                                                          hiddenSize,
                                                          arch.intermediateSize));
    const auto postFeedforward1 =
            host.bf16Tensor(prefix + ".post_feedforward_layernorm_1.weight");
    expectedShared = throughHalf(rmsNorm(expectedShared, postFeedforward1, eps));

    {
        // sharedBranch is post-normed in place, so this compares the branch
        // after post_feedforward_layernorm_1.
        const auto actual =
                runner.readScratch("sharedBranch", static_cast<usize>(hiddenSize));
        REQUIRE(actual.has_value());
        const double versusRms = deviationVersusRms(expectedShared, *actual);
        INFO("dense branch after post_feedforward_layernorm_1: vs RMS " << versusRms);
        CHECK(versusRms < kRmsTolerance);
    }

    // ---- Stage 11: the routed experts ------------------------------------
    auto expertFileResult = io::MappedFile::open(
            dir / gturbo::kExpertsDir /
            model.manifest().experts.layerFiles[static_cast<usize>(kLayer)]);
    REQUIRE(expertFileResult.has_value());

    const auto& expertLayout = model.manifest().experts;
    const auto gateUpShape = arch.expertGateUpLayout();
    const auto downShape = arch.expertDownLayout();

    std::vector<float> expertOutputs(
            static_cast<usize>(arch.topKExperts * hiddenSize), 0.0f);

    for (u64 k = 0; k < arch.topKExperts; ++k) {
        const u32 expert = expectedRouting.indices[static_cast<usize>(k)];

        const auto expertGateDense = HostWeights::expertDense(
                *expertFileResult, expertLayout, kLayer, expert, "gate", gateUpShape);
        const auto expertUpDense = HostWeights::expertDense(
                *expertFileResult, expertLayout, kLayer, expert, "up", gateUpShape);
        const auto expertDownDense = HostWeights::expertDense(
                *expertFileResult, expertLayout, kLayer, expert, "down", downShape);

        const auto gateExact = HostWeights::matvec(expertGateDense, expectedExpertInput,
                                                    arch.moeIntermediateSize, hiddenSize);
        const auto upExact = HostWeights::matvec(expertUpDense, expectedExpertInput,
                                                  arch.moeIntermediateSize, hiddenSize);

        if (k == 0) {
            const auto rmsOf = [](std::span<const float> values) {
                double sum = 0.0;
                for (const float value : values) {
                    sum += static_cast<double>(value) * value;
                }
                return std::sqrt(sum / static_cast<double>(values.size()));
            };
            const auto peak = [](std::span<const float> values) {
                double worst = 0.0;
                for (const float value : values) {
                    worst = std::max(worst, std::abs(static_cast<double>(value)));
                }
                return worst;
            };
            // fp16 saturates at 65504, so anything approaching that overflows
            // once GeGLU multiplies the two projections together.
            WARN(std::format(
                    "layer {} expert input rms {:.3f} peak {:.3f} | gate rms {:.1f} peak "
                    "{:.1f} | up rms {:.1f} peak {:.1f} | product peak {:.1f}",
                    kLayer, rmsOf(expectedExpertInput), peak(expectedExpertInput),
                    rmsOf(gateExact), peak(gateExact), rmsOf(upExact), peak(upExact),
                    peak(gateExact) * peak(upExact)));
        }

        const auto gate = throughHalf(gateExact);
        const auto up = throughHalf(upExact);
        const auto activated = throughHalf(geglu(gate, up));
        const auto out = throughHalf(HostWeights::matvec(expertDownDense, activated,
                                                          hiddenSize,
                                                          arch.moeIntermediateSize));

        std::ranges::copy(out, expertOutputs.begin() + static_cast<isize>(k * hiddenSize));
    }

    {
        const auto actual = runner.readScratch(
                "expertOutputs", static_cast<usize>(arch.topKExperts * hiddenSize));
        REQUIRE(actual.has_value());

        // Per expert, so a disagreement points at which selection is wrong
        // rather than at the block as a whole.
        for (u64 k = 0; k < arch.topKExperts; ++k) {
            const auto begin = static_cast<usize>(k * hiddenSize);
            const auto expectedSlice =
                    std::span{expertOutputs}.subspan(begin, static_cast<usize>(hiddenSize));
            const auto actualSlice =
                    std::span{*actual}.subspan(begin, static_cast<usize>(hiddenSize));
            const double versusRms = deviationVersusRms(expectedSlice, actualSlice);
            const auto deviation = compare(expectedSlice, actualSlice);
            INFO("expert slot " << k << " (id " << expectedRouting.indices[static_cast<usize>(k)]
                                << "): vs RMS " << versusRms << "  |  " << deviation.describe());
            CHECK(versusRms < kRmsTolerance);
        }

        // Also across the whole block, which is what would catch an expert
        // written to the wrong slot: each slice can look right individually
        // while the concatenation is permuted.
        std::string magnitudes;
        for (u64 k = 0; k < arch.topKExperts; ++k) {
            const auto begin = static_cast<usize>(k * hiddenSize);
            double referenceSum = 0.0;
            double actualSum = 0.0;
            for (u64 i = 0; i < hiddenSize; ++i) {
                const double r = expertOutputs[begin + static_cast<usize>(i)];
                const double a = (*actual)[begin + static_cast<usize>(i)];
                referenceSum += r * r;
                actualSum += a * a;
            }
            magnitudes += std::format(
                    "\n    slot {} id {:3}: reference rms {:.6f}, actual rms {:.6f}", k,
                    expectedRouting.indices[static_cast<usize>(k)],
                    std::sqrt(referenceSum / static_cast<double>(hiddenSize)),
                    std::sqrt(actualSum / static_cast<double>(hiddenSize)));
        }

        const double aggregate = deviationVersusRms(expertOutputs, *actual);
        INFO("routed expert outputs, all slots: vs RMS " << aggregate << magnitudes);
        CHECK(aggregate < kRmsTolerance);
    }

    auto expectedRouted = throughHalf(reference::moeCombine(
            expertOutputs, expectedRouting.weights, arch.topKExperts, hiddenSize));

    const auto postFeedforward2 =
            host.bf16Tensor(prefix + ".post_feedforward_layernorm_2.weight");
    expectedRouted = throughHalf(rmsNorm(expectedRouted, postFeedforward2, eps));

    {
        const auto actual =
                runner.readScratch("routedBranch", static_cast<usize>(hiddenSize));
        REQUIRE(actual.has_value());
        const double versusRms = deviationVersusRms(expectedRouted, *actual);
        INFO("routed branch after post_feedforward_layernorm_2: vs RMS " << versusRms);
        CHECK(versusRms < kRmsTolerance);
    }

    // ---- Stage 12: the layer output --------------------------------------
    std::vector<float> combined(static_cast<usize>(hiddenSize));
    for (usize i = 0; i < combined.size(); ++i) {
        combined[i] = expectedShared[i] + expectedRouted[i];
    }
    combined = throughHalf(combined);

    const auto postFeedforward =
            host.bf16Tensor(prefix + ".post_feedforward_layernorm.weight");
    combined = throughHalf(rmsNorm(combined, postFeedforward, eps));

    // The residual join, then the layer scalar over the whole thing.
    const float layerScalar = model.layer(kLayer).layerScalar;
    std::vector<float> expectedLayerOutput(static_cast<usize>(hiddenSize));
    for (usize i = 0; i < expectedLayerOutput.size(); ++i) {
        expectedLayerOutput[i] = (afterAttention[i] + combined[i]) * layerScalar;
    }
    expectedLayerOutput = throughHalf(expectedLayerOutput);

    {
        const auto actual = runner.readHiddenState();
        REQUIRE(actual.has_value());
        const double versusRms = deviationVersusRms(expectedLayerOutput, *actual);
        INFO("layer output: vs RMS " << versusRms);
        CHECK(versusRms < kRmsTolerance);
    }
}
