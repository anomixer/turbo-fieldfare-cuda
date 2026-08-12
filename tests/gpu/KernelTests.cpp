// Every CUDA kernel checked against its scalar CPU reference.
//
// Inputs are rounded through fp16/bf16 before both sides run, so what is being
// measured is the kernel's own error rather than the cost of representing the
// test data. Tolerances come from src/reference/Tolerance.h, where each budget
// reflects the accumulation the operation actually performs.
//
// Skipped when no GPU is available.

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <chrono>
#include <span>
#include <vector>

#include "tf/core/math/Float.h"
#include "Harness.h"
#include "tf/gpu/Backend.h"
#include "tf/gpu/Kernels.h"
#include "tf/reference/Ops.h"
#include "tf/reference/Prng.h"
#include "tf/reference/Tolerance.h"

using namespace tf;
using namespace tf::gpu;
using namespace tf::reference;

namespace {


/// Activations are fp32 on both sides now, so this is an identity. Kept as a
/// named step because the reference and the kernel must always be fed the same
/// representation, and that is easy to forget when a storage type changes.
[[nodiscard]] std::vector<float> throughHalf(std::span<const float> values) {
    return {values.begin(), values.end()};
}

[[nodiscard]] std::vector<float> throughBf16(std::span<const float> values) {
    std::vector<float> out(values.size());
    for (usize i = 0; i < values.size(); ++i) {
        out[i] = toFloat(toBf16(values[i]));
    }
    return out;
}

class Device {
public:
    Device(IGpuBackend& backend, Stream& stream) : backend_(backend), stream_(stream) {}

    // Named with an "upload" prefix so no member shadows the tf::bf16 and
    // tf::fp16 types inside this class scope.
    [[nodiscard]] BufferPtr uploadHalf(std::span<const float> values, std::string name) {
        // Named "half" for continuity, but activations are fp32: Gemma 4's do
        // not fit in fp16.
        return upload(ByteSpan{reinterpret_cast<const u8*>(values.data()),
                               values.size() * sizeof(float)},
                      std::move(name));
    }

    [[nodiscard]] BufferPtr uploadBf16(std::span<const float> values, std::string name) {
        const auto encoded = toBf16Buffer(values);
        return upload(ByteSpan{reinterpret_cast<const u8*>(encoded.data()),
                               encoded.size() * sizeof(bf16)},
                      std::move(name));
    }

    [[nodiscard]] BufferPtr uploadWords(std::span<const u32> values, std::string name) {
        return upload(ByteSpan{reinterpret_cast<const u8*>(values.data()),
                               values.size() * sizeof(u32)},
                      std::move(name));
    }

    [[nodiscard]] BufferPtr uploadBf16Raw(std::span<const bf16> values, std::string name) {
        return upload(ByteSpan{reinterpret_cast<const u8*>(values.data()),
                               values.size() * sizeof(bf16)},
                      std::move(name));
    }

    [[nodiscard]] BufferPtr empty(u64 bytes, std::string name) {
        auto buffer = backend_.allocate(MemoryKind::Device, bytes, std::move(name));
        REQUIRE(buffer.has_value());
        return std::move(*buffer);
    }

    [[nodiscard]] std::vector<float> readHalf(Buffer& buffer, usize count) {
        std::vector<float> values(count);
        REQUIRE(backend_
                        .enqueueDownload(stream_,
                                         MutableByteSpan{reinterpret_cast<u8*>(values.data()),
                                                         count * sizeof(float)},
                                         buffer, 0)
                        .has_value());
        REQUIRE(stream_.synchronize().has_value());
        return values;
    }

    [[nodiscard]] u32 readWord(Buffer& buffer) {
        u32 value = 0;
        REQUIRE(backend_
                        .enqueueDownload(
                                stream_,
                                MutableByteSpan{reinterpret_cast<u8*>(&value), sizeof(value)},
                                buffer, 0)
                        .has_value());
        REQUIRE(stream_.synchronize().has_value());
        return value;
    }

private:
    [[nodiscard]] BufferPtr upload(ByteSpan bytes, std::string name) {
        auto buffer = backend_.allocate(MemoryKind::Device, bytes.size(), std::move(name));
        REQUIRE(buffer.has_value());
        REQUIRE(backend_.enqueueUpload(stream_, **buffer, 0, bytes).has_value());
        REQUIRE(stream_.synchronize().has_value());
        return std::move(*buffer);
    }

    IGpuBackend& backend_;
    Stream& stream_;
};

DeviceView view(const BufferPtr& buffer) {
    return DeviceView{.buffer = buffer.get(), .offset = 0};
}

}  // namespace

// ---------------------------------------------------------------------------
// RMSNorm
// ---------------------------------------------------------------------------

TEST_CASE("rmsNorm matches the reference", "[kernels]") {
    REQUIRE_GPU();
    Device device{backend, stream};

    // The real hidden size, so the block reduction runs at its production shape.
    constexpr u32 kCount = 2816;
    SplitMix64 rng{101};

    const auto input = throughHalf(randomGaussians(rng, kCount, 3.0f));
    // Norm weights in this checkpoint sit well above 1; mirror that.
    const auto weight = throughBf16(randomFloats(rng, kCount, 0.5f, 8.0f));

    auto inputBuffer = device.uploadHalf(input, "input");
    auto weightBuffer = device.uploadBf16(weight, "weight");
    auto outputBuffer = device.empty(kCount * sizeof(float), "output");

    const auto status = kernels.rmsNorm(stream, RmsNormArgs{.input = view(inputBuffer),
                                                            .weight = view(weightBuffer),
                                                            .output = view(outputBuffer),
                                                            .rows = 1,
                                                            .count = kCount,
                                                            .eps = 1e-6f});
    INFO((status ? std::string{} : status.error().toString()));
    REQUIRE(status.has_value());

    const auto actual = device.readHalf(*outputBuffer, kCount);
    const auto expected = rmsNorm(input, weight, 1e-6);

    const auto deviation = compare(expected, actual);
    INFO(deviation.describe());
    CHECK(deviation.maxRelative < tolerance::kNorm);
}

TEST_CASE("rmsNorm without a weight is the no-scale form", "[kernels]") {
    REQUIRE_GPU();
    Device device{backend, stream};

    // v_norm: an RMS normalization with no learnable weight at all.
    constexpr u32 kCount = 256;
    SplitMix64 rng{102};
    const auto input = throughHalf(randomGaussians(rng, kCount, 2.0f));

    auto inputBuffer = device.uploadHalf(input, "input");
    auto outputBuffer = device.empty(kCount * sizeof(float), "output");

    REQUIRE(kernels
                    .rmsNorm(stream, RmsNormArgs{.input = view(inputBuffer),
                                                 .weight = {},  // deliberately unset
                                                 .output = view(outputBuffer),
                                                 .rows = 1,
                                                 .count = kCount,
                                                 .eps = 1e-6f})
                    .has_value());

    const auto deviation =
            compare(rmsNormNoScale(input, 1e-6), device.readHalf(*outputBuffer, kCount));
    INFO(deviation.describe());
    CHECK(deviation.maxRelative < tolerance::kNorm);
}

TEST_CASE("rmsNorm normalizes each row independently", "[kernels]") {
    REQUIRE_GPU();
    Device device{backend, stream};

    // q_norm and k_norm run per head: 16 heads of 256, each with its own RMS.
    constexpr u32 kRows = 16;
    constexpr u32 kCount = 256;
    SplitMix64 rng{103};

    std::vector<float> input;
    input.reserve(kRows * kCount);
    for (u32 row = 0; row < kRows; ++row) {
        // Wildly different magnitudes per row, so a kernel that normalized
        // across the whole buffer instead of per row could not pass.
        const auto scale = static_cast<float>(1 << row) * 0.001f;
        const auto rowValues = randomGaussians(rng, kCount, scale);
        input.insert(input.end(), rowValues.begin(), rowValues.end());
    }
    input = throughHalf(input);
    const auto weight = throughBf16(randomFloats(rng, kCount, 0.8f, 1.2f));

    auto inputBuffer = device.uploadHalf(input, "input");
    auto weightBuffer = device.uploadBf16(weight, "weight");
    auto outputBuffer = device.empty(kRows * kCount * sizeof(float), "output");

    REQUIRE(kernels
                    .rmsNorm(stream, RmsNormArgs{.input = view(inputBuffer),
                                                 .weight = view(weightBuffer),
                                                 .output = view(outputBuffer),
                                                 .rows = kRows,
                                                 .count = kCount,
                                                 .eps = 1e-6f})
                    .has_value());

    const auto actual = device.readHalf(*outputBuffer, kRows * kCount);

    for (u32 row = 0; row < kRows; ++row) {
        const auto begin = static_cast<usize>(row) * kCount;
        const auto expected =
                rmsNorm(std::span{input}.subspan(begin, kCount), weight, 1e-6);
        const auto deviation =
                compare(expected, std::span{actual}.subspan(begin, kCount));
        INFO("row " << row << ": " << deviation.describe());
        REQUIRE(deviation.maxRelative < tolerance::kNorm);
    }
}

// ---------------------------------------------------------------------------
// Elementwise
// ---------------------------------------------------------------------------

TEST_CASE("geglu matches the reference", "[kernels]") {
    REQUIRE_GPU();
    Device device{backend, stream};

    constexpr u32 kCount = 2112;  // the dense expert width
    SplitMix64 rng{104};
    const auto gate = throughHalf(randomGaussians(rng, kCount, 2.0f));
    const auto up = throughHalf(randomGaussians(rng, kCount, 2.0f));

    auto gateBuffer = device.uploadHalf(gate, "gate");
    auto upBuffer = device.uploadHalf(up, "up");
    auto outputBuffer = device.empty(kCount * sizeof(float), "output");

    REQUIRE(kernels
                    .geglu(stream, GegluArgs{.gate = view(gateBuffer),
                                             .up = view(upBuffer),
                                             .output = view(outputBuffer),
                                             .count = kCount})
                    .has_value());

    const auto deviation =
            compare(geglu(gate, up), device.readHalf(*outputBuffer, kCount));
    INFO(deviation.describe());
    CHECK(deviation.maxRelative < tolerance::kElementwise);
}

TEST_CASE("geglu activates the gate, not the up projection", "[kernels]") {
    REQUIRE_GPU();
    Device device{backend, stream};

    // Asymmetric inputs, so swapping the roles gives a visibly different answer.
    const std::vector<float> gate{1.0f, -1.0f, 2.0f, -2.0f};
    const std::vector<float> up{3.0f, 3.0f, 3.0f, 3.0f};

    auto gateBuffer = device.uploadHalf(gate, "gate");
    auto upBuffer = device.uploadHalf(up, "up");
    auto outputBuffer = device.empty(gate.size() * sizeof(float), "output");

    REQUIRE(kernels
                    .geglu(stream, GegluArgs{.gate = view(gateBuffer),
                                             .up = view(upBuffer),
                                             .output = view(outputBuffer),
                                             .count = 4})
                    .has_value());

    const auto actual = device.readHalf(*outputBuffer, 4);
    const auto correct = geglu(gate, up);
    const auto swapped = geglu(up, gate);

    CHECK(compare(correct, actual).maxRelative < tolerance::kElementwise);
    // The reversed form is genuinely different, so this test can detect it.
    CHECK(compare(swapped, actual).maxRelative > 0.1);
}

TEST_CASE("add and scale match the reference", "[kernels]") {
    REQUIRE_GPU();
    Device device{backend, stream};

    constexpr u32 kCount = 2816;
    SplitMix64 rng{105};
    const auto a = throughHalf(randomGaussians(rng, kCount));
    const auto b = throughHalf(randomGaussians(rng, kCount));

    auto aBuffer = device.uploadHalf(a, "a");
    auto bBuffer = device.uploadHalf(b, "b");
    auto sumBuffer = device.empty(kCount * sizeof(float), "sum");

    REQUIRE(kernels
                    .add(stream, AddArgs{.a = view(aBuffer),
                                         .b = view(bBuffer),
                                         .output = view(sumBuffer),
                                         .count = kCount})
                    .has_value());

    std::vector<float> expectedSum(kCount);
    for (usize i = 0; i < kCount; ++i) {
        expectedSum[i] = a[i] + b[i];
    }
    CHECK(compare(expectedSum, device.readHalf(*sumBuffer, kCount)).maxRelative <
          tolerance::kElementwise);

    // layer_scalar for layer 0 of the real checkpoint.
    constexpr float kLayerScalar = 0.0703125f;
    auto scaledBuffer = device.empty(kCount * sizeof(float), "scaled");
    REQUIRE(kernels
                    .scale(stream, ScaleArgs{.input = view(aBuffer),
                                             .output = view(scaledBuffer),
                                             .count = kCount,
                                             .scalar = kLayerScalar})
                    .has_value());

    std::vector<float> expectedScaled(kCount);
    for (usize i = 0; i < kCount; ++i) {
        expectedScaled[i] = a[i] * kLayerScalar;
    }
    CHECK(compare(expectedScaled, device.readHalf(*scaledBuffer, kCount)).maxRelative <
          tolerance::kElementwise);
}

TEST_CASE("logit softcap matches the reference", "[kernels]") {
    REQUIRE_GPU();
    Device device{backend, stream};

    constexpr u32 kCount = 4096;
    SplitMix64 rng{106};
    // Spread well past the cap so the saturating region is exercised.
    const auto logits = throughHalf(randomGaussians(rng, kCount, 40.0f));

    auto inputBuffer = device.uploadHalf(logits, "logits");
    auto outputBuffer = device.empty(kCount * sizeof(float), "capped");

    REQUIRE(kernels
                    .logitSoftcap(stream, LogitSoftcapArgs{.input = view(inputBuffer),
                                                           .output = view(outputBuffer),
                                                           .count = kCount,
                                                           .cap = 30.0f})
                    .has_value());

    const auto actual = device.readHalf(*outputBuffer, kCount);
    const auto deviation = compare(logitSoftcap(logits, 30.0f), actual);
    INFO(deviation.describe());
    CHECK(deviation.maxRelative < tolerance::kElementwise);

    for (const float value : actual) {
        REQUIRE(value <= 30.01f);
        REQUIRE(value >= -30.01f);
    }
}

// ---------------------------------------------------------------------------
// Argmax
// ---------------------------------------------------------------------------

TEST_CASE("argmax finds the maximum over the full vocabulary", "[kernels]") {
    REQUIRE_GPU();
    Device device{backend, stream};

    constexpr u32 kVocab = 262144;
    SplitMix64 rng{107};
    auto logits = throughHalf(randomGaussians(rng, kVocab, 5.0f));

    // Plant an unambiguous maximum at an awkward index.
    constexpr u32 kExpected = 197531;
    logits[kExpected] = 100.0f;

    auto inputBuffer = device.uploadHalf(logits, "logits");
    auto outputBuffer = device.empty(sizeof(u32), "argmax");

    REQUIRE(kernels
                    .argmax(stream, ArgmaxArgs{.input = view(inputBuffer),
                                               .output = view(outputBuffer),
                                               .count = kVocab})
                    .has_value());

    CHECK(device.readWord(*outputBuffer) == kExpected);
    CHECK(device.readWord(*outputBuffer) == reference::argmax(logits));
}

TEST_CASE("argmax breaks ties toward the lowest index", "[kernels]") {
    REQUIRE_GPU();
    Device device{backend, stream};

    // Greedy decoding must be reproducible, which depends on this.
    constexpr u32 kCount = 4096;
    std::vector<float> logits(kCount, 1.0f);

    auto inputBuffer = device.uploadHalf(logits, "logits");
    auto outputBuffer = device.empty(sizeof(u32), "argmax");

    REQUIRE(kernels
                    .argmax(stream, ArgmaxArgs{.input = view(inputBuffer),
                                               .output = view(outputBuffer),
                                               .count = kCount})
                    .has_value());

    CHECK(device.readWord(*outputBuffer) == 0);
}

// ---------------------------------------------------------------------------
// RoPE
// ---------------------------------------------------------------------------

TEST_CASE("rope matches the reference on a sliding layer", "[kernels]") {
    REQUIRE_GPU();
    Device device{backend, stream};

    // Sliding layers: 16 heads of 256, theta 10000, full rotation.
    constexpr u32 kHeads = 16;
    constexpr u32 kHeadDim = 256;
    constexpr u64 kPosition = 1337;

    SplitMix64 rng{108};
    const auto input = throughHalf(randomGaussians(rng, kHeads * kHeadDim));

    auto buffer = device.uploadHalf(input, "heads");
    REQUIRE(kernels
                    .rope(stream, RopeArgs{.data = view(buffer),
                                           .heads = kHeads,
                                           .headDim = kHeadDim,
                                           .position = kPosition,
                                           .theta = 10000.0f,
                                           .partialRotaryFactor = 1.0f})
                    .has_value());

    const auto actual = device.readHalf(*buffer, kHeads * kHeadDim);

    std::vector<float> expected = input;
    for (u32 head = 0; head < kHeads; ++head) {
        applyRope(std::span{expected}.subspan(static_cast<usize>(head) * kHeadDim, kHeadDim),
                  kPosition, RopeParams{.headDim = kHeadDim, .theta = 10000.0});
    }

    const auto deviation = compare(expected, actual);
    INFO(deviation.describe());
    CHECK(deviation.maxAbsolute < 0.01);
}

TEST_CASE("rope leaves the tail alone on a full-attention layer", "[kernels]") {
    REQUIRE_GPU();
    Device device{backend, stream};

    // Full-attention layers: head dim 512, theta 1e6, only the first quarter
    // rotated.
    constexpr u32 kHeads = 2;
    constexpr u32 kHeadDim = 512;
    constexpr u64 kPosition = 4095;

    SplitMix64 rng{109};
    const auto input = throughHalf(randomGaussians(rng, kHeads * kHeadDim));

    auto buffer = device.uploadHalf(input, "heads");
    REQUIRE(kernels
                    .rope(stream, RopeArgs{.data = view(buffer),
                                           .heads = kHeads,
                                           .headDim = kHeadDim,
                                           .position = kPosition,
                                           .theta = 1000000.0f,
                                           .partialRotaryFactor = 0.25f})
                    .has_value());

    const auto actual = device.readHalf(*buffer, kHeads * kHeadDim);

    std::vector<float> expected = input;
    for (u32 head = 0; head < kHeads; ++head) {
        applyRope(std::span{expected}.subspan(static_cast<usize>(head) * kHeadDim, kHeadDim),
                  kPosition,
                  RopeParams{.headDim = kHeadDim,
                             .theta = 1000000.0,
                             .partialRotaryFactor = 0.25});
    }

    const auto deviation = compare(expected, actual);
    INFO(deviation.describe());
    CHECK(deviation.maxAbsolute < 0.01);

    // Pair j joins j and j + 256, and only pairs 0..63 turn, so exactly two
    // bands move: 0..63 and 256..319. Everything else must be byte-identical.
    for (u32 head = 0; head < kHeads; ++head) {
        for (u32 i = 0; i < kHeadDim; ++i) {
            const usize index = static_cast<usize>(head) * kHeadDim + i;
            const bool shouldMove = (i < 64) || (i >= 256 && i < 320);
            INFO("head " << head << " dim " << i);
            if (!shouldMove) {
                REQUIRE(actual[index] == input[index]);
            }
        }
    }
}

TEST_CASE("rope at position zero is the identity", "[kernels]") {
    REQUIRE_GPU();
    Device device{backend, stream};

    SplitMix64 rng{110};
    const auto input = throughHalf(randomGaussians(rng, 256));

    auto buffer = device.uploadHalf(input, "heads");
    REQUIRE(kernels
                    .rope(stream, RopeArgs{.data = view(buffer),
                                           .heads = 1,
                                           .headDim = 256,
                                           .position = 0,
                                           .theta = 10000.0f})
                    .has_value());

    CHECK(compare(input, device.readHalf(*buffer, 256)).maxAbsolute < 1e-3);
}

// ---------------------------------------------------------------------------
// Quantized GEMV
// ---------------------------------------------------------------------------

TEST_CASE("4-bit dequant GEMV matches the reference", "[kernels]") {
    REQUIRE_GPU();
    Device device{backend, stream};

    // A routed expert's gate projection at its real shape.
    const QuantizedLinearLayout layout{
            .outFeatures = 704, .inFeatures = 2816, .spec = kWeightQuant};

    SplitMix64 rng{111};
    const auto weights =
            randomGaussians(rng, static_cast<usize>(layout.outFeatures * layout.inFeatures),
                            0.05f);
    const auto quantized = quantizeMatrix(weights, layout);
    const auto input = throughHalf(randomGaussians(rng, static_cast<usize>(layout.inFeatures)));

    auto packedBuffer = device.uploadWords(quantized.packed, "packed");
    auto scaleBuffer = device.uploadBf16Raw(quantized.scales, "scales");
    auto biasBuffer = device.uploadBf16Raw(quantized.biases, "biases");
    auto inputBuffer = device.uploadHalf(input, "input");
    auto outputBuffer = device.empty(layout.outFeatures * sizeof(float), "output");

    REQUIRE(kernels
                    .dequantGemv(stream,
                                 DequantGemvArgs{
                                         .weights = {.packed = view(packedBuffer),
                                                     .scales = view(scaleBuffer),
                                                     .biases = view(biasBuffer),
                                                     .layout = layout},
                                         .input = view(inputBuffer),
                                         .output = view(outputBuffer)})
                    .has_value());

    const auto expected = dequantGemv(quantized.packed, quantized.scales, quantized.biases,
                                      input, layout);
    const auto actual =
            device.readHalf(*outputBuffer, static_cast<usize>(layout.outFeatures));

    const auto deviation = compare(expected, actual);
    INFO(deviation.describe());
    // An fp16 output over a 2816-term accumulation; the reference keeps double.
    CHECK(deviation.maxRelative < tolerance::kGemv);
}

TEST_CASE("8-bit dequant GEMV matches the reference", "[kernels]") {
    REQUIRE_GPU();
    Device device{backend, stream};

    // The router projection: 8-bit, 128 experts out of hidden 2816.
    const QuantizedLinearLayout layout{
            .outFeatures = 128, .inFeatures = 2816, .spec = kRouterQuant};

    SplitMix64 rng{112};
    const auto weights =
            randomGaussians(rng, static_cast<usize>(layout.outFeatures * layout.inFeatures),
                            0.02f);
    const auto quantized = quantizeMatrix(weights, layout);
    const auto input = throughHalf(randomGaussians(rng, static_cast<usize>(layout.inFeatures)));

    auto packedBuffer = device.uploadWords(quantized.packed, "packed");
    auto scaleBuffer = device.uploadBf16Raw(quantized.scales, "scales");
    auto biasBuffer = device.uploadBf16Raw(quantized.biases, "biases");
    auto inputBuffer = device.uploadHalf(input, "input");
    auto outputBuffer = device.empty(layout.outFeatures * sizeof(float), "output");

    REQUIRE(kernels
                    .dequantGemv(stream,
                                 DequantGemvArgs{
                                         .weights = {.packed = view(packedBuffer),
                                                     .scales = view(scaleBuffer),
                                                     .biases = view(biasBuffer),
                                                     .layout = layout},
                                         .input = view(inputBuffer),
                                         .output = view(outputBuffer)})
                    .has_value());

    const auto expected = dequantGemv(quantized.packed, quantized.scales, quantized.biases,
                                      input, layout);
    const auto deviation = compare(
            expected, device.readHalf(*outputBuffer, static_cast<usize>(layout.outFeatures)));
    INFO(deviation.describe());
    CHECK(deviation.maxRelative < tolerance::kGemv);
}

TEST_CASE("dequant GEMV handles negative scales", "[kernels]") {
    REQUIRE_GPU();
    Device device{backend, stream};

    // Real MLX weights carry negative scales, so this is not hypothetical.
    const QuantizedLinearLayout layout{
            .outFeatures = 8, .inFeatures = 128, .spec = kWeightQuant};

    std::vector<u32> packed(static_cast<usize>(layout.outFeatures * layout.packedWordsPerRow()));
    for (usize i = 0; i < packed.size(); ++i) {
        packed[i] = static_cast<u32>(i * 2654435761u);
    }

    const auto groups = static_cast<usize>(layout.outFeatures * layout.groupsPerRow());
    std::vector<bf16> scales(groups);
    std::vector<bf16> biases(groups);
    for (usize i = 0; i < groups; ++i) {
        // Alternate sign so both paths are covered in one launch.
        scales[i] = toBf16((i % 2 == 0) ? 0.011779785f : -0.011779785f);
        biases[i] = toBf16(-0.06005859f);
    }

    SplitMix64 rng{113};
    const auto input = throughHalf(randomGaussians(rng, static_cast<usize>(layout.inFeatures)));

    auto packedBuffer = device.uploadWords(packed, "packed");
    auto scaleBuffer = device.uploadBf16Raw(scales, "scales");
    auto biasBuffer = device.uploadBf16Raw(biases, "biases");
    auto inputBuffer = device.uploadHalf(input, "input");
    auto outputBuffer = device.empty(layout.outFeatures * sizeof(float), "output");

    REQUIRE(kernels
                    .dequantGemv(stream,
                                 DequantGemvArgs{
                                         .weights = {.packed = view(packedBuffer),
                                                     .scales = view(scaleBuffer),
                                                     .biases = view(biasBuffer),
                                                     .layout = layout},
                                         .input = view(inputBuffer),
                                         .output = view(outputBuffer)})
                    .has_value());

    const auto expected = dequantGemv(packed, scales, biases, input, layout);
    const auto deviation = compare(
            expected, device.readHalf(*outputBuffer, static_cast<usize>(layout.outFeatures)));
    INFO(deviation.describe());
    CHECK(deviation.maxRelative < tolerance::kGemv);
}

// ---------------------------------------------------------------------------
// Embedding lookup
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Batched GEMM
// ---------------------------------------------------------------------------

namespace {

/// Runs the batched GEMM over `tokens` and checks every token against the GEMV
/// reference applied to that token's row.
///
/// Token counts either side of the kernel's tile matter more than the values
/// do: the interesting failures are a partial tile writing past its bounds, and
/// out-of-range rows skipping a barrier the rest of the block waits on.
void checkGemm(IGpuBackend& backend, IKernels& kernels, Stream& stream,
               const QuantizedLinearLayout& layout, u32 tokens, u64 seed) {
    Device device{backend, stream};

    SplitMix64 rng{seed};
    const auto weights = randomGaussians(
            rng, static_cast<usize>(layout.outFeatures * layout.inFeatures), 0.05f);
    const auto quantized = quantizeMatrix(weights, layout);
    const auto input = throughHalf(
            randomGaussians(rng, static_cast<usize>(tokens) *
                                         static_cast<usize>(layout.inFeatures)));

    auto packedBuffer = device.uploadWords(quantized.packed, "packed");
    auto scaleBuffer = device.uploadBf16Raw(quantized.scales, "scales");
    auto biasBuffer = device.uploadBf16Raw(quantized.biases, "biases");
    auto inputBuffer = device.uploadHalf(input, "input");
    auto outputBuffer =
            device.empty(u64{tokens} * layout.outFeatures * sizeof(float), "output");

    REQUIRE(kernels
                    .dequantGemm(stream,
                                 DequantGemmArgs{
                                         .weights = {.packed = view(packedBuffer),
                                                     .scales = view(scaleBuffer),
                                                     .biases = view(biasBuffer),
                                                     .layout = layout},
                                         .input = view(inputBuffer),
                                         .output = view(outputBuffer),
                                         .tokens = tokens})
                    .has_value());

    const auto actual = device.readHalf(
            *outputBuffer, static_cast<usize>(tokens) * layout.outFeatures);

    for (u32 token = 0; token < tokens; ++token) {
        const auto row = std::span<const float>{input}.subspan(
                static_cast<usize>(token) * layout.inFeatures,
                static_cast<usize>(layout.inFeatures));
        const auto expected = dequantGemv(quantized.packed, quantized.scales,
                                          quantized.biases, row, layout);
        const auto produced = std::span<const float>{actual}.subspan(
                static_cast<usize>(token) * layout.outFeatures,
                static_cast<usize>(layout.outFeatures));

        const auto deviation = compare(expected, produced);
        INFO("token " << token << ": " << deviation.describe());
        CHECK(deviation.maxRelative < tolerance::kGemv);
    }
}

}  // namespace

TEST_CASE("batched GEMM matches the GEMV reference for every token", "[kernels]") {
    REQUIRE_GPU();

    // A routed expert's gate projection at its real shape.
    const QuantizedLinearLayout layout{
            .outFeatures = 704, .inFeatures = 2816, .spec = kWeightQuant};

    // One tile, a partial tile, and a count spanning several tiles unevenly.
    const u32 tokens = GENERATE(u32{1}, u32{16}, u32{17}, u32{64}, u32{100});
    CAPTURE(tokens);
    checkGemm(backend, kernels, stream, layout, tokens, 2200 + tokens);
}

TEST_CASE("batched GEMM agrees with the GEMV kernel it replaces", "[kernels]") {
    REQUIRE_GPU();
    Device device{backend, stream};

    const QuantizedLinearLayout layout{
            .outFeatures = 256, .inFeatures = 2816, .spec = kWeightQuant};
    constexpr u32 kTokens = 24;

    SplitMix64 rng{2311};
    const auto weights = randomGaussians(
            rng, static_cast<usize>(layout.outFeatures * layout.inFeatures), 0.05f);
    const auto quantized = quantizeMatrix(weights, layout);
    const auto input = throughHalf(randomGaussians(
            rng, static_cast<usize>(kTokens) * static_cast<usize>(layout.inFeatures)));

    auto packedBuffer = device.uploadWords(quantized.packed, "packed");
    auto scaleBuffer = device.uploadBf16Raw(quantized.scales, "scales");
    auto biasBuffer = device.uploadBf16Raw(quantized.biases, "biases");
    auto inputBuffer = device.uploadHalf(input, "input");
    auto gemmOutput =
            device.empty(u64{kTokens} * layout.outFeatures * sizeof(float), "gemm");
    auto gemvOutput = device.empty(layout.outFeatures * sizeof(float), "gemv");

    const QuantizedWeights quantizedViews{.packed = view(packedBuffer),
                                          .scales = view(scaleBuffer),
                                          .biases = view(biasBuffer),
                                          .layout = layout};

    REQUIRE(kernels
                    .dequantGemm(stream, DequantGemmArgs{.weights = quantizedViews,
                                                         .input = view(inputBuffer),
                                                         .output = view(gemmOutput),
                                                         .tokens = kTokens})
                    .has_value());
    const auto batched =
            device.readHalf(*gemmOutput, static_cast<usize>(kTokens) * layout.outFeatures);

    // Both kernels sum the same 2816 products in a different order, so they
    // agree far more tightly than either agrees with the double-precision
    // reference. A regression in the batched path shows up here first.
    for (u32 token = 0; token < kTokens; ++token) {
        const u64 offset = u64{token} * layout.inFeatures * sizeof(float);
        REQUIRE(kernels
                        .dequantGemv(stream,
                                     DequantGemvArgs{
                                             .weights = quantizedViews,
                                             .input = view(inputBuffer).at(offset),
                                             .output = view(gemvOutput)})
                        .has_value());
        const auto single =
                device.readHalf(*gemvOutput, static_cast<usize>(layout.outFeatures));
        const auto produced = std::span<const float>{batched}.subspan(
                static_cast<usize>(token) * layout.outFeatures,
                static_cast<usize>(layout.outFeatures));

        const auto deviation = compare(single, produced);
        INFO("token " << token << ": " << deviation.describe());
        CHECK(deviation.maxRelative < 1e-4f);
    }
}

// Hidden by the leading dot: run with `tf_tests "[bench]"`. Not an assertion
// about performance, just the measurement that justifies the kernel existing.
TEST_CASE("batched GEMM throughput against repeated GEMV", "[.][bench]") {
    REQUIRE_GPU();
    Device device{backend, stream};

    const QuantizedLinearLayout layout{
            .outFeatures = 2816, .inFeatures = 2816, .spec = kWeightQuant};
    constexpr u32 kTokens = 128;
    constexpr u32 kRepeats = 20;

    SplitMix64 rng{4242};
    const auto weights = randomGaussians(
            rng, static_cast<usize>(layout.outFeatures * layout.inFeatures), 0.05f);
    const auto quantized = quantizeMatrix(weights, layout);
    const auto input = randomGaussians(
            rng, static_cast<usize>(kTokens) * static_cast<usize>(layout.inFeatures));

    auto packedBuffer = device.uploadWords(quantized.packed, "packed");
    auto scaleBuffer = device.uploadBf16Raw(quantized.scales, "scales");
    auto biasBuffer = device.uploadBf16Raw(quantized.biases, "biases");
    auto inputBuffer = device.uploadHalf(input, "input");
    auto outputBuffer =
            device.empty(u64{kTokens} * layout.outFeatures * sizeof(float), "output");

    const QuantizedWeights views{.packed = view(packedBuffer),
                                 .scales = view(scaleBuffer),
                                 .biases = view(biasBuffer),
                                 .layout = layout};

    const auto time = [&](auto&& body) {
        body();
        REQUIRE(stream.synchronize().has_value());
        const auto start = std::chrono::steady_clock::now();
        for (u32 i = 0; i < kRepeats; ++i) {
            body();
        }
        REQUIRE(stream.synchronize().has_value());
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() /
               kRepeats;
    };

    const double gemvSeconds = time([&] {
        for (u32 token = 0; token < kTokens; ++token) {
            REQUIRE(kernels
                            .dequantGemv(
                                    stream,
                                    DequantGemvArgs{
                                            .weights = views,
                                            .input = view(inputBuffer)
                                                             .at(u64{token} * layout.inFeatures *
                                                                 sizeof(float)),
                                            .output = view(outputBuffer)
                                                              .at(u64{token} *
                                                                  layout.outFeatures *
                                                                  sizeof(float))})
                            .has_value());
        }
    });

    const double gemmSeconds = time([&] {
        REQUIRE(kernels
                        .dequantGemm(stream, DequantGemmArgs{.weights = views,
                                                             .input = view(inputBuffer),
                                                             .output = view(outputBuffer),
                                                             .tokens = kTokens})
                        .has_value());
    });

    WARN("GEMV x" << kTokens << ": " << gemvSeconds * 1e3 << " ms, GEMM: "
                  << gemmSeconds * 1e3 << " ms, speedup " << gemvSeconds / gemmSeconds << "x");
    CHECK(gemmSeconds > 0.0);
}

TEST_CASE("batched GEMM handles the 8-bit router projection", "[kernels]") {
    REQUIRE_GPU();

    const QuantizedLinearLayout layout{
            .outFeatures = 128, .inFeatures = 2816, .spec = kRouterQuant};
    checkGemm(backend, kernels, stream, layout, 20, 907);
}

TEST_CASE("embedding lookup matches the reference", "[kernels]") {
    REQUIRE_GPU();
    Device device{backend, stream};

    // A reduced vocabulary keeps the fixture small; the hidden size is real,
    // since that is what the sqrt scale depends on.
    const QuantizedLinearLayout layout{
            .outFeatures = 512, .inFeatures = 2816, .spec = kWeightQuant};

    SplitMix64 rng{114};
    const auto table =
            randomGaussians(rng, static_cast<usize>(layout.outFeatures * layout.inFeatures),
                            0.05f);
    const auto quantized = quantizeMatrix(table, layout);

    auto packedBuffer = device.uploadWords(quantized.packed, "packed");
    auto scaleBuffer = device.uploadBf16Raw(quantized.scales, "scales");
    auto biasBuffer = device.uploadBf16Raw(quantized.biases, "biases");
    auto outputBuffer = device.empty(layout.inFeatures * sizeof(float), "embedding");

    for (const u32 tokenId : {u32{0}, u32{1}, u32{255}, u32{511}}) {
        REQUIRE(kernels
                        .embedLookup(stream,
                                     EmbedLookupArgs{
                                             .table = {.packed = view(packedBuffer),
                                                       .scales = view(scaleBuffer),
                                                       .biases = view(biasBuffer),
                                                       .layout = layout},
                                             .output = view(outputBuffer),
                                             .tokenId = tokenId})
                        .has_value());

        const auto expected = embedLookup(quantized.packed, quantized.scales,
                                          quantized.biases, tokenId, layout);
        const auto actual =
                device.readHalf(*outputBuffer, static_cast<usize>(layout.inFeatures));

        const auto deviation = compare(expected, actual);
        INFO("token " << tokenId << ": " << deviation.describe());
        REQUIRE(deviation.maxRelative < tolerance::kGemv);
    }
}

TEST_CASE("an out-of-range token id is refused", "[kernels]") {
    REQUIRE_GPU();
    Device device{backend, stream};

    const QuantizedLinearLayout layout{
            .outFeatures = 16, .inFeatures = 128, .spec = kWeightQuant};

    std::vector<u32> packed(static_cast<usize>(layout.outFeatures * layout.packedWordsPerRow()));
    std::vector<bf16> scales(static_cast<usize>(layout.outFeatures * layout.groupsPerRow()));
    std::vector<bf16> biases(scales.size());

    auto packedBuffer = device.uploadWords(packed, "packed");
    auto scaleBuffer = device.uploadBf16Raw(scales, "scales");
    auto biasBuffer = device.uploadBf16Raw(biases, "biases");
    auto outputBuffer = device.empty(layout.inFeatures * sizeof(float), "embedding");

    const auto status =
            kernels.embedLookup(stream, EmbedLookupArgs{.table = {.packed = view(packedBuffer),
                                                                  .scales = view(scaleBuffer),
                                                                  .biases = view(biasBuffer),
                                                                  .layout = layout},
                                                        .output = view(outputBuffer),
                                                        .tokenId = 999});
    REQUIRE_FALSE(status.has_value());
    CHECK(status.error().code() == ErrorCode::InvalidArgument);
    CHECK_THAT(status.error().message(), Catch::Matchers::ContainsSubstring("vocabulary"));
}

// ---------------------------------------------------------------------------
// Argument validation
// ---------------------------------------------------------------------------

TEST_CASE("kernels reject unset and undersized views", "[kernels]") {
    REQUIRE_GPU();
    Device device{backend, stream};

    auto small = device.empty(64, "small");

    SECTION("unset view") {
        const auto status = kernels.rmsNorm(
                stream, RmsNormArgs{.input = {}, .output = view(small), .count = 32});
        REQUIRE_FALSE(status.has_value());
        CHECK(status.error().code() == ErrorCode::InvalidArgument);
        CHECK_THAT(status.error().message(), Catch::Matchers::ContainsSubstring("not set"));
    }

    SECTION("extent past the end of the buffer") {
        // 1024 fp16 elements need 2048 bytes, but the buffer holds 64.
        const auto status = kernels.rmsNorm(
                stream, RmsNormArgs{
                                .input = view(small), .output = view(small), .count = 1024});
        REQUIRE_FALSE(status.has_value());
        CHECK(status.error().code() == ErrorCode::InvalidArgument);
        CHECK_THAT(status.error().message(),
                   Catch::Matchers::ContainsSubstring("runs past buffer"));
        CHECK_THAT(status.error().message(), Catch::Matchers::ContainsSubstring("small"));
    }

    SECTION("zero extent") {
        CHECK_FALSE(kernels
                            .rmsNorm(stream, RmsNormArgs{.input = view(small),
                                                         .output = view(small),
                                                         .count = 0})
                            .has_value());
    }
}
