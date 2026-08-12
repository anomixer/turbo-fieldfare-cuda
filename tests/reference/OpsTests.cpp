// Tests for the CPU reference implementations.
//
// These are the ground truth every CUDA kernel is measured against, so an error
// here propagates silently into M5 and M8. They are therefore checked against
// independently derived properties - analytic identities, closed forms, and the
// semantics recorded in docs/MODEL_SEMANTICS.md - rather than against golden
// numbers this same code produced.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

#include "tf/reference/Ops.h"
#include "tf/reference/Prng.h"
#include "tf/reference/Tolerance.h"

using namespace tf;
using namespace tf::reference;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

// ---------------------------------------------------------------------------
// PRNG
// ---------------------------------------------------------------------------

TEST_CASE("SplitMix64 matches the reference sequence", "[prng]") {
    // Known values for the canonical SplitMix64 with seed 0.
    SplitMix64 rng{0};
    CHECK(rng.next() == 0xE220A8397B1DCDAFull);
    CHECK(rng.next() == 0x6E789E6AA1B965F4ull);
    CHECK(rng.next() == 0x06C45D188009454Full);
}

TEST_CASE("SplitMix64 is reproducible and seed-sensitive", "[prng]") {
    SplitMix64 a{12345};
    SplitMix64 b{12345};
    SplitMix64 c{12346};

    for (int i = 0; i < 32; ++i) {
        REQUIRE(a.next() == b.next());
    }
    CHECK(SplitMix64{12345}.next() != c.next());
}

TEST_CASE("SeedTree streams are independent", "[prng]") {
    const SeedTree tree{0xABCDEF};
    auto first = tree.stream(0);
    auto second = tree.stream(1);
    auto firstAgain = tree.stream(0);

    CHECK(first.next() != second.next());
    // Adding a tensor to a test must not perturb existing streams.
    CHECK(tree.stream(0).next() == firstAgain.next());
}

TEST_CASE("Gaussian samples have the expected moments", "[prng]") {
    SplitMix64 rng{99};
    constexpr usize kCount = 100000;

    double sum = 0.0;
    double sumSquares = 0.0;
    for (usize i = 0; i < kCount; ++i) {
        const double value = rng.nextGaussian();
        sum += value;
        sumSquares += value * value;
    }
    const double mean = sum / kCount;
    const double variance = sumSquares / kCount - mean * mean;

    CHECK(std::abs(mean) < 0.02);
    CHECK(std::abs(variance - 1.0) < 0.02);
}

// ---------------------------------------------------------------------------
// Quantization
// ---------------------------------------------------------------------------

TEST_CASE("quantize/dequantize round-trips within the 4-bit step", "[quant-ref]") {
    SplitMix64 rng{7};
    const QuantizedLinearLayout layout{
            .outFeatures = 8, .inFeatures = 128, .spec = kWeightQuant};

    const auto original = randomGaussians(rng, static_cast<usize>(8 * 128));
    const auto quantized = quantizeMatrix(original, layout);
    const auto recovered =
            dequantizeMatrix(quantized.packed, quantized.scales, quantized.biases, layout);

    REQUIRE(recovered.size() == original.size());

    // 4 bits over a group's range gives 15 steps, so error is bounded by half a
    // step of that group's span, plus bf16 error on the scale itself.
    for (u64 row = 0; row < layout.outFeatures; ++row) {
        for (u64 group = 0; group < layout.groupsPerRow(); ++group) {
            const u64 begin = row * layout.inFeatures + group * layout.spec.groupSize;
            const auto span = std::span{original}.subspan(
                    static_cast<usize>(begin), static_cast<usize>(layout.spec.groupSize));
            const float low = *std::ranges::min_element(span);
            const float high = *std::ranges::max_element(span);
            const float step = (high - low) / 15.0f;

            for (u64 i = 0; i < layout.spec.groupSize; ++i) {
                const usize index = static_cast<usize>(begin + i);
                INFO("row " << row << " group " << group << " element " << i);
                REQUIRE(std::abs(recovered[index] - original[index]) <= step * 0.55f + 1e-4f);
            }
        }
    }
}

TEST_CASE("8-bit quantization is markedly finer than 4-bit", "[quant-ref]") {
    SplitMix64 rng{11};
    const auto original = randomGaussians(rng, 256);

    const QuantizedLinearLayout fourBit{
            .outFeatures = 4, .inFeatures = 64, .spec = kWeightQuant};
    const QuantizedLinearLayout eightBit{
            .outFeatures = 4, .inFeatures = 64, .spec = kRouterQuant};

    const auto q4 = quantizeMatrix(original, fourBit);
    const auto q8 = quantizeMatrix(original, eightBit);

    const auto d4 = dequantizeMatrix(q4.packed, q4.scales, q4.biases, fourBit);
    const auto d8 = dequantizeMatrix(q8.packed, q8.scales, q8.biases, eightBit);

    const auto error = [&](const std::vector<float>& recovered) {
        double worst = 0.0;
        for (usize i = 0; i < original.size(); ++i) {
            worst = std::max(worst, std::abs(static_cast<double>(recovered[i]) - original[i]));
        }
        return worst;
    };

    // 255 levels versus 15: at least an order of magnitude better.
    CHECK(error(d8) * 10.0 < error(d4));
}

TEST_CASE("packing is low-order first", "[quant-ref]") {
    // One row of 8 values at 4 bits packs into a single word, element 0 in the
    // low nibble. Getting this backwards would transpose every weight.
    const QuantizedLinearLayout layout{
            .outFeatures = 1, .inFeatures = 8, .spec = QuantSpec{.bits = 4, .groupSize = 8}};

    std::vector<u32> packed{0x76543210u};
    std::vector<bf16> scales{toBf16(1.0f)};
    std::vector<bf16> biases{toBf16(0.0f)};

    for (u64 i = 0; i < 8; ++i) {
        INFO("element " << i);
        CHECK(dequantizeElement(packed, scales, biases, 0, i, 8, layout.spec) ==
              static_cast<float>(i));
    }
}

TEST_CASE("dequantization is affine: bias is added, not subtracted", "[quant-ref]") {
    const QuantSpec spec{.bits = 4, .groupSize = 8};
    const std::vector<u32> packed{0x00000005u};  // element 0 = 5
    const std::vector<bf16> scales{toBf16(2.0f)};
    const std::vector<bf16> biases{toBf16(-3.0f)};

    // 5 * 2 + (-3) = 7. A zero-point convention would give (5 - (-3)) * 2 = 16.
    CHECK(dequantizeElement(packed, scales, biases, 0, 0, 8, spec) == 7.0f);
}

TEST_CASE("dequantGemv agrees with an explicit dense matmul", "[quant-ref]") {
    SplitMix64 rng{21};
    const QuantizedLinearLayout layout{
            .outFeatures = 16, .inFeatures = 128, .spec = kWeightQuant};

    const auto weights = randomGaussians(rng, static_cast<usize>(16 * 128));
    const auto quantized = quantizeMatrix(weights, layout);
    const auto x = randomGaussians(rng, 128);

    const auto gemv =
            dequantGemv(quantized.packed, quantized.scales, quantized.biases, x, layout);

    // Independent dense reference over the dequantized matrix.
    const auto dense =
            dequantizeMatrix(quantized.packed, quantized.scales, quantized.biases, layout);
    std::vector<float> expected(16);
    for (u64 row = 0; row < 16; ++row) {
        double sum = 0.0;
        for (u64 column = 0; column < 128; ++column) {
            sum += static_cast<double>(dense[static_cast<usize>(row * 128 + column)]) *
                   x[static_cast<usize>(column)];
        }
        expected[static_cast<usize>(row)] = static_cast<float>(sum);
    }

    const auto deviation = compare(expected, gemv);
    INFO(deviation.describe());
    CHECK(deviation.maxRelative < 1e-6);
}

TEST_CASE("embedding lookup selects the row and applies the sqrt scale",
          "[quant-ref]") {
    const QuantizedLinearLayout layout{
            .outFeatures = 4, .inFeatures = 64, .spec = kWeightQuant};

    SplitMix64 rng{33};
    const auto table = randomGaussians(rng, 4 * 64);
    const auto quantized = quantizeMatrix(table, layout);
    const auto dense =
            dequantizeMatrix(quantized.packed, quantized.scales, quantized.biases, layout);

    const auto row = embedLookup(quantized.packed, quantized.scales, quantized.biases,
                                 /*tokenId=*/2, layout);
    REQUIRE(row.size() == 64);

    const auto scale = static_cast<float>(std::sqrt(64.0));
    for (u64 i = 0; i < 64; ++i) {
        INFO("column " << i);
        REQUIRE_THAT(row[static_cast<usize>(i)],
                     WithinRel(dense[static_cast<usize>(2 * 64 + i)] * scale, 1e-5f));
    }
}

// ---------------------------------------------------------------------------
// Normalization
// ---------------------------------------------------------------------------

TEST_CASE("RMSNorm uses plain weight, not 1 + weight", "[norm-ref]") {
    // The Gemma 4 convention. Under the Gemma 1/2/3 unit-offset form the result
    // here would be doubled, so this test is what stops that regression.
    const std::vector<float> x{3.0f, 4.0f, 0.0f, 0.0f};
    const std::vector<float> weight(4, 1.0f);

    const auto out = rmsNorm(x, weight, 0.0);

    // rms = sqrt((9 + 16) / 4) = 2.5
    CHECK_THAT(out[0], WithinRel(3.0f / 2.5f, 1e-6f));
    CHECK_THAT(out[1], WithinRel(4.0f / 2.5f, 1e-6f));
    CHECK(out[2] == 0.0f);
}

TEST_CASE("RMSNorm output has unit RMS before weighting", "[norm-ref]") {
    SplitMix64 rng{5};
    const auto x = randomGaussians(rng, 512, 7.0f);
    const std::vector<float> ones(512, 1.0f);

    const auto out = rmsNorm(x, ones, 1e-6);

    double sumSquares = 0.0;
    for (const float value : out) {
        sumSquares += static_cast<double>(value) * value;
    }
    CHECK_THAT(std::sqrt(sumSquares / 512.0), WithinAbs(1.0, 1e-4));
}

TEST_CASE("RMSNorm is scale invariant in its input", "[norm-ref]") {
    SplitMix64 rng{6};
    const auto x = randomGaussians(rng, 256);
    std::vector<float> scaled(x.size());
    std::ranges::transform(x, scaled.begin(), [](float v) { return v * 100.0f; });

    const std::vector<float> weight(256, 1.5f);
    const auto a = rmsNorm(x, weight, 0.0);
    const auto b = rmsNorm(scaled, weight, 0.0);

    const auto deviation = compare(a, b);
    INFO(deviation.describe());
    CHECK(deviation.maxRelative < 1e-5);
}

TEST_CASE("rmsNormNoScale equals rmsNorm with unit weights", "[norm-ref]") {
    SplitMix64 rng{8};
    const auto x = randomGaussians(rng, 128);
    const std::vector<float> ones(128, 1.0f);

    const auto deviation = compare(rmsNorm(x, ones, 1e-6), rmsNormNoScale(x, 1e-6));
    INFO(deviation.describe());
    CHECK(deviation.maxRelative < 1e-6);
}

TEST_CASE("eps prevents division by zero on an all-zero input", "[norm-ref]") {
    const std::vector<float> zeros(16, 0.0f);
    const std::vector<float> weight(16, 1.0f);
    const auto out = rmsNorm(zeros, weight, 1e-6);
    for (const float value : out) {
        CHECK(value == 0.0f);
    }
}

// ---------------------------------------------------------------------------
// RoPE
// ---------------------------------------------------------------------------

TEST_CASE("RoPE at position zero is the identity", "[rope-ref]") {
    SplitMix64 rng{13};
    auto head = randomGaussians(rng, 64);
    const auto original = head;

    applyRope(head, 0, RopeParams{.headDim = 64, .theta = 10000.0});

    const auto deviation = compare(original, head);
    INFO(deviation.describe());
    CHECK(deviation.maxAbsolute < 1e-6);
}

TEST_CASE("RoPE preserves the norm of each rotated pair", "[rope-ref]") {
    SplitMix64 rng{14};
    auto head = randomGaussians(rng, 64);
    const auto original = head;

    applyRope(head, 37, RopeParams{.headDim = 64, .theta = 10000.0});

    // A rotation cannot change a pair's magnitude.
    for (usize i = 0; i < 32; ++i) {
        const double before = static_cast<double>(original[i]) * original[i] +
                              static_cast<double>(original[i + 32]) * original[i + 32];
        const double after = static_cast<double>(head[i]) * head[i] +
                             static_cast<double>(head[i + 32]) * head[i + 32];
        INFO("pair " << i);
        REQUIRE_THAT(after, WithinRel(before, 1e-5));
    }
}

TEST_CASE("RoPE rotations compose additively in position", "[rope-ref]") {
    SplitMix64 rng{15};
    const auto original = randomGaussians(rng, 32);
    const RopeParams params{.headDim = 32, .theta = 10000.0};

    // Rotating by 3 then by 5 is not the same call, but the relative angle
    // between positions 3 and 8 must match a direct rotation by 5 applied to
    // the position-3 result. Verified through the dot product, which depends
    // only on the position difference.
    auto atThree = original;
    applyRope(atThree, 3, params);
    auto atEight = original;
    applyRope(atEight, 8, params);

    auto shifted = atThree;
    applyRope(shifted, 5, params);

    const auto deviation = compare(atEight, shifted);
    INFO(deviation.describe());
    CHECK(deviation.maxAbsolute < 1e-4);
}

TEST_CASE("partial rotary pairs across the whole head", "[rope-ref]") {
    SplitMix64 rng{16};
    auto head = randomGaussians(rng, 512);
    const auto original = head;

    // Full-attention layers: 512-dim heads with a quarter rotated.
    const RopeParams params{
            .headDim = 512, .theta = 1000000.0, .partialRotaryFactor = 0.25};
    REQUIRE(params.rotatedDims() == 128);
    REQUIRE(params.rotatedPairs() == 64);

    applyRope(head, 100, params);

    // Pair j joins j and j + 256, and only pairs 0..63 rotate. So exactly two
    // disjoint bands move: 0..63 and 256..319. Everything else is untouched.
    //
    // Deriving the pairing from rotatedDims instead would move 0..127, which is
    // still a plausible-looking rotation - and was the bug that made the model
    // emit fluent nonsense.
    for (usize i = 0; i < 512; ++i) {
        const bool shouldMove = (i < 64) || (i >= 256 && i < 320);
        INFO("dimension " << i << (shouldMove ? " should rotate" : " should be untouched"));
        if (shouldMove) {
            REQUIRE(head[i] != original[i]);
        } else {
            REQUIRE(head[i] == original[i]);
        }
    }
}

TEST_CASE("the partial rotary exponent divides by headDim", "[rope-ref]") {
    // Pair 0 has exponent 0 in both conventions, so compare pair 1, where the
    // two differ: 2/512 against 2/128.
    std::vector<float> head(512, 0.0f);
    head[1] = 1.0f;
    head[1 + 256] = 0.0f;

    const RopeParams params{
            .headDim = 512, .theta = 1000000.0, .partialRotaryFactor = 0.25};
    applyRope(head, 1, params);

    const double correctFrequency = 1.0 / std::pow(1000000.0, 2.0 / 512.0);
    CHECK_THAT(head[1], WithinAbs(static_cast<float>(std::cos(correctFrequency)), 1e-5f));
    CHECK_THAT(head[1 + 256],
               WithinAbs(static_cast<float>(std::sin(correctFrequency)), 1e-5f));

    // The rotatedDims-based exponent would give a visibly different angle.
    const double wrongFrequency = 1.0 / std::pow(1000000.0, 2.0 / 128.0);
    CHECK(std::abs(std::cos(correctFrequency) - std::cos(wrongFrequency)) > 1e-3);
}

TEST_CASE("full rotation is the degenerate case of partial", "[rope-ref]") {
    // With partialRotaryFactor 1.0 the pairing and exponent are the standard
    // ones, so the sliding layers are unaffected by the partial-rotary rules.
    SplitMix64 rng{161};
    const auto original = randomGaussians(rng, 256);

    auto viaPartial = original;
    applyRope(viaPartial, 42,
              RopeParams{.headDim = 256, .theta = 10000.0, .partialRotaryFactor = 1.0});

    // Hand-rolled standard RoPE over 256 dims.
    auto expected = original;
    for (usize j = 0; j < 128; ++j) {
        const double frequency =
                1.0 / std::pow(10000.0, static_cast<double>(2 * j) / 256.0);
        const double angle = 42.0 * frequency;
        const double lower = original[j];
        const double upper = original[j + 128];
        expected[j] = static_cast<float>(lower * std::cos(angle) - upper * std::sin(angle));
        expected[j + 128] =
                static_cast<float>(lower * std::sin(angle) + upper * std::cos(angle));
    }

    const auto deviation = compare(expected, viaPartial);
    INFO(deviation.describe());
    CHECK(deviation.maxAbsolute < 1e-5);
}

TEST_CASE("rotatedDims stays even", "[rope-ref]") {
    // An odd count would leave one dimension without a partner.
    CHECK(RopeParams{.headDim = 100, .partialRotaryFactor = 0.25}.rotatedDims() == 24);
    CHECK(RopeParams{.headDim = 256, .partialRotaryFactor = 1.0}.rotatedDims() == 256);
    CHECK(RopeParams{.headDim = 512, .partialRotaryFactor = 0.25}.rotatedDims() == 128);
}

// ---------------------------------------------------------------------------
// Activations
// ---------------------------------------------------------------------------

TEST_CASE("GELU matches known values and limits", "[act-ref]") {
    CHECK(geluApprox(0.0f) == 0.0f);
    CHECK_THAT(geluApprox(1.0f), WithinAbs(0.8411920f, 1e-5f));
    CHECK_THAT(geluApprox(-1.0f), WithinAbs(-0.1588080f, 1e-5f));
    CHECK_THAT(geluApprox(2.0f), WithinAbs(1.9545977f, 1e-5f));

    // Asymptotically the identity for large positive, zero for large negative.
    CHECK_THAT(geluApprox(10.0f), WithinRel(10.0f, 1e-4f));
    CHECK_THAT(geluApprox(-10.0f), WithinAbs(0.0f, 1e-4f));
}

TEST_CASE("GeGLU applies GELU to the gate, not the up projection", "[act-ref]") {
    // Inverting these is a classic port bug and would still produce plausible
    // output, so it is pinned explicitly.
    const std::vector<float> gate{1.0f, -1.0f, 2.0f};
    const std::vector<float> up{3.0f, 3.0f, 3.0f};

    const auto out = geglu(gate, up);
    CHECK_THAT(out[0], WithinAbs(geluApprox(1.0f) * 3.0f, 1e-6f));
    CHECK_THAT(out[1], WithinAbs(geluApprox(-1.0f) * 3.0f, 1e-6f));

    // The reversed form would give geluApprox(3.0f) * 1.0f for element 0.
    CHECK(std::abs(out[0] - geluApprox(3.0f) * 1.0f) > 0.1f);
}

// ---------------------------------------------------------------------------
// Softmax and attention
// ---------------------------------------------------------------------------

TEST_CASE("softmax sums to one and is shift invariant", "[attn-ref]") {
    SplitMix64 rng{17};
    const auto logits = randomGaussians(rng, 100, 5.0f);

    const auto probabilities = softmax(logits);
    const double total = std::accumulate(probabilities.begin(), probabilities.end(), 0.0);
    CHECK_THAT(total, WithinAbs(1.0, 1e-5));

    std::vector<float> shifted(logits.size());
    std::ranges::transform(logits, shifted.begin(), [](float v) { return v + 1000.0f; });
    const auto deviation = compare(probabilities, softmax(shifted));
    INFO(deviation.describe());
    // Not exact: adding 1000.0f to values of magnitude ~5 discards low mantissa
    // bits, so the shifted logits are not a perfect translation of the
    // originals. The residual is float32 addition error, not softmax error.
    CHECK(deviation.maxAbsolute < 1e-5);
}

TEST_CASE("softmax survives extreme logits without overflow", "[attn-ref]") {
    const std::vector<float> extreme{1000.0f, -1000.0f, 0.0f};
    const auto out = softmax(extreme);
    CHECK_THAT(out[0], WithinAbs(1.0f, 1e-6f));
    CHECK_THAT(out[1], WithinAbs(0.0f, 1e-6f));
    for (const float value : out) {
        CHECK(std::isfinite(value));
    }
}

TEST_CASE("attention over one position returns that value", "[attn-ref]") {
    const AttentionParams params{
            .numHeads = 2, .numKVHeads = 1, .headDim = 4, .scale = 1.0f};

    const std::vector<float> queries(2 * 4, 1.0f);
    const std::vector<float> keys{1.0f, 0.0f, 0.0f, 0.0f};
    const std::vector<float> values{5.0f, 6.0f, 7.0f, 8.0f};

    const auto out = decodeAttention(queries, keys, values, 1, 0, params);
    REQUIRE(out.size() == 8);

    // With a single position, softmax gives weight 1 regardless of the score.
    for (u64 head = 0; head < 2; ++head) {
        for (u64 d = 0; d < 4; ++d) {
            INFO("head " << head << " dim " << d);
            REQUIRE_THAT(out[static_cast<usize>(head * 4 + d)],
                         WithinRel(values[static_cast<usize>(d)], 1e-5f));
        }
    }
}

TEST_CASE("attention output is a convex combination of the values",
          "[attn-ref]") {
    SplitMix64 rng{19};
    const AttentionParams params{
            .numHeads = 4, .numKVHeads = 2, .headDim = 8, .scale = 1.0f};
    constexpr u64 kCached = 16;

    const auto queries = randomGaussians(rng, 4 * 8);
    const auto keys = randomGaussians(rng, 2 * kCached * 8);
    const auto values = randomGaussians(rng, 2 * kCached * 8);

    const auto out = decodeAttention(queries, keys, values, kCached, kCached - 1, params);

    // Every output must lie within the range of the values it averaged.
    for (u64 head = 0; head < 4; ++head) {
        const u64 kvHead = head / 2;
        for (u64 d = 0; d < 8; ++d) {
            float low = values[static_cast<usize>((kvHead * kCached) * 8 + d)];
            float high = low;
            for (u64 position = 0; position < kCached; ++position) {
                const float value =
                        values[static_cast<usize>((kvHead * kCached + position) * 8 + d)];
                low = std::min(low, value);
                high = std::max(high, value);
            }
            const float result = out[static_cast<usize>(head * 8 + d)];
            INFO("head " << head << " dim " << d << " value " << result << " in [" << low
                         << ", " << high << "]");
            REQUIRE(result >= low - 1e-4f);
            REQUIRE(result <= high + 1e-4f);
        }
    }
}

TEST_CASE("the sliding window masks older positions", "[attn-ref]") {
    const AttentionParams params{
            .numHeads = 1, .numKVHeads = 1, .headDim = 2, .scale = 1.0f, .slidingWindow = 4};
    constexpr u64 kCached = 10;

    // All keys identical, so weights depend only on which positions are visible.
    std::vector<float> keys(kCached * 2, 1.0f);
    std::vector<float> values(kCached * 2, 0.0f);
    // Mark the oldest position with a distinctive value.
    values[0] = 1000.0f;
    values[1] = 1000.0f;

    const auto out = decodeAttention(std::vector<float>{1.0f, 1.0f}, keys, values, kCached,
                                     /*queryPosition=*/9, params);

    // A window of 4 ending at position 9 covers positions 6-9, so position 0
    // must contribute nothing at all.
    CHECK_THAT(out[0], WithinAbs(0.0f, 1e-6f));
    CHECK_THAT(out[1], WithinAbs(0.0f, 1e-6f));
}

TEST_CASE("a window wider than the history sees everything", "[attn-ref]") {
    const AttentionParams params{
            .numHeads = 1, .numKVHeads = 1, .headDim = 2, .scale = 1.0f, .slidingWindow = 1024};

    const std::vector<float> keys(3 * 2, 1.0f);
    const std::vector<float> values{1.0f, 1.0f, 2.0f, 2.0f, 3.0f, 3.0f};

    const auto out =
            decodeAttention(std::vector<float>{0.0f, 0.0f}, keys, values, 3, 2, params);

    // Equal scores, so this is a plain mean of 1, 2 and 3.
    CHECK_THAT(out[0], WithinRel(2.0f, 1e-5f));
}

TEST_CASE("grouped-query heads share their KV head", "[attn-ref]") {
    // 4 Q heads over 2 KV heads: heads 0-1 use KV 0, heads 2-3 use KV 1.
    const AttentionParams params{
            .numHeads = 4, .numKVHeads = 2, .headDim = 2, .scale = 1.0f};

    const std::vector<float> queries(4 * 2, 0.0f);  // equal scores everywhere
    const std::vector<float> keys(2 * 1 * 2, 1.0f);
    const std::vector<float> values{10.0f, 10.0f, 20.0f, 20.0f};

    const auto out = decodeAttention(queries, keys, values, 1, 0, params);

    CHECK_THAT(out[0], WithinRel(10.0f, 1e-5f));  // head 0 -> kv 0
    CHECK_THAT(out[2], WithinRel(10.0f, 1e-5f));  // head 1 -> kv 0
    CHECK_THAT(out[4], WithinRel(20.0f, 1e-5f));  // head 2 -> kv 1
    CHECK_THAT(out[6], WithinRel(20.0f, 1e-5f));  // head 3 -> kv 1
}

// ---------------------------------------------------------------------------
// Router
// ---------------------------------------------------------------------------

TEST_CASE("router selects the highest scores", "[moe-ref]") {
    const std::vector<float> scores{0.1f, 5.0f, 0.2f, 3.0f, 0.3f, 4.0f};
    const std::vector<float> perExpertScale(6, 1.0f);

    const auto result = routerTopK(scores, perExpertScale, 3);
    REQUIRE(result.indices.size() == 3);
    CHECK(result.indices[0] == 1);  // 5.0
    CHECK(result.indices[1] == 5);  // 4.0
    CHECK(result.indices[2] == 3);  // 3.0
}

TEST_CASE("router softmaxes over the selected experts only", "[moe-ref]") {
    // Softmaxing over all experts before selection would give different
    // weights that no longer sum to one.
    const std::vector<float> scores{1.0f, 2.0f, 3.0f, 100.0f, 100.0f};
    const std::vector<float> perExpertScale(5, 1.0f);

    const auto result = routerTopK(scores, perExpertScale, 2);
    REQUIRE(result.weights.size() == 2);

    const double total = result.weights[0] + result.weights[1];
    CHECK_THAT(total, WithinAbs(1.0, 1e-5));
    // Two equal top scores split evenly.
    CHECK_THAT(result.weights[0], WithinRel(0.5f, 1e-5f));
}

TEST_CASE("per-expert scale multiplies after the softmax", "[moe-ref]") {
    const std::vector<float> scores{1.0f, 1.0f};
    const std::vector<float> perExpertScale{2.0f, 4.0f};

    const auto result = routerTopK(scores, perExpertScale, 2);

    // Equal scores give 0.5 each, then the per-expert scale applies. Weights
    // deliberately no longer sum to one.
    CHECK_THAT(result.weights[0], WithinRel(1.0f, 1e-5f));
    CHECK_THAT(result.weights[1], WithinRel(2.0f, 1e-5f));
}

TEST_CASE("router ties break toward the lower index", "[moe-ref]") {
    const std::vector<float> scores(8, 1.0f);
    const std::vector<float> perExpertScale(8, 1.0f);

    const auto result = routerTopK(scores, perExpertScale, 3);
    CHECK(result.indices == std::vector<u32>{0, 1, 2});
}

TEST_CASE("router scale folding matches an explicit division", "[moe-ref]") {
    const std::vector<float> routerScale{32.0f, 30.5f, 33.75f};
    const auto folded = foldRouterScale(routerScale, 2816);

    const auto expected = static_cast<float>(1.0 / std::sqrt(2816.0));
    for (usize i = 0; i < routerScale.size(); ++i) {
        INFO("element " << i);
        REQUIRE_THAT(folded[i], WithinRel(routerScale[i] * expected, 1e-6f));
    }
}

// ---------------------------------------------------------------------------
// Output head
// ---------------------------------------------------------------------------

TEST_CASE("logit softcap saturates and is odd", "[head-ref]") {
    const std::vector<float> logits{0.0f, 30.0f, -30.0f, 1000.0f, -1000.0f};
    const auto capped = logitSoftcap(logits, 30.0f);

    CHECK(capped[0] == 0.0f);
    CHECK_THAT(capped[1], WithinRel(std::tanh(1.0f) * 30.0f, 1e-5f));
    CHECK_THAT(capped[2], WithinRel(-std::tanh(1.0f) * 30.0f, 1e-5f));

    // Saturates at the cap rather than growing without bound.
    CHECK_THAT(capped[3], WithinAbs(30.0f, 1e-3f));
    CHECK_THAT(capped[4], WithinAbs(-30.0f, 1e-3f));
}

TEST_CASE("softcap is monotonic, so it cannot reorder logits", "[head-ref]") {
    SplitMix64 rng{23};
    auto logits = randomGaussians(rng, 512, 40.0f);
    std::ranges::sort(logits);

    const auto capped = logitSoftcap(logits, 30.0f);
    for (usize i = 1; i < capped.size(); ++i) {
        INFO("index " << i);
        REQUIRE(capped[i] >= capped[i - 1] - 1e-6f);
    }
}

TEST_CASE("argmax resolves ties to the lowest index", "[head-ref]") {
    CHECK(argmax(std::vector<float>{1.0f, 5.0f, 3.0f}) == 1);
    // Determinism in greedy decoding depends on this.
    CHECK(argmax(std::vector<float>{5.0f, 5.0f, 5.0f}) == 0);
    CHECK(argmax(std::vector<float>{-3.0f, -1.0f, -2.0f}) == 1);
}

// ---------------------------------------------------------------------------
// Tolerance helper
// ---------------------------------------------------------------------------

TEST_CASE("compare reports the worst relative deviation and where", "[tolerance]") {
    const std::vector<float> reference{1.0f, 2.0f, 100.0f};
    const std::vector<float> candidate{1.0f, 2.2f, 100.1f};

    const auto deviation = compare(reference, candidate);
    CHECK(deviation.worstIndex == 1);
    CHECK_THAT(deviation.maxRelative, WithinRel(0.1, 1e-3));
    CHECK_THAT(deviation.maxAbsolute, WithinRel(0.2, 1e-3));
    CHECK_FALSE(deviation.hasNonFinite);
}

TEST_CASE("compare floors the denominator near zero", "[tolerance]") {
    // Without a floor, a tiny absolute difference against a near-zero reference
    // would report enormous relative error and fail every kernel test.
    const std::vector<float> reference{1e-9f};
    const std::vector<float> candidate{2e-9f};

    const auto deviation = compare(reference, candidate);
    CHECK(deviation.maxRelative < 1e-5);
}

TEST_CASE("compare flags a NaN the other side does not have", "[tolerance]") {
    const std::vector<float> reference{1.0f, 2.0f};
    const std::vector<float> candidate{1.0f, std::numeric_limits<float>::quiet_NaN()};

    const auto deviation = compare(reference, candidate);
    CHECK(deviation.hasNonFinite);
    CHECK(deviation.worstIndex == 1);
}
