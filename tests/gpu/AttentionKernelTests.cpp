// Attention, KV cache writes, router selection and the MoE combine.
//
// These are the kernels with the most room for a silent indexing mistake: a
// grouped-query head mapped to the wrong KV head, a ring slot off by one, or a
// window boundary that includes one position too many all produce plausible
// output. Each is checked against the scalar reference, and several tests are
// built so that the specific wrong answer would be visibly different.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <algorithm>
#include <numeric>
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


/// Rounds through fp16, which is how the KV cache stores keys and values. The
/// reference has to see the same rounding or the comparison charges attention
/// for the cache's storage format.
[[nodiscard]] std::vector<float> throughKv(std::span<const float> values) {
    std::vector<float> out(values.size());
    for (usize i = 0; i < values.size(); ++i) {
        out[i] = toFloat(toFp16(values[i]));
    }
    return out;
}

[[nodiscard]] std::vector<float> throughHalf(std::span<const float> values) {
    return {values.begin(), values.end()};
}

class Device {
public:
    Device(IGpuBackend& backend, Stream& stream) : backend_(backend), stream_(stream) {}

    [[nodiscard]] BufferPtr uploadHalf(std::span<const float> values, std::string name) {
        // Named "half" for continuity, but activations are fp32: Gemma 4's do
        // not fit in fp16.
        return upload(ByteSpan{reinterpret_cast<const u8*>(values.data()),
                               values.size() * sizeof(float)},
                      std::move(name));
    }

    [[nodiscard]] BufferPtr uploadKv(std::span<const float> values, std::string name) {
        // The KV cache stays fp16: keys and values are post-normalization and
        // bounded near unit RMS, unlike the activations around them.
        const auto encoded = toFp16Buffer(values);
        return upload(ByteSpan{reinterpret_cast<const u8*>(encoded.data()),
                               encoded.size() * sizeof(fp16)},
                      std::move(name));
    }

    [[nodiscard]] std::vector<float> readKv(Buffer& buffer, usize count) {
        std::vector<fp16> encoded(count);
        REQUIRE(backend_
                        .enqueueDownload(stream_,
                                         MutableByteSpan{reinterpret_cast<u8*>(encoded.data()),
                                                         count * sizeof(fp16)},
                                         buffer, 0)
                        .has_value());
        REQUIRE(stream_.synchronize().has_value());
        return fromFp16Buffer(encoded);
    }

    [[nodiscard]] BufferPtr uploadBf16(std::span<const float> values, std::string name) {
        const auto encoded = toBf16Buffer(values);
        return upload(ByteSpan{reinterpret_cast<const u8*>(encoded.data()),
                               encoded.size() * sizeof(bf16)},
                      std::move(name));
    }

    [[nodiscard]] BufferPtr empty(u64 bytes, std::string name) {
        auto buffer = backend_.allocate(MemoryKind::Device, bytes, std::move(name));
        REQUIRE(buffer.has_value());
        // Zeroed so an unwritten region reads as zero rather than as whatever
        // the allocator handed back.
        REQUIRE(backend_.enqueueFill(stream_, **buffer, 0, bytes, 0).has_value());
        REQUIRE(stream_.synchronize().has_value());
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

    [[nodiscard]] std::vector<u32> readWords(Buffer& buffer, usize count) {
        std::vector<u32> values(count);
        REQUIRE(backend_
                        .enqueueDownload(stream_,
                                         MutableByteSpan{reinterpret_cast<u8*>(values.data()),
                                                         count * sizeof(u32)},
                                         buffer, 0)
                        .has_value());
        REQUIRE(stream_.synchronize().has_value());
        return values;
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
// KV cache writes
// ---------------------------------------------------------------------------

TEST_CASE("kvWrite places a token at its linear slot", "[kernels][kv]") {
    REQUIRE_GPU();
    Device device{backend, stream};

    constexpr u32 kKvHeads = 2;
    constexpr u32 kHeadDim = 8;
    constexpr u32 kCapacity = 16;

    auto keyCache = device.empty(kKvHeads * kCapacity * kHeadDim * sizeof(fp16), "k");
    auto valueCache = device.empty(kKvHeads * kCapacity * kHeadDim * sizeof(fp16), "v");

    std::vector<float> key(kKvHeads * kHeadDim);
    std::vector<float> value(kKvHeads * kHeadDim);
    std::iota(key.begin(), key.end(), 1.0f);
    std::iota(value.begin(), value.end(), 100.0f);

    auto keyBuffer = device.uploadHalf(key, "key");
    auto valueBuffer = device.uploadHalf(value, "value");

    constexpr u64 kPosition = 5;
    REQUIRE(kernels
                    .kvWrite(stream, KvWriteArgs{.key = view(keyBuffer),
                                                 .value = view(valueBuffer),
                                                 .keyCache = view(keyCache),
                                                 .valueCache = view(valueCache),
                                                 .kvHeads = kKvHeads,
                                                 .headDim = kHeadDim,
                                                 .capacity = kCapacity,
                                                 .position = kPosition,
                                                 .circular = false})
                    .has_value());

    const auto keys = device.readKv(*keyCache, kKvHeads * kCapacity * kHeadDim);
    const auto values = device.readKv(*valueCache, kKvHeads * kCapacity * kHeadDim);

    for (u32 head = 0; head < kKvHeads; ++head) {
        for (u32 d = 0; d < kHeadDim; ++d) {
            const usize slot = (static_cast<usize>(head) * kCapacity + kPosition) * kHeadDim + d;
            const usize source = static_cast<usize>(head) * kHeadDim + d;
            INFO("head " << head << " dim " << d);
            REQUIRE(keys[slot] == key[source]);
            REQUIRE(values[slot] == value[source]);
        }
        // Neighbouring rows must be untouched, which is what catches a stride
        // computed with the wrong head count.
        for (u32 d = 0; d < kHeadDim; ++d) {
            const usize before = (static_cast<usize>(head) * kCapacity + 4) * kHeadDim + d;
            const usize after = (static_cast<usize>(head) * kCapacity + 6) * kHeadDim + d;
            REQUIRE(keys[before] == 0.0f);
            REQUIRE(keys[after] == 0.0f);
        }
    }
}

TEST_CASE("kvWrite wraps a circular cache", "[kernels][kv]") {
    REQUIRE_GPU();
    Device device{backend, stream};

    // The sliding-window layers keep a ring, so position 19 with capacity 16
    // must land on row 3 and overwrite whatever was there.
    constexpr u32 kKvHeads = 1;
    constexpr u32 kHeadDim = 4;
    constexpr u32 kCapacity = 16;

    auto keyCache = device.empty(kCapacity * kHeadDim * sizeof(fp16), "k");
    auto valueCache = device.empty(kCapacity * kHeadDim * sizeof(fp16), "v");

    const std::vector<float> first{1.0f, 2.0f, 3.0f, 4.0f};
    const std::vector<float> second{9.0f, 9.0f, 9.0f, 9.0f};
    auto firstBuffer = device.uploadHalf(first, "first");
    auto secondBuffer = device.uploadHalf(second, "second");

    const auto write = [&](const BufferPtr& source, u64 position) {
        REQUIRE(kernels
                        .kvWrite(stream, KvWriteArgs{.key = view(source),
                                                     .value = view(source),
                                                     .keyCache = view(keyCache),
                                                     .valueCache = view(valueCache),
                                                     .kvHeads = kKvHeads,
                                                     .headDim = kHeadDim,
                                                     .capacity = kCapacity,
                                                     .position = position,
                                                     .circular = true})
                        .has_value());
    };

    write(firstBuffer, 3);
    auto keys = device.readKv(*keyCache, kCapacity * kHeadDim);
    CHECK(keys[3 * kHeadDim + 0] == 1.0f);

    // 19 % 16 == 3.
    write(secondBuffer, 19);
    keys = device.readKv(*keyCache, kCapacity * kHeadDim);
    for (u32 d = 0; d < kHeadDim; ++d) {
        INFO("dim " << d);
        REQUIRE(keys[3 * kHeadDim + d] == 9.0f);
    }
}

TEST_CASE("kvWrite refuses to run off a linear cache", "[kernels][kv]") {
    REQUIRE_GPU();
    Device device{backend, stream};

    auto cache = device.empty(4 * 8 * sizeof(fp16), "cache");
    const std::vector<float> token(8, 1.0f);
    auto tokenBuffer = device.uploadHalf(token, "token");

    const auto status = kernels.kvWrite(stream, KvWriteArgs{.key = view(tokenBuffer),
                                                            .value = view(tokenBuffer),
                                                            .keyCache = view(cache),
                                                            .valueCache = view(cache),
                                                            .kvHeads = 1,
                                                            .headDim = 8,
                                                            .capacity = 4,
                                                            .position = 4,
                                                            .circular = false});
    REQUIRE_FALSE(status.has_value());
    CHECK(status.error().code() == ErrorCode::InvalidArgument);
    CHECK_THAT(status.error().message(), Catch::Matchers::ContainsSubstring("capacity"));
}

// ---------------------------------------------------------------------------
// Decode attention
// ---------------------------------------------------------------------------

namespace {

/// Runs one attention configuration on both the GPU and the reference and
/// returns the deviation between them.
struct AttentionCase {
    u32 numHeads = 0;
    u32 kvHeads = 0;
    u32 headDim = 0;
    u32 capacity = 0;
    /// The query sits here and attends to everything up to and including it,
    /// so the history it sees is basePosition + 1 long.
    u64 basePosition = 0;
    u32 slidingWindow = 0;
    bool circular = false;
    float scale = 1.0f;
    u64 seed = 0;
};

Deviation runAttention(IGpuBackend& backend, IKernels& kernels, Stream& stream,
                       const AttentionCase& config) {
    Device device{backend, stream};
    SplitMix64 rng{config.seed};

    const usize queryCount = static_cast<usize>(config.numHeads) * config.headDim;
    const usize cacheCount =
            static_cast<usize>(config.kvHeads) * config.capacity * config.headDim;

    const auto queries = throughHalf(randomGaussians(rng, queryCount));
    const auto keys = throughKv(randomGaussians(rng, cacheCount));
    const auto values = throughKv(randomGaussians(rng, cacheCount));

    auto queryBuffer = device.uploadHalf(queries, "queries");
    auto keyBuffer = device.uploadKv(keys, "keys");
    auto valueBuffer = device.uploadKv(values, "values");
    auto outputBuffer = device.empty(queryCount * sizeof(float), "output");

    REQUIRE(kernels
                    .attention(
                            stream,
                            AttentionArgs{.queries = view(queryBuffer),
                                                .keyCache = view(keyBuffer),
                                                .valueCache = view(valueBuffer),
                                                .output = view(outputBuffer),
                                                .numHeads = config.numHeads,
                                                .kvHeads = config.kvHeads,
                                                .headDim = config.headDim,
                                                .capacity = config.capacity,
                                                .basePosition = config.basePosition,
                                                .slidingWindow = config.slidingWindow,
                                                .circular = config.circular,
                                                .scale = config.scale})
                    .has_value());

    const auto actual = device.readHalf(*outputBuffer, queryCount);

    const AttentionParams params{.numHeads = config.numHeads,
                                 .numKVHeads = config.kvHeads,
                                 .headDim = config.headDim,
                                 .scale = config.scale,
                                 .slidingWindow = config.slidingWindow,
                                 .capacity = config.capacity,
                                 .circular = config.circular};
    const auto expected = decodeAttention(queries, keys, values, config.basePosition + 1,
                                          config.basePosition, params);

    return compare(expected, actual);
}

}  // namespace

TEST_CASE("attention matches the reference on a full-attention layer",
          "[kernels][attention]") {
    REQUIRE_GPU();

    // Layer 5 geometry: 16 query heads of 512 over 2 KV heads.
    const auto deviation = runAttention(backend, kernels, stream,
                                        AttentionCase{.numHeads = 16,
                                                      .kvHeads = 2,
                                                      .headDim = 512,
                                                      .capacity = 256,
                                                      .basePosition = 199,
                                                      .slidingWindow = 0,
                                                      .circular = false,
                                                      .seed = 201});
    INFO(deviation.describe());
    CHECK(deviation.maxRelative < tolerance::kAttention);
}

TEST_CASE("attention matches the reference on a sliding-window layer",
          "[kernels][attention]") {
    REQUIRE_GPU();

    // Sliding geometry: 16 heads of 256 over 8 KV heads, window 1024, and a
    // 1152-row ring exactly as the real KV cache is shaped.
    const auto deviation = runAttention(backend, kernels, stream,
                                        AttentionCase{.numHeads = 16,
                                                      .kvHeads = 8,
                                                      .headDim = 256,
                                                      .capacity = 1152,
                                                      .basePosition = 999,
                                                      .slidingWindow = 1024,
                                                      .circular = true,
                                                      .seed = 202});
    INFO(deviation.describe());
    CHECK(deviation.maxRelative < tolerance::kAttention);
}

TEST_CASE("attention matches the reference once the ring has wrapped",
          "[kernels][attention]") {
    REQUIRE_GPU();

    // History longer than the ring, so slot indices wrap and the oldest rows
    // are gone. An off-by-one in the wrap would show here and nowhere else.
    const auto deviation = runAttention(backend, kernels, stream,
                                        AttentionCase{.numHeads = 8,
                                                      .kvHeads = 2,
                                                      .headDim = 64,
                                                      .capacity = 128,
                                                      .basePosition = 499,
                                                      .slidingWindow = 100,
                                                      .circular = true,
                                                      .seed = 203});
    INFO(deviation.describe());
    CHECK(deviation.maxRelative < tolerance::kAttention);
}

TEST_CASE("attention handles a short history and a single position",
          "[kernels][attention]") {
    REQUIRE_GPU();

    for (const u64 length : {u64{1}, u64{2}, u64{7}, u64{31}, u64{32}, u64{33}}) {
        const auto deviation = runAttention(backend, kernels, stream,
                                            AttentionCase{.numHeads = 4,
                                                          .kvHeads = 1,
                                                          .headDim = 32,
                                                          .capacity = 64,
                                                          .basePosition = length - 1,
                                                          .seed = 204 + length});
        INFO("history " << length << ": " << deviation.describe());
        REQUIRE(deviation.maxRelative < tolerance::kAttention);
    }
}

TEST_CASE("grouped-query heads read the right KV head", "[kernels][attention]") {
    REQUIRE_GPU();
    Device device{backend, stream};

    // 4 query heads over 2 KV heads. Each KV head holds a distinct constant, so
    // a head mapped to the wrong one produces an obviously wrong value rather
    // than a slightly wrong one.
    constexpr u32 kNumHeads = 4;
    constexpr u32 kKvHeads = 2;
    constexpr u32 kHeadDim = 8;
    constexpr u32 kCapacity = 4;
    constexpr u64 kCached = 4;

    const std::vector<float> queries(kNumHeads * kHeadDim, 0.0f);  // equal scores

    std::vector<float> keys(kKvHeads * kCapacity * kHeadDim, 1.0f);
    std::vector<float> values(kKvHeads * kCapacity * kHeadDim);
    for (u32 head = 0; head < kKvHeads; ++head) {
        const float marker = (head == 0) ? 10.0f : 20.0f;
        for (u32 i = 0; i < kCapacity * kHeadDim; ++i) {
            values[static_cast<usize>(head) * kCapacity * kHeadDim + i] = marker;
        }
    }

    auto queryBuffer = device.uploadHalf(queries, "queries");
    auto keyBuffer = device.uploadKv(keys, "keys");
    auto valueBuffer = device.uploadKv(values, "values");
    auto outputBuffer = device.empty(kNumHeads * kHeadDim * sizeof(float), "output");

    REQUIRE(kernels
                    .attention(stream, AttentionArgs{
                                                     .queries = view(queryBuffer),
                                                     .keyCache = view(keyBuffer),
                                                     .valueCache = view(valueBuffer),
                                                     .output = view(outputBuffer),
                                                     .numHeads = kNumHeads,
                                                     .kvHeads = kKvHeads,
                                                     .headDim = kHeadDim,
                                                     .capacity = kCapacity,
                                                     .basePosition = kCached - 1})
                    .has_value());

    const auto actual = device.readHalf(*outputBuffer, kNumHeads * kHeadDim);

    // Heads 0 and 1 map to KV head 0; heads 2 and 3 to KV head 1.
    for (u32 head = 0; head < kNumHeads; ++head) {
        const float expected = (head < 2) ? 10.0f : 20.0f;
        for (u32 d = 0; d < kHeadDim; ++d) {
            INFO("head " << head << " dim " << d);
            REQUIRE(std::abs(actual[static_cast<usize>(head) * kHeadDim + d] - expected) <
                    0.05f);
        }
    }
}

TEST_CASE("the sliding window excludes older positions exactly",
          "[kernels][attention]") {
    REQUIRE_GPU();
    Device device{backend, stream};

    // A window of 4 ending at position 9 covers 6, 7, 8 and 9. Position 5 is
    // marked with an enormous value: including it by one would be unmissable.
    constexpr u32 kHeadDim = 4;
    constexpr u32 kCapacity = 16;
    constexpr u32 kWindow = 4;

    const std::vector<float> queries(kHeadDim, 0.0f);
    const std::vector<float> keys(kCapacity * kHeadDim, 1.0f);

    std::vector<float> values(kCapacity * kHeadDim, 1.0f);
    for (u32 d = 0; d < kHeadDim; ++d) {
        values[5 * kHeadDim + d] = 1000.0f;
    }

    auto queryBuffer = device.uploadHalf(queries, "queries");
    auto keyBuffer = device.uploadKv(keys, "keys");
    auto valueBuffer = device.uploadKv(values, "values");
    auto outputBuffer = device.empty(kHeadDim * sizeof(float), "output");

    REQUIRE(kernels
                    .attention(stream, AttentionArgs{
                                                     .queries = view(queryBuffer),
                                                     .keyCache = view(keyBuffer),
                                                     .valueCache = view(valueBuffer),
                                                     .output = view(outputBuffer),
                                                     .numHeads = 1,
                                                     .kvHeads = 1,
                                                     .headDim = kHeadDim,
                                                     .capacity = kCapacity,
                                                     .basePosition = 9,
                                                     .slidingWindow = kWindow})
                    .has_value());

    const auto actual = device.readHalf(*outputBuffer, kHeadDim);
    for (u32 d = 0; d < kHeadDim; ++d) {
        INFO("dim " << d << " value " << actual[d]);
        // All visible values are 1.0, so the mean is 1.0. Position 5 leaking in
        // would pull this to roughly 200.
        REQUIRE(std::abs(actual[d] - 1.0f) < 0.01f);
    }
}

TEST_CASE("attention is numerically stable against extreme scores",
          "[kernels][attention]") {
    REQUIRE_GPU();
    Device device{backend, stream};

    // The streaming softmax rescales as new maxima arrive. Large dot products
    // would overflow a naive exp, so this checks the rescaling actually works.
    constexpr u32 kHeadDim = 64;
    constexpr u32 kCapacity = 32;
    constexpr u64 kCached = 32;

    const std::vector<float> queries(kHeadDim, 10.0f);
    std::vector<float> keys(kCapacity * kHeadDim, 0.0f);
    std::vector<float> values(kCapacity * kHeadDim, 0.0f);

    // Scores ascend steeply, so the maximum keeps moving and every step forces
    // a rescale. The last position dominates.
    for (u32 position = 0; position < kCapacity; ++position) {
        for (u32 d = 0; d < kHeadDim; ++d) {
            keys[position * kHeadDim + d] = static_cast<float>(position) * 0.5f;
            values[position * kHeadDim + d] = static_cast<float>(position);
        }
    }

    auto queryBuffer = device.uploadHalf(queries, "queries");
    auto keyBuffer = device.uploadKv(keys, "keys");
    auto valueBuffer = device.uploadKv(values, "values");
    auto outputBuffer = device.empty(kHeadDim * sizeof(float), "output");

    REQUIRE(kernels
                    .attention(stream, AttentionArgs{
                                                     .queries = view(queryBuffer),
                                                     .keyCache = view(keyBuffer),
                                                     .valueCache = view(valueBuffer),
                                                     .output = view(outputBuffer),
                                                     .numHeads = 1,
                                                     .kvHeads = 1,
                                                     .headDim = kHeadDim,
                                                     .capacity = kCapacity,
                                                     .basePosition = kCached - 1})
                    .has_value());

    const auto actual = device.readHalf(*outputBuffer, kHeadDim);
    for (const float value : actual) {
        REQUIRE(std::isfinite(value));
        // The last position wins overwhelmingly.
        REQUIRE(std::abs(value - 31.0f) < 0.5f);
    }
}

TEST_CASE("attention rejects mismatched head counts", "[kernels][attention]") {
    REQUIRE_GPU();
    Device device{backend, stream};

    auto buffer = device.empty(4096, "buffer");
    const auto status = kernels.attention(
            stream, AttentionArgs{.queries = view(buffer),
                                  .keyCache = view(buffer),
                                  .valueCache = view(buffer),
                                  .output = view(buffer),
                                  .numHeads = 5,  // not a multiple of 2
                                  .kvHeads = 2,
                                  .headDim = 8,
                                  .capacity = 8});
    REQUIRE_FALSE(status.has_value());
    CHECK(status.error().code() == ErrorCode::InvalidArgument);
    CHECK_THAT(status.error().message(), Catch::Matchers::ContainsSubstring("divide evenly"));
}

// ---------------------------------------------------------------------------
// Router
// ---------------------------------------------------------------------------

TEST_CASE("router top-k matches the reference", "[kernels][router]") {
    REQUIRE_GPU();
    Device device{backend, stream};

    // The real router shape: top-8 of 128.
    constexpr u32 kExperts = 128;
    constexpr u32 kTopK = 8;

    SplitMix64 rng{301};
    const auto scores = throughHalf(randomGaussians(rng, kExperts, 3.0f));
    const auto perExpertScale = randomFloats(rng, kExperts, 0.95f, 1.05f);

    auto scoreBuffer = device.uploadHalf(scores, "scores");
    auto scaleBuffer = device.uploadBf16(perExpertScale, "per-expert-scale");
    auto indexBuffer = device.empty(kTopK * sizeof(u32), "indices");
    auto weightBuffer = device.empty(kTopK * sizeof(float), "weights");

    REQUIRE(kernels
                    .routerTopK(stream, RouterTopKArgs{.scores = view(scoreBuffer),
                                                       .perExpertScale = view(scaleBuffer),
                                                       .outIndices = view(indexBuffer),
                                                       .outWeights = view(weightBuffer),
                                                       .numExperts = kExperts,
                                                       .topK = kTopK})
                    .has_value());

    const auto actualIndices = device.readWords(*indexBuffer, kTopK);
    const auto actualWeights = device.readHalf(*weightBuffer, kTopK);

    const auto bf16Scale = [&] {
        std::vector<float> rounded(perExpertScale.size());
        for (usize i = 0; i < rounded.size(); ++i) {
            rounded[i] = toFloat(toBf16(perExpertScale[i]));
        }
        return rounded;
    }();
    const auto expected = reference::routerTopK(scores, bf16Scale, kTopK);

    CHECK(actualIndices == expected.indices);

    const auto deviation = compare(expected.weights, actualWeights);
    INFO(deviation.describe());
    CHECK(deviation.maxRelative < tolerance::kSoftmax);
}

TEST_CASE("router softmaxes only the selected experts", "[kernels][router]") {
    REQUIRE_GPU();
    Device device{backend, stream};

    // Two experts score 100 and the rest score near zero. Softmaxing over all
    // 128 before selection would give a different, smaller pair of weights.
    constexpr u32 kExperts = 128;
    constexpr u32 kTopK = 2;

    std::vector<float> scores(kExperts, 0.0f);
    scores[42] = 100.0f;
    scores[7] = 100.0f;
    const std::vector<float> perExpertScale(kExperts, 1.0f);

    auto scoreBuffer = device.uploadHalf(scores, "scores");
    auto scaleBuffer = device.uploadBf16(perExpertScale, "scale");
    auto indexBuffer = device.empty(kTopK * sizeof(u32), "indices");
    auto weightBuffer = device.empty(kTopK * sizeof(float), "weights");

    REQUIRE(kernels
                    .routerTopK(stream, RouterTopKArgs{.scores = view(scoreBuffer),
                                                       .perExpertScale = view(scaleBuffer),
                                                       .outIndices = view(indexBuffer),
                                                       .outWeights = view(weightBuffer),
                                                       .numExperts = kExperts,
                                                       .topK = kTopK})
                    .has_value());

    const auto indices = device.readWords(*indexBuffer, kTopK);
    const auto weights = device.readHalf(*weightBuffer, kTopK);

    // Equal scores, so ties resolve to the lower index first.
    CHECK(indices[0] == 7);
    CHECK(indices[1] == 42);

    // Softmax over just these two gives 0.5 each and they sum to 1.
    CHECK(std::abs(weights[0] - 0.5f) < 0.01f);
    CHECK(std::abs(weights[0] + weights[1] - 1.0f) < 0.01f);
}

TEST_CASE("the per-expert scale applies after the softmax", "[kernels][router]") {
    REQUIRE_GPU();
    Device device{backend, stream};

    constexpr u32 kExperts = 4;
    constexpr u32 kTopK = 2;

    const std::vector<float> scores{1.0f, 1.0f, -10.0f, -10.0f};
    const std::vector<float> perExpertScale{2.0f, 4.0f, 1.0f, 1.0f};

    auto scoreBuffer = device.uploadHalf(scores, "scores");
    auto scaleBuffer = device.uploadBf16(perExpertScale, "scale");
    auto indexBuffer = device.empty(kTopK * sizeof(u32), "indices");
    auto weightBuffer = device.empty(kTopK * sizeof(float), "weights");

    REQUIRE(kernels
                    .routerTopK(stream, RouterTopKArgs{.scores = view(scoreBuffer),
                                                       .perExpertScale = view(scaleBuffer),
                                                       .outIndices = view(indexBuffer),
                                                       .outWeights = view(weightBuffer),
                                                       .numExperts = kExperts,
                                                       .topK = kTopK})
                    .has_value());

    const auto weights = device.readHalf(*weightBuffer, kTopK);

    // 0.5 each after the softmax, then scaled by 2 and 4. The result
    // deliberately does not sum to one.
    CHECK(std::abs(weights[0] - 1.0f) < 0.01f);
    CHECK(std::abs(weights[1] - 2.0f) < 0.02f);
}

TEST_CASE("router rejects a top-k larger than the expert count",
          "[kernels][router]") {
    REQUIRE_GPU();
    Device device{backend, stream};

    auto buffer = device.empty(1024, "buffer");
    const auto status =
            kernels.routerTopK(stream, RouterTopKArgs{.scores = view(buffer),
                                                      .perExpertScale = view(buffer),
                                                      .outIndices = view(buffer),
                                                      .outWeights = view(buffer),
                                                      .numExperts = 4,
                                                      .topK = 8});
    REQUIRE_FALSE(status.has_value());
    CHECK(status.error().code() == ErrorCode::InvalidArgument);
}

// ---------------------------------------------------------------------------
// MoE combine
// ---------------------------------------------------------------------------

TEST_CASE("moeCombine matches the reference", "[kernels][moe]") {
    REQUIRE_GPU();
    Device device{backend, stream};

    constexpr u32 kTopK = 8;
    constexpr u32 kHidden = 2816;

    SplitMix64 rng{401};
    const auto expertOutputs = throughHalf(randomGaussians(rng, kTopK * kHidden));
    const auto weights = throughHalf(randomFloats(rng, kTopK, 0.0f, 1.0f));

    auto outputsBuffer = device.uploadHalf(expertOutputs, "expert-outputs");
    auto weightsBuffer = device.uploadHalf(weights, "weights");
    auto resultBuffer = device.empty(kHidden * sizeof(float), "combined");

    REQUIRE(kernels
                    .moeCombine(stream, MoeCombineArgs{.expertOutputs = view(outputsBuffer),
                                                       .weights = view(weightsBuffer),
                                                       .output = view(resultBuffer),
                                                       .topK = kTopK,
                                                       .hidden = kHidden})
                    .has_value());

    const auto expected = reference::moeCombine(expertOutputs, weights, kTopK, kHidden);
    const auto deviation = compare(expected, device.readHalf(*resultBuffer, kHidden));
    INFO(deviation.describe());
    CHECK(deviation.maxRelative < tolerance::kElementwise);
}

TEST_CASE("moeCombine weights each expert row separately", "[kernels][moe]") {
    REQUIRE_GPU();
    Device device{backend, stream};

    // Each expert emits a distinct constant, so the answer is a simple dot
    // product that a transposed indexing would get wrong.
    constexpr u32 kTopK = 3;
    constexpr u32 kHidden = 4;

    const std::vector<float> expertOutputs{
            1.0f, 1.0f, 1.0f, 1.0f,  // expert 0
            2.0f, 2.0f, 2.0f, 2.0f,  // expert 1
            4.0f, 4.0f, 4.0f, 4.0f,  // expert 2
    };
    const std::vector<float> weights{0.5f, 0.25f, 0.125f};

    auto outputsBuffer = device.uploadHalf(expertOutputs, "expert-outputs");
    auto weightsBuffer = device.uploadHalf(weights, "weights");
    auto resultBuffer = device.empty(kHidden * sizeof(float), "combined");

    REQUIRE(kernels
                    .moeCombine(stream, MoeCombineArgs{.expertOutputs = view(outputsBuffer),
                                                       .weights = view(weightsBuffer),
                                                       .output = view(resultBuffer),
                                                       .topK = kTopK,
                                                       .hidden = kHidden})
                    .has_value());

    // 0.5*1 + 0.25*2 + 0.125*4 = 1.5
    const auto actual = device.readHalf(*resultBuffer, kHidden);
    for (u32 d = 0; d < kHidden; ++d) {
        INFO("dim " << d);
        REQUIRE(std::abs(actual[d] - 1.5f) < 0.01f);
    }
}
