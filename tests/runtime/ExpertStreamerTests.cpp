// Expert streaming and KV cache allocation.
//
// A synthetic packed_experts directory is used rather than the real 13 GiB
// install, so these run fast and deterministically. Each expert blob is filled
// with a pattern derived from its layer and index, which means a slot holding
// the wrong expert is detected exactly, not statistically.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <windows.h>

#include <filesystem>
#include <format>
#include <set>
#include <vector>

#include "tf/core/io/File.h"
#include "tf/gpu/Backend.h"
#include "tf/runtime/ExpertStreamer.h"
#include "tf/runtime/KVCache.h"
#include "tf/runtime/Residency.h"

using namespace tf;
using namespace tf::runtime;

namespace {

struct Harness {
    gpu::BackendPtr backend;
    gpu::StreamPtr stream;

    static Harness* instance() {
        static Harness harness = [] {
            Harness result;
            if (gpu::compiledBackends().empty()) {
                return result;
            }
            auto created = gpu::createBackend(gpu::compiledBackends().front());
            if (!created) {
                return result;
            }
            result.backend = std::move(*created);
            auto stream = result.backend->createStream("streamer-tests");
            if (stream) {
                result.stream = std::move(*stream);
            }
            return result;
        }();
        return harness.backend && harness.stream ? &harness : nullptr;
    }
};

// backend and stream are marked maybe_unused: several cases exercise only the
// planning or validation paths and never touch the GPU.
#define REQUIRE_GPU()                                                  \
    Harness* harness = Harness::instance();                            \
    if (harness == nullptr) {                                          \
        SKIP("no usable GPU backend on this machine");                 \
    }                                                                  \
    [[maybe_unused]] gpu::IGpuBackend& backend = *harness->backend;     \
    [[maybe_unused]] gpu::Stream& stream = *harness->stream

class TempDir {
public:
    TempDir() {
        static int counter = 0;
        path_ = std::filesystem::temp_directory_path() /
                std::format("tf-streamer-{}-{}", ::GetCurrentProcessId(), ++counter);
        std::filesystem::create_directories(path_);
    }
    ~TempDir() {
        std::error_code ec;
        std::filesystem::remove_all(path_, ec);
    }
    TempDir(const TempDir&) = delete;
    TempDir& operator=(const TempDir&) = delete;

    [[nodiscard]] const std::filesystem::path& path() const { return path_; }

private:
    std::filesystem::path path_;
};

// Small enough to write quickly, but the stride is still sector-aligned so the
// unbuffered path can be exercised.
constexpr u64 kLayers = 4;
constexpr u64 kExpertsPerLayer = 8;
constexpr u64 kBlobBytes = 8192;

gturbo::ExpertLayout makeLayout() {
    gturbo::ExpertLayout layout;
    layout.numLayers = kLayers;
    layout.expertsPerLayer = kExpertsPerLayer;
    layout.blobBytes = kBlobBytes;
    layout.alignment = align::kSector;
    layout.stride = alignUp(kBlobBytes, align::kSector);
    for (u64 layer = 0; layer < kLayers; ++layer) {
        layout.layerFiles.push_back(gturbo::ExpertLayout::layerFileName(layer));
    }
    layout.components.push_back(gturbo::ExpertComponent{
            .role = "all", .dtype = DType::U32, .shape = {}, .offset = 0,
            .size = kBlobBytes});
    return layout;
}

/// Distinct per (layer, expert, offset) so a misdirected fetch is unambiguous.
u8 patternByte(u64 layer, u64 expert, u64 offset) {
    return static_cast<u8>((layer * 61 + expert * 17 + offset * 7 + 3) & 0xFF);
}

void writeExpertFiles(const std::filesystem::path& installDir,
                      const gturbo::ExpertLayout& layout) {
    const auto expertsDir = installDir / gturbo::kExpertsDir;
    std::filesystem::create_directories(expertsDir);

    for (u64 layer = 0; layer < layout.numLayers; ++layer) {
        std::vector<u8> contents(static_cast<usize>(layout.layerFileBytes()), 0);
        for (u64 expert = 0; expert < layout.expertsPerLayer; ++expert) {
            const auto base = static_cast<usize>(expert * layout.stride);
            for (u64 i = 0; i < layout.blobBytes; ++i) {
                contents[base + static_cast<usize>(i)] = patternByte(layer, expert, i);
            }
        }
        REQUIRE(io::writeFileAtomic(expertsDir / layout.layerFiles[static_cast<usize>(layer)],
                                    contents)
                        .has_value());
    }
}

/// Reads a slot back off the device.
std::vector<u8> readSlot(gpu::IGpuBackend& backend, gpu::Stream& stream,
                         const ExpertStreamer& streamer, u64 layer, u32 slot, u64 bytes) {
    const gpu::DeviceView slotView = streamer.slotView(layer, slot);
    REQUIRE(slotView.valid());

    std::vector<u8> data(static_cast<usize>(bytes));
    REQUIRE(backend
                    .enqueueDownload(stream, MutableByteSpan{data.data(), data.size()},
                                     *slotView.buffer, slotView.offset)
                    .has_value());
    REQUIRE(stream.synchronize().has_value());
    return data;
}

std::vector<u8> expectedBlob(u64 layer, u64 expert, u64 bytes) {
    std::vector<u8> data(static_cast<usize>(bytes));
    for (u64 i = 0; i < bytes; ++i) {
        data[static_cast<usize>(i)] = patternByte(layer, expert, i);
    }
    return data;
}

std::vector<u64> allLayers(u64 count) {
    std::vector<u64> layers(static_cast<usize>(count));
    for (u64 i = 0; i < count; ++i) {
        layers[static_cast<usize>(i)] = i;
    }
    return layers;
}

}  // namespace

TEST_CASE("a fetched expert lands in its slot byte for byte", "[streamer]") {
    REQUIRE_GPU();
    const TempDir temp;
    const auto layout = makeLayout();
    writeExpertFiles(temp.path(), layout);

    const auto layers = allLayers(kLayers);
    auto streamer = ExpertStreamer::create(backend, temp.path(), layout, layers,
                                          StreamerOptions{.slotsPerLayer = 4});
    REQUIRE(streamer.has_value());

    const std::vector<u32> experts{3, 5};
    const auto plan = streamer->plan(2, experts);
    REQUIRE(plan.has_value());
    CHECK(plan->misses.size() == 2);
    CHECK(plan->hits.empty());

    REQUIRE(streamer->fetch(*plan, stream).has_value());
    REQUIRE(stream.synchronize().has_value());

    for (usize i = 0; i < experts.size(); ++i) {
        const auto actual =
                readSlot(backend, stream, *streamer, 2, plan->slots[i], kBlobBytes);
        INFO("expert " << experts[i] << " in slot " << plan->slots[i]);
        REQUIRE(actual == expectedBlob(2, experts[i], kBlobBytes));
    }
}

TEST_CASE("layers are addressed independently", "[streamer]") {
    REQUIRE_GPU();
    const TempDir temp;
    const auto layout = makeLayout();
    writeExpertFiles(temp.path(), layout);

    const auto layers = allLayers(kLayers);
    auto streamer = ExpertStreamer::create(backend, temp.path(), layout, layers,
                                          StreamerOptions{.slotsPerLayer = 4});
    REQUIRE(streamer.has_value());

    // The same expert index in different layers holds different data, so a
    // kernel reading the wrong layer's file would be caught here.
    for (u64 layer = 0; layer < kLayers; ++layer) {
        const std::vector<u32> experts{1};
        const auto plan = streamer->plan(layer, experts);
        REQUIRE(plan.has_value());
        REQUIRE(streamer->fetch(*plan, stream).has_value());
        REQUIRE(stream.synchronize().has_value());

        const auto actual =
                readSlot(backend, stream, *streamer, layer, plan->slots[0], kBlobBytes);
        INFO("layer " << layer);
        REQUIRE(actual == expectedBlob(layer, 1, kBlobBytes));
    }
}

TEST_CASE("a repeated expert hits the cache", "[streamer]") {
    REQUIRE_GPU();
    const TempDir temp;
    const auto layout = makeLayout();
    writeExpertFiles(temp.path(), layout);

    auto streamer = ExpertStreamer::create(backend, temp.path(), layout, allLayers(kLayers),
                                          StreamerOptions{.slotsPerLayer = 4});
    REQUIRE(streamer.has_value());

    const std::vector<u32> experts{2, 6};
    {
        const auto first = streamer->plan(0, experts);
        REQUIRE(first.has_value());
        CHECK(first->misses.size() == 2);
        REQUIRE(streamer->fetch(*first, stream).has_value());
        REQUIRE(stream.synchronize().has_value());
    }

    const u64 readsAfterFirst = streamer->stats().readCount;

    const auto second = streamer->plan(0, experts);
    REQUIRE(second.has_value());
    CHECK(second->allHit());
    CHECK(second->hits.size() == 2);
    CHECK(second->misses.empty());

    REQUIRE(streamer->fetch(*second, stream).has_value());
    // A hit must not touch the disk at all.
    CHECK(streamer->stats().readCount == readsAfterFirst);

    CHECK(streamer->stats().requests == 4);
    CHECK(streamer->stats().hits == 2);
    CHECK(streamer->stats().misses == 2);
    CHECK(streamer->stats().hitRate() == 0.5);
}

TEST_CASE("a duplicate expert within one request shares its slot", "[streamer]") {
    REQUIRE_GPU();
    const TempDir temp;
    const auto layout = makeLayout();
    writeExpertFiles(temp.path(), layout);

    auto streamer = ExpertStreamer::create(backend, temp.path(), layout, allLayers(kLayers),
                                          StreamerOptions{.slotsPerLayer = 4});
    REQUIRE(streamer.has_value());

    // The router can select the same expert twice; fetching it twice would
    // waste a read and a slot.
    const std::vector<u32> experts{3, 3, 5};
    const auto plan = streamer->plan(0, experts);
    REQUIRE(plan.has_value());

    CHECK(plan->slots[0] == plan->slots[1]);
    CHECK(plan->slots[2] != plan->slots[0]);
    CHECK(plan->misses.size() == 2);  // expert 3 once, expert 5 once

    REQUIRE(streamer->fetch(*plan, stream).has_value());
    REQUIRE(stream.synchronize().has_value());

    CHECK(readSlot(backend, stream, *streamer, 0, plan->slots[0], kBlobBytes) ==
          expectedBlob(0, 3, kBlobBytes));
}

TEST_CASE("more distinct experts than slots forces eviction", "[streamer]") {
    REQUIRE_GPU();
    const TempDir temp;
    const auto layout = makeLayout();
    writeExpertFiles(temp.path(), layout);

    constexpr u32 kSlots = 2;
    auto streamer = ExpertStreamer::create(backend, temp.path(), layout, allLayers(kLayers),
                                          StreamerOptions{.slotsPerLayer = kSlots});
    REQUIRE(streamer.has_value());

    // Walk through more experts than there are slots; whatever survives must
    // still be correct, and every fetch must land the right bytes.
    for (u32 expert = 0; expert < kExpertsPerLayer; ++expert) {
        const std::vector<u32> experts{expert};
        const auto plan = streamer->plan(0, experts);
        REQUIRE(plan.has_value());
        REQUIRE(streamer->fetch(*plan, stream).has_value());
        REQUIRE(stream.synchronize().has_value());

        const auto actual =
                readSlot(backend, stream, *streamer, 0, plan->slots[0], kBlobBytes);
        INFO("expert " << expert << " slot " << plan->slots[0]);
        REQUIRE(actual == expectedBlob(0, expert, kBlobBytes));
        REQUIRE(plan->slots[0] < kSlots);
    }
}

TEST_CASE("a plan never evicts a slot it is itself using", "[streamer]") {
    REQUIRE_GPU();
    const TempDir temp;
    const auto layout = makeLayout();
    writeExpertFiles(temp.path(), layout);

    // Exactly as many slots as experts requested: every slot is claimed by this
    // plan, so a naive eviction could overwrite one of its own.
    constexpr u32 kSlots = 4;
    auto streamer = ExpertStreamer::create(backend, temp.path(), layout, allLayers(kLayers),
                                          StreamerOptions{.slotsPerLayer = kSlots});
    REQUIRE(streamer.has_value());

    const std::vector<u32> experts{0, 1, 2, 3};
    const auto plan = streamer->plan(1, experts);
    REQUIRE(plan.has_value());

    // Four distinct experts must occupy four distinct slots.
    const std::set<u32> distinct{plan->slots.begin(), plan->slots.end()};
    CHECK(distinct.size() == experts.size());

    REQUIRE(streamer->fetch(*plan, stream).has_value());
    REQUIRE(stream.synchronize().has_value());

    for (usize i = 0; i < experts.size(); ++i) {
        INFO("expert " << experts[i]);
        REQUIRE(readSlot(backend, stream, *streamer, 1, plan->slots[i], kBlobBytes) ==
                expectedBlob(1, experts[i], kBlobBytes));
    }
}

TEST_CASE("LFU keeps the frequently used expert", "[streamer]") {
    REQUIRE_GPU();
    const TempDir temp;
    const auto layout = makeLayout();
    writeExpertFiles(temp.path(), layout);

    auto streamer =
            ExpertStreamer::create(backend, temp.path(), layout, allLayers(kLayers),
                                   StreamerOptions{.slotsPerLayer = 2,
                                                   .policy = EvictionPolicy::Lfu});
    REQUIRE(streamer.has_value());

    const auto request = [&](u32 expert) {
        const std::vector<u32> experts{expert};
        const auto plan = streamer->plan(0, experts);
        REQUIRE(plan.has_value());
        REQUIRE(streamer->fetch(*plan, stream).has_value());
        REQUIRE(stream.synchronize().has_value());
        return plan->allHit();
    };

    // Expert 1 is used repeatedly; expert 2 once.
    request(1);
    request(1);
    request(1);
    request(2);

    // Bringing in expert 3 must evict the less-used expert 2, not expert 1.
    request(3);
    CHECK(request(1));  // still resident
}

TEST_CASE("LRU keeps the recently used expert", "[streamer]") {
    REQUIRE_GPU();
    const TempDir temp;
    const auto layout = makeLayout();
    writeExpertFiles(temp.path(), layout);

    auto streamer =
            ExpertStreamer::create(backend, temp.path(), layout, allLayers(kLayers),
                                   StreamerOptions{.slotsPerLayer = 2,
                                                   .policy = EvictionPolicy::Lru});
    REQUIRE(streamer.has_value());

    const auto request = [&](u32 expert) {
        const std::vector<u32> experts{expert};
        const auto plan = streamer->plan(0, experts);
        REQUIRE(plan.has_value());
        REQUIRE(streamer->fetch(*plan, stream).has_value());
        REQUIRE(stream.synchronize().has_value());
        return plan->allHit();
    };

    request(1);
    request(1);
    request(1);
    request(2);  // most recent

    // Under LRU the stale expert 1 goes, despite its higher use count - which
    // is exactly how LRU differs from LFU here.
    request(3);
    CHECK(request(2));
}

TEST_CASE("clearing the cache drops every slot", "[streamer]") {
    REQUIRE_GPU();
    const TempDir temp;
    const auto layout = makeLayout();
    writeExpertFiles(temp.path(), layout);

    auto streamer = ExpertStreamer::create(backend, temp.path(), layout, allLayers(kLayers),
                                          StreamerOptions{.slotsPerLayer = 4});
    REQUIRE(streamer.has_value());

    const std::vector<u32> experts{1, 2};
    {
        const auto plan = streamer->plan(0, experts);
        REQUIRE(streamer->fetch(*plan, stream).has_value());
        REQUIRE(stream.synchronize().has_value());
    }
    CHECK(streamer->plan(0, experts)->allHit());

    streamer->clearCache();
    const auto after = streamer->plan(0, experts);
    REQUIRE(after.has_value());
    CHECK(after->misses.size() == 2);
}

TEST_CASE("many experts fetched at once all land correctly", "[streamer]") {
    REQUIRE_GPU();
    const TempDir temp;
    const auto layout = makeLayout();
    writeExpertFiles(temp.path(), layout);

    // 8 misses in one plan exercises the parallel read path across the pool.
    auto streamer = ExpertStreamer::create(backend, temp.path(), layout, allLayers(kLayers),
                                          StreamerOptions{.slotsPerLayer = 8,
                                                          .readThreads = 4});
    REQUIRE(streamer.has_value());

    const std::vector<u32> experts{0, 1, 2, 3, 4, 5, 6, 7};
    const auto plan = streamer->plan(3, experts);
    REQUIRE(plan.has_value());
    CHECK(plan->misses.size() == 8);

    REQUIRE(streamer->fetch(*plan, stream).has_value());
    REQUIRE(stream.synchronize().has_value());

    // Every blob must be in the right slot: a race in the staging assignment
    // would show up as two slots holding the same expert.
    for (usize i = 0; i < experts.size(); ++i) {
        INFO("expert " << experts[i] << " slot " << plan->slots[i]);
        REQUIRE(readSlot(backend, stream, *streamer, 3, plan->slots[i], kBlobBytes) ==
                expectedBlob(3, experts[i], kBlobBytes));
    }
}

TEST_CASE("repeated fetches reuse staging safely", "[streamer]") {
    REQUIRE_GPU();
    const TempDir temp;
    const auto layout = makeLayout();
    writeExpertFiles(temp.path(), layout);

    auto streamer = ExpertStreamer::create(backend, temp.path(), layout, allLayers(kLayers),
                                          StreamerOptions{.slotsPerLayer = 2});
    REQUIRE(streamer.has_value());

    // Alternating experts forces an eviction and a fresh read every round, so
    // staging is overwritten while the previous upload may still be in flight.
    // Wrong synchronization here corrupts a blob.
    for (u32 round = 0; round < 24; ++round) {
        const u32 expert = round % kExpertsPerLayer;
        const std::vector<u32> experts{expert};
        const auto plan = streamer->plan(0, experts);
        REQUIRE(plan.has_value());
        REQUIRE(streamer->fetch(*plan, stream).has_value());
        REQUIRE(stream.synchronize().has_value());

        INFO("round " << round << " expert " << expert);
        REQUIRE(readSlot(backend, stream, *streamer, 0, plan->slots[0], kBlobBytes) ==
                expectedBlob(0, expert, kBlobBytes));
    }
}

TEST_CASE("only registered layers are streamed", "[streamer]") {
    REQUIRE_GPU();
    const TempDir temp;
    const auto layout = makeLayout();
    writeExpertFiles(temp.path(), layout);

    // Layers 0 and 1 stream; 2 and 3 are resident and so not registered.
    const std::vector<u64> streamed{0, 1};
    auto streamer = ExpertStreamer::create(backend, temp.path(), layout, streamed,
                                           StreamerOptions{.slotsPerLayer = 4});
    REQUIRE(streamer.has_value());

    CHECK(streamer->isStreamed(0));
    CHECK(streamer->isStreamed(1));
    CHECK_FALSE(streamer->isStreamed(2));

    const std::vector<u32> experts{1};
    const auto refused = streamer->plan(2, experts);
    REQUIRE_FALSE(refused.has_value());
    CHECK(refused.error().code() == ErrorCode::InvalidArgument);
    CHECK_THAT(refused.error().message(), Catch::Matchers::ContainsSubstring("resident"));
}

TEST_CASE("invalid streamer configurations are refused", "[streamer]") {
    REQUIRE_GPU();
    const TempDir temp;
    const auto layout = makeLayout();
    writeExpertFiles(temp.path(), layout);

    SECTION("zero slots") {
        CHECK_FALSE(ExpertStreamer::create(backend, temp.path(), layout, allLayers(kLayers),
                                           StreamerOptions{.slotsPerLayer = 0})
                            .has_value());
    }

    SECTION("more slots than experts") {
        CHECK_FALSE(ExpertStreamer::create(backend, temp.path(), layout, allLayers(kLayers),
                                           StreamerOptions{.slotsPerLayer = 99})
                            .has_value());
    }

    SECTION("layer outside the layout") {
        const std::vector<u64> layers{99};
        CHECK_FALSE(ExpertStreamer::create(backend, temp.path(), layout, layers,
                                           StreamerOptions{.slotsPerLayer = 4})
                            .has_value());
    }

    SECTION("more experts requested than slots") {
        auto streamer =
                ExpertStreamer::create(backend, temp.path(), layout, allLayers(kLayers),
                                       StreamerOptions{.slotsPerLayer = 2});
        REQUIRE(streamer.has_value());
        const std::vector<u32> experts{0, 1, 2, 3};
        const auto plan = streamer->plan(0, experts);
        REQUIRE_FALSE(plan.has_value());
        CHECK_THAT(plan.error().message(), Catch::Matchers::ContainsSubstring("slots"));
    }

    SECTION("expert outside the layer") {
        auto streamer =
                ExpertStreamer::create(backend, temp.path(), layout, allLayers(kLayers),
                                       StreamerOptions{.slotsPerLayer = 4});
        REQUIRE(streamer.has_value());
        const std::vector<u32> experts{99};
        CHECK_FALSE(streamer->plan(0, experts).has_value());
    }
}

TEST_CASE("unbuffered reads work when the stride is aligned", "[streamer]") {
    REQUIRE_GPU();
    const TempDir temp;
    const auto layout = makeLayout();
    writeExpertFiles(temp.path(), layout);

    // The stride is sector-aligned by construction, so this path is available.
    auto streamer = ExpertStreamer::create(backend, temp.path(), layout, allLayers(kLayers),
                                          StreamerOptions{.slotsPerLayer = 4,
                                                          .unbufferedReads = true});
    REQUIRE(streamer.has_value());

    const std::vector<u32> experts{2, 4};
    const auto plan = streamer->plan(0, experts);
    REQUIRE(plan.has_value());
    REQUIRE(streamer->fetch(*plan, stream).has_value());
    REQUIRE(stream.synchronize().has_value());

    for (usize i = 0; i < experts.size(); ++i) {
        INFO("expert " << experts[i]);
        REQUIRE(readSlot(backend, stream, *streamer, 0, plan->slots[i], kBlobBytes) ==
                expectedBlob(0, experts[i], kBlobBytes));
    }
}

// ---------------------------------------------------------------------------
// KV cache
// ---------------------------------------------------------------------------

namespace {

ArchInfo miniArch() {
    ArchInfo arch;
    arch.hiddenSize = 128;
    arch.numLayers = 6;
    arch.vocabSize = 256;
    arch.maxPositionEmbeddings = 4096;
    arch.numHeads = 4;
    arch.numKVHeads = 2;
    arch.numGlobalKVHeads = 1;
    arch.headDim = 32;
    arch.globalHeadDim = 64;
    arch.slidingWindow = 64;
    arch.attentionKEqV = true;
    arch.intermediateSize = 64;
    arch.moeIntermediateSize = 64;
    arch.numExperts = 8;
    arch.topKExperts = 2;
    arch.hiddenActivation = "gelu_pytorch_tanh";
    arch.rmsNormEps = 1e-6;
    arch.tieWordEmbeddings = true;
    arch.slidingRopeTheta = 10000.0;
    arch.fullRopeTheta = 1000000.0;
    arch.partialRotaryFactor = 0.25;
    arch.bosTokenId = 2;
    arch.eosTokenIds = {1};
    for (u64 layer = 0; layer < 6; ++layer) {
        arch.layerTypes.push_back(layer % 6 == 5 ? AttentionKind::Full
                                                 : AttentionKind::Sliding);
    }
    return arch;
}

}  // namespace

TEST_CASE("the KV cache allocates per-layer shapes", "[kvcache]") {
    REQUIRE_GPU();
    const ArchInfo arch = miniArch();
    constexpr u64 kContext = 256;

    auto cache = KVCacheManager::create(backend, arch, kContext);
    REQUIRE(cache.has_value());
    CHECK(cache->layerCount() == 6);
    CHECK(cache->contextLength() == kContext);

    // Sliding layers: window 64 plus headroom, 2 KV heads of 32.
    const auto sliding = cache->layer(0);
    CHECK(sliding.circular);
    CHECK(sliding.capacity == 64 + kSlidingWindowHeadroomRows);
    CHECK(sliding.kvHeads == 2);
    CHECK(sliding.headDim == 32);

    // Full-attention layer: the whole context, 1 KV head of 64.
    const auto full = cache->layer(5);
    CHECK_FALSE(full.circular);
    CHECK(full.capacity == kContext);
    CHECK(full.kvHeads == 1);
    CHECK(full.headDim == 64);

    // Keys and values share one allocation but must not alias.
    CHECK(sliding.keys.buffer == sliding.values.buffer);
    CHECK(sliding.values.offset > sliding.keys.offset);
    CHECK(sliding.values.offset ==
          u64{sliding.kvHeads} * sliding.capacity * sliding.headDim * 2);
}

TEST_CASE("the KV cache total matches the residency estimate", "[kvcache]") {
    REQUIRE_GPU();
    const ArchInfo arch = miniArch();
    constexpr u64 kContext = 256;

    auto cache = KVCacheManager::create(backend, arch, kContext);
    REQUIRE(cache.has_value());

    // The planner budgets against this figure, so the two must agree exactly.
    u64 expected = 0;
    for (u64 layer = 0; layer < arch.numLayers; ++layer) {
        expected += kvCacheBytesForLayer(arch, layer, kContext);
    }
    CHECK(cache->totalBytes() == expected);
}

TEST_CASE("the KV cache tracks position and capacity", "[kvcache]") {
    REQUIRE_GPU();
    const ArchInfo arch = miniArch();

    auto cache = KVCacheManager::create(backend, arch, 16);
    REQUIRE(cache.has_value());

    CHECK(cache->position() == 0);
    CHECK(cache->canAccept(16));
    CHECK_FALSE(cache->canAccept(17));

    cache->advance(10);
    CHECK(cache->position() == 10);
    CHECK(cache->canAccept(6));
    // The full-attention layers do not wrap, so writing past the context would
    // overrun them.
    CHECK_FALSE(cache->canAccept(7));

    cache->reset();
    CHECK(cache->position() == 0);
    CHECK(cache->canAccept(16));
}

TEST_CASE("a zero context length is refused", "[kvcache]") {
    REQUIRE_GPU();
    CHECK_FALSE(KVCacheManager::create(backend, miniArch(), 0).has_value());
}
