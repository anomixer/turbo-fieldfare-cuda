#include "tf/runtime/KVCache.h"

#include <algorithm>
#include <format>

#include "tf/runtime/Residency.h"

namespace tf::runtime {

Result<KVCacheManager> KVCacheManager::create(gpu::IGpuBackend& backend, const ArchInfo& arch,
                                              u64 contextLength, u64 maxPrefillTokens) {
    TF_CHECK(arch.validate());
    if (contextLength == 0) {
        return makeError(ErrorCode::InvalidArgument, "context length must be non-zero");
    }
    if (maxPrefillTokens == 0) {
        return makeError(ErrorCode::InvalidArgument, "a prefill chunk of zero tokens");
    }

    KVCacheManager manager;
    manager.contextLength_ = contextLength;
    manager.maxPrefillTokens_ = maxPrefillTokens;
    manager.layers_.reserve(static_cast<usize>(arch.numLayers));

    for (u64 layer = 0; layer < arch.numLayers; ++layer) {
        const bool full = arch.isFullAttention(layer);
        const u64 rows = kvRowsForLayer(arch, layer, contextLength, maxPrefillTokens);
        const u64 kvHeads = arch.kvHeadsFor(layer);
        const u64 headDim = arch.headDimFor(layer);

        constexpr u64 kFp16Bytes = 2;
        const u64 tensorBytes = kvHeads * rows * headDim * kFp16Bytes;

        // Keys and values share one allocation: half the allocation count, and
        // the two are always accessed together.
        TF_TRY(gpu::BufferPtr buffer,
               backend.allocate(gpu::MemoryKind::Device, tensorBytes * 2,
                                std::format("kv-layer-{:02}", layer)));

        manager.totalBytes_ += tensorBytes * 2;
        manager.layers_.push_back(LayerStorage{
                .buffer = std::move(buffer),
                .valueOffset = tensorBytes,
                .kvHeads = static_cast<u32>(kvHeads),
                .headDim = static_cast<u32>(headDim),
                .capacity = static_cast<u32>(rows),
                // Full-attention layers append; sliding layers wrap.
                .circular = !full});
    }

    return manager;
}

u64 KVCacheManager::earliestSafeRewind() const noexcept {
    u64 earliest = 0;
    for (const LayerStorage& storage : layers_) {
        if (!storage.circular) {
            // Append-only: every position ever written is still there.
            continue;
        }
        if (position_ <= storage.capacity) {
            // The ring has not wrapped, so nothing has been lost yet.
            continue;
        }
        // Valid rows are [position_ - capacity, position_). A token at P needs
        // back to P - window + 1, and window is capacity minus the headroom the
        // ring was built with. Solving for P gives this bound; the +1 keeps the
        // oldest needed row strictly inside the valid range.
        const u64 window = storage.capacity > maxPrefillTokens_
                                   ? storage.capacity - maxPrefillTokens_
                                   : storage.capacity;
        const u64 limit = position_ + window - storage.capacity;
        earliest = std::max(earliest, limit);
    }
    return earliest;
}

LayerKVCache KVCacheManager::layer(u64 index) const {
    const LayerStorage& storage = layers_[static_cast<usize>(index)];
    return LayerKVCache{
            .keys = gpu::DeviceView{.buffer = storage.buffer.get(), .offset = 0},
            .values = gpu::DeviceView{.buffer = storage.buffer.get(),
                                      .offset = storage.valueOffset},
            .kvHeads = storage.kvHeads,
            .headDim = storage.headDim,
            .capacity = storage.capacity,
            .circular = storage.circular};
}

}  // namespace tf::runtime
