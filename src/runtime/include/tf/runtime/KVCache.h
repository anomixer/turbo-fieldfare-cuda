#pragma once

#include <vector>

#include "tf/core/base/Error.h"
#include "tf/core/base/Types.h"
#include "tf/core/format/ArchInfo.h"
#include "tf/gpu/Backend.h"
#include "tf/gpu/Kernels.h"
#include "tf/runtime/Residency.h"

namespace tf::runtime {

/// One layer's slice of the KV cache, in the shape the attention kernels take.
struct LayerKVCache {
    gpu::DeviceView keys;
    gpu::DeviceView values;
    u32 kvHeads = 0;
    u32 headDim = 0;
    /// Rows per KV head.
    u32 capacity = 0;
    /// True for the sliding-window layers, whose rows wrap.
    bool circular = false;
};

/// Owns the fp16 key and value storage for every layer.
///
/// The two layer types are shaped differently and the manager keeps that
/// difference explicit rather than padding everything to the largest: sliding
/// layers get a fixed ring sized by the window, full-attention layers get rows
/// for the whole context. At 4K that is roughly 225 MiB of rings against 80 MiB
/// of linear storage.
class KVCacheManager {
public:
    KVCacheManager() = default;

    KVCacheManager(const KVCacheManager&) = delete;
    KVCacheManager& operator=(const KVCacheManager&) = delete;
    KVCacheManager(KVCacheManager&&) noexcept = default;
    KVCacheManager& operator=(KVCacheManager&&) noexcept = default;

    /// `maxPrefillTokens` sizes the sliding-window rings: a chunk writes all of
    /// its keys and values before any of its queries read them, so the ring
    /// needs at least that much headroom beyond the window or the chunk
    /// overwrites history its own early tokens still need. See
    /// kSlidingWindowHeadroomRows.
    [[nodiscard]] static Result<KVCacheManager> create(
            gpu::IGpuBackend& backend, const ArchInfo& arch, u64 contextLength,
            u64 maxPrefillTokens = kDefaultPrefillTokens);

    [[nodiscard]] LayerKVCache layer(u64 index) const;

    [[nodiscard]] u64 layerCount() const noexcept { return layers_.size(); }
    [[nodiscard]] u64 contextLength() const noexcept { return contextLength_; }

    /// Largest prefill chunk the rings were sized for. prefillChunk refuses
    /// anything wider rather than corrupting the oldest visible history.
    [[nodiscard]] u64 maxPrefillTokens() const noexcept { return maxPrefillTokens_; }
    [[nodiscard]] u64 totalBytes() const noexcept { return totalBytes_; }

    /// Tokens written so far. The attention kernels need this as the cached
    /// length, and the KV writer needs it as the next position.
    [[nodiscard]] u64 position() const noexcept { return position_; }

    /// Advances after a token has been written to every layer.
    void advance(u64 tokens = 1) noexcept { position_ += tokens; }

    /// Rewinds to the start of a new sequence. Does not clear the buffers:
    /// stale rows beyond `position` are never read, because the attention
    /// kernels are bounded by cachedLength.
    void reset() noexcept { position_ = 0; }

    /// Rewinds to `position`, keeping everything before it.
    ///
    /// This is what lets a second request reuse the prefix it shares with the
    /// first instead of reprocessing it. Must not be called with a position
    /// below earliestSafeRewind(); see there for why.
    void rewindTo(u64 position) noexcept {
        position_ = position < position_ ? position : position_;
    }

    /// Earliest position a rewind can target while leaving the tokens after it
    /// able to see the history they need.
    ///
    /// The sliding-window rings hold only the most recent `capacity` positions.
    /// Rewinding to P and re-prefilling from there means the token at P attends
    /// back to P - window + 1, and those rows must still be in the ring. Once
    /// the sequence is longer than a ring, that bounds the rewind to roughly the
    /// headroom beyond the window - a hundred-odd tokens, not thousands.
    ///
    /// Rewinding past this point does not fail loudly: the model simply attends
    /// over rows belonging to a different part of the sequence and produces
    /// confident nonsense. So callers must ask rather than assume.
    [[nodiscard]] u64 earliestSafeRewind() const noexcept;

    /// Highest position this cache can hold. Writing beyond it would overrun
    /// the full-attention layers, which do not wrap.
    [[nodiscard]] bool canAccept(u64 tokens) const noexcept {
        return position_ + tokens <= contextLength_;
    }

private:
    struct LayerStorage {
        gpu::BufferPtr buffer;  ///< keys then values in one allocation
        u64 valueOffset = 0;
        u32 kvHeads = 0;
        u32 headDim = 0;
        u32 capacity = 0;
        bool circular = false;
    };

    std::vector<LayerStorage> layers_;
    u64 contextLength_ = 0;
    u64 maxPrefillTokens_ = kDefaultPrefillTokens;
    u64 totalBytes_ = 0;
    u64 position_ = 0;
};

}  // namespace tf::runtime
