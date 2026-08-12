#pragma once

#include <filesystem>
#include <memory>
#include <mutex>
#include <span>
#include <vector>

#include "tf/core/base/Error.h"
#include "tf/core/base/Types.h"
#include "tf/core/format/GTurbo.h"
#include "tf/core/io/File.h"
#include "tf/gpu/Backend.h"
#include "tf/gpu/Kernels.h"

namespace tf::runtime {

/// Which expert a slot currently holds.
inline constexpr u32 kEmptySlot = 0xFFFFFFFFu;

enum class EvictionPolicy {
    /// Least frequently used, ties broken by recency. Upstream's default, and
    /// the better fit: expert popularity is stable across a generation, so
    /// usage count predicts future need better than recency alone.
    Lfu,
    /// Least recently used.
    Lru,
};

/// Outcome of planning one layer's expert fetch.
struct ExpertCachePlan {
    u64 layer = 0;
    /// Requested experts, in the order the caller asked for them.
    std::vector<u32> experts;
    /// Slot assigned to each requested expert, parallel to `experts`.
    std::vector<u32> slots;
    /// Indices into `experts` that were already resident.
    std::vector<u32> hits;
    /// Indices into `experts` that must be read from disk.
    std::vector<u32> misses;

    [[nodiscard]] bool allHit() const noexcept { return misses.empty(); }
};

struct StreamerStats {
    u64 requests = 0;
    u64 hits = 0;
    u64 misses = 0;
    u64 bytesRead = 0;
    u64 readCount = 0;

    [[nodiscard]] double hitRate() const {
        return requests == 0 ? 0.0 : static_cast<double>(hits) / static_cast<double>(requests);
    }
};

struct StreamerOptions {
    u32 slotsPerLayer = 16;
    EvictionPolicy policy = EvictionPolicy::Lfu;

    /// Buffered reads let the Windows page cache serve repeat hits at RAM
    /// speed, at the cost of one memcpy into the staging buffer. With 63 GiB of
    /// RAM against 12 GiB of experts, the whole set stays cached after warmup,
    /// so this is the right default here. Unbuffered DMAs straight into
    /// sector-aligned pinned memory but bypasses the cache, capping throughput
    /// at SSD speed forever.
    bool unbufferedReads = false;

    /// Threads issuing reads. Misses within a layer are independent, so they
    /// overlap; beyond a handful the NVMe queue is already saturated.
    u32 readThreads = 4;
};

/// Streams routed experts from the packed layer files into VRAM slots.
///
/// One instance owns every streamed layer's slots and file handles. Layers the
/// residency planner marked resident are not registered here at all - the
/// runtime reads those directly from their permanent allocation.
///
/// Use is two-phase, mirroring the decode loop: `plan` runs on the CPU as soon
/// as the router's expert ids come back, and `fetch` performs the reads and
/// uploads while the shared-expert branch occupies the GPU.
class ExpertStreamer {
public:
    // Every special member is out of line: Impl is incomplete here, and an
    // inline defaulted constructor would still need ~unique_ptr<Impl> to unwind
    // if a later member threw.
    ExpertStreamer();
    ~ExpertStreamer();

    ExpertStreamer(const ExpertStreamer&) = delete;
    ExpertStreamer& operator=(const ExpertStreamer&) = delete;
    ExpertStreamer(ExpertStreamer&&) noexcept;
    ExpertStreamer& operator=(ExpertStreamer&&) noexcept;

    /// Opens the layer files listed in `layout` and allocates slots for each
    /// layer in `streamedLayers`.
    [[nodiscard]] static Result<ExpertStreamer> create(
            gpu::IGpuBackend& backend, const std::filesystem::path& installDir,
            const gturbo::ExpertLayout& layout, std::span<const u64> streamedLayers,
            const StreamerOptions& options = {});

    /// Decides which requested experts are already resident and which slots the
    /// misses will occupy. Updates the usage statistics the eviction policy
    /// reads, and reserves the chosen slots so a second plan for the same layer
    /// cannot pick them.
    ///
    /// Never evicts a slot holding an expert this same plan needs.
    [[nodiscard]] Result<ExpertCachePlan> plan(u64 layer, std::span<const u32> experts);

    /// Reads every miss and uploads it into its slot. Reads run across the
    /// thread pool; uploads are enqueued on `stream`, so the caller must
    /// synchronize before the kernels read the slots.
    [[nodiscard]] Status fetch(const ExpertCachePlan& plan, gpu::Stream& stream);

    /// Where a slot's expert blob lives, for binding into a kernel.
    [[nodiscard]] gpu::DeviceView slotView(u64 layer, u32 slot) const;

    [[nodiscard]] bool isStreamed(u64 layer) const;

    [[nodiscard]] const StreamerStats& stats() const noexcept { return stats_; }
    void resetStats();

    /// Empties every slot, as at the start of a new sequence.
    void clearCache();

    [[nodiscard]] u64 slotBytes() const noexcept { return blobBytes_; }
    [[nodiscard]] u32 slotsPerLayer() const noexcept { return slotsPerLayer_; }

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;

    u64 blobBytes_ = 0;
    u32 slotsPerLayer_ = 0;
    StreamerStats stats_;
};

}  // namespace tf::runtime
