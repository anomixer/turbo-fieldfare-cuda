#pragma once

#include <string>
#include <vector>

#include "tf/core/base/Error.h"
#include "tf/core/base/Types.h"
#include "tf/core/format/ArchInfo.h"
#include "tf/core/format/GTurbo.h"

namespace tf::runtime {

/// Prefill chunk assumed when none is given. Kept equal to the headroom floor
/// so the default configuration needs no extra KV rows.
inline constexpr u64 kDefaultPrefillTokens = 128;

/// How one layer's routed experts are held.
struct LayerResidency {
    /// True when all `numExperts` experts for this layer sit in VRAM and no
    /// streaming happens for it.
    bool expertsResident = false;
    /// Expert slots allocated when streaming. Zero when resident.
    u32 slots = 0;

    friend bool operator==(const LayerResidency&, const LayerResidency&) = default;
};

/// Floor for the scratch reservation, covering the staging ring and the
/// allocator slack that the forward runner's own buffers do not account for.
inline constexpr u64 kMinimumScratchBytes = 64ull * 1024 * 1024;

struct ResidencyBudget {
    /// Device memory the plan may use. The planner works to this figure rather
    /// than to the card's total, so it can be set below what the hardware has -
    /// which is how the streamed path stays testable on a machine whose VRAM
    /// would otherwise hold the whole model.
    u64 deviceBytes = 0;

    /// Tokens of context the KV cache must hold.
    u64 contextLength = 4096;

    /// Expert slots per streamed layer, and the single most valuable knob when
    /// anything streams: every slot is a cached expert that does not cross
    /// PCIe. Measured on a 6 GiB budget, 8 slots gave 9.8 tok/s and 40 gave
    /// 26.0. Upstream uses 16; 32 is the largest that still fits a 6 GiB budget
    /// with a 4096 token context, and the planner reduces it when it does not
    /// fit rather than refusing to start.
    u32 slotsPerStreamedLayer = 32;

    /// Prompt tokens the prefill path batches at once. Every per-token
    /// intermediate is allocated this wide, so it sets `scratchBytes` when that
    /// is left at zero.
    u32 maxPrefillTokens = 128;

    /// Reserved for activations, scratch and the staging ring. Zero means
    /// "work it out", which is what callers should leave it at: the plan then
    /// takes the larger of kMinimumScratchBytes and what a prefill chunk of
    /// maxPrefillTokens actually needs.
    u64 scratchBytes = 0;
};

struct ResidencyPlan {
    /// Slots each streamed layer actually got, which is the requested figure
    /// unless the budget could not hold it.
    u32 slotsPerStreamedLayer = 0;
    bool slotsWereReduced = false;

    u64 coreBytes = 0;      ///< resident weights, always in VRAM
    u64 kvCacheBytes = 0;   ///< sized by contextLength
    u64 expertBytes = 0;    ///< resident experts plus streamed slots
    u64 scratchBytes = 0;

    std::vector<LayerResidency> layers;

    /// Bytes the expert streamer must move per token in the worst case, when
    /// every routed expert on a streamed layer misses its cache.
    u64 worstCaseStreamedBytesPerToken = 0;

    [[nodiscard]] u64 totalBytes() const {
        return coreBytes + kvCacheBytes + expertBytes + scratchBytes;
    }

    [[nodiscard]] u64 residentLayerCount() const;
    [[nodiscard]] u64 streamedLayerCount() const;

    /// True when nothing streams, which is what a large card produces. Correct,
    /// but it means the streaming path gets no coverage in that configuration.
    [[nodiscard]] bool isFullyResident() const { return streamedLayerCount() == 0; }

    /// Human-readable breakdown for the CLI and the diagnostics panel.
    [[nodiscard]] std::string describe() const;
};

/// Bytes of KV cache one layer needs at the given context length.
///
/// The two layer types differ in every dimension that matters: sliding layers
/// keep a fixed ring sized by the window, while full-attention layers grow with
/// the requested context.
[[nodiscard]] u64 kvCacheBytesForLayer(const ArchInfo& arch, u64 layer, u64 contextLength,
                                       u64 maxPrefillTokens = kDefaultPrefillTokens);

/// Rows allocated per KV head for a layer.
///
/// Sliding layers get the window plus a chunk of headroom, so a 128-token
/// prefill chunk can be written without wrapping into rows the same chunk still
/// needs to read.
[[nodiscard]] u64 kvRowsForLayer(const ArchInfo& arch, u64 layer, u64 contextLength,
                                 u64 maxPrefillTokens = kDefaultPrefillTokens);

/// Extra rows beyond the sliding window, matching upstream's 1152 rows for a
/// 1024-token window.
///
/// This is a floor, not the whole story: the headroom must be at least the
/// prefill chunk width. A chunk writes all of its keys and values before any of
/// its queries read them, so with a ring of `window + h` rows, writing token
/// `i` of the chunk overwrites the row for position `p + i - window - h`. The
/// chunk's own first query still needs everything back to `p - window + 1`,
/// which survives only while `i <= h`. A chunk wider than the headroom
/// therefore silently destroys history its early tokens are about to read -
/// and, because the damage is confined to the oldest visible positions, it
/// produces plausible output rather than an obvious failure.
inline constexpr u64 kSlidingWindowHeadroomRows = 128;



/// Assigns layers to resident or streamed against a device memory budget.
///
/// Layers are made resident from the bottom up until the budget is spent; the
/// remainder stream with `slotsPerStreamedLayer` slots each. That degrades
/// continuously rather than cliff-edging: an 8 GB card streams everything, a
/// 16 GB card holds most layers, and the same code path serves both.
///
/// Fails when the budget cannot cover the resident core, the KV cache, scratch
/// and at least a minimal slot allocation for every layer.
[[nodiscard]] Result<ResidencyPlan> planResidency(const ArchInfo& arch,
                                                  const gturbo::ExpertLayout& experts,
                                                  u64 residentCoreBytes,
                                                  const ResidencyBudget& budget);

}  // namespace tf::runtime
