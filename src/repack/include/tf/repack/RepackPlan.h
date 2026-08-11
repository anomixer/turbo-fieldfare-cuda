#pragma once

#include <string>
#include <vector>

#include "tf/core/base/Error.h"
#include "tf/core/base/Types.h"
#include "tf/core/format/ArchInfo.h"
#include "tf/core/format/GTurbo.h"
#include "tf/core/format/Safetensors.h"

namespace tf::repack {

/// Which output file a copy lands in.
enum class DestKind {
    Resident,     ///< model_weights.bin
    ExpertLayer,  ///< packed_experts/layer_NN.bin
};

/// A single bounded byte copy from a source shard to an output file.
///
/// Operations carry no interpretation of the data: the repack copies quantized
/// values through unchanged, never dequantizing and requantizing. That is what
/// makes the install bit-exact against the source checkpoint.
struct CopyOp {
    u32 shardIndex = 0;  ///< index into RepackPlan::shards
    ByteRange source;    ///< absolute offset within that shard file
    DestKind destKind = DestKind::Resident;
    u32 destLayer = 0;   ///< meaningful only when destKind == ExpertLayer
    u64 destOffset = 0;

    friend bool operator==(const CopyOp&, const CopyOp&) = default;
};

/// Tuning knobs for plan construction.
struct PlanOptions {
    /// Largest single copy. Operations are pre-split to this size so the
    /// executor runs on a fixed scratch buffer regardless of tensor size - the
    /// embedding table alone is 369 MB. M1b will reuse the same splitting to
    /// bound HTTP range requests.
    u64 maxOpBytes = 8ull * 1024 * 1024;

    /// Expert blob stride alignment. 4 KiB is the minimum FILE_FLAG_NO_BUFFERING
    /// accepts; at a 3.19 MiB blob, 2 MiB alignment would waste 25% of the file.
    u64 expertAlignment = align::kSector;
};

/// A complete description of how to build a .gturbo install from a checkpoint,
/// computed before any byte is moved.
struct RepackPlan {
    ArchInfo arch;
    gturbo::ResidentIndex resident;
    gturbo::ExpertLayout experts;

    /// Shard file names, indexed by CopyOp::shardIndex.
    std::vector<std::string> shards;

    /// Sorted by (shardIndex, source.offset) so execution reads each shard
    /// front to back exactly once.
    std::vector<CopyOp> ops;

    [[nodiscard]] u64 residentBytes() const { return resident.totalBytes(); }
    [[nodiscard]] u64 expertBytes() const { return experts.totalBytes(); }
    [[nodiscard]] u64 totalOutputBytes() const { return residentBytes() + expertBytes(); }

    /// Sum of all copy lengths. Less than totalOutputBytes by exactly the
    /// alignment padding, which is written as zeros rather than copied.
    [[nodiscard]] u64 totalCopiedBytes() const;

    /// Confirms the operations tile every output byte except stride padding,
    /// so a planning bug cannot silently leave a projection uninitialized.
    [[nodiscard]] Status validate() const;
};

/// Canonical prefix on text-model tensors in the multimodal checkpoint.
inline constexpr std::string_view kLanguageModelPrefix = "language_model.model.";

/// Marks a tensor as a stacked routed-expert weight rather than a resident one.
inline constexpr std::string_view kExpertMarker = ".experts.switch_glu.";

/// Component roles inside an expert blob, in the order they are laid out.
[[nodiscard]] const std::vector<std::string>& expertComponentRoles();

/// Builds the plan. `headers` must be parallel to `index.shardFiles()`.
///
/// Vision-tower tensors are skipped: the port is text-only, which is the
/// difference between the 15.3 GB source and the 14.3 GB install.
[[nodiscard]] Result<RepackPlan> buildPlan(const ArchInfo& arch,
                                           const SafetensorsIndex& index,
                                           const std::vector<SafetensorsHeader>& headers,
                                           const PlanOptions& options = {});

}  // namespace tf::repack
