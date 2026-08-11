#pragma once

#include <filesystem>
#include <span>
#include <string>
#include <vector>

#include "tf/core/base/Error.h"
#include "tf/core/base/Types.h"

namespace tf::runtime {

enum class ModelFamily {
    /// Every weight participates in every token. The whole model must be
    /// resident, so it is bounded hard by VRAM.
    Dense,
    /// Routed experts: only a fraction of the weights fire per token, which is
    /// what makes streaming them from SSD viable.
    MixtureOfExperts,
};

[[nodiscard]] std::string_view toString(ModelFamily family) noexcept;

/// Whether this build can actually run a variant.
///
/// The engine is deliberately model-specific, exactly as upstream is. Its
/// forward pass assumes routed experts, so a dense variant is not "not yet
/// tuned" - it needs a different path.
enum class RuntimeSupport {
    /// Runs today.
    Supported,
    /// Needs a dense forward path: no router, no experts, no streaming. The
    /// kernels already exist; the layer sequencing does not.
    RequiresDensePath,
    /// Dense, and additionally needs per-layer input gating and cross-layer KV
    /// sharing, neither of which the runtime models.
    RequiresEdgeFeatures,
};

[[nodiscard]] std::string_view toString(RuntimeSupport support) noexcept;

struct ModelVariant {
    /// Short handle for the CLI, e.g. "26b-a4b".
    std::string id;
    std::string displayName;
    std::string repoId;
    /// Pinned commit, where one has been verified. Empty means track main.
    std::string revision;

    ModelFamily family = ModelFamily::Dense;
    RuntimeSupport support = RuntimeSupport::RequiresDensePath;

    /// Effective parameters that fire per token, in billions. This is what
    /// governs speed; `totalParamsB` governs quality.
    double activeParamsB = 0.0;
    double totalParamsB = 0.0;

    /// Measured from the Hugging Face repository.
    u64 downloadBytes = 0;
    /// Bytes on disk after repacking. Measured for the supported variant;
    /// approximated by the download size for the others.
    u64 installBytes = 0;

    /// Weights that must stay in VRAM for the whole session.
    u64 residentBytes = 0;
    /// Weights that may be streamed on demand. Zero for dense models, which is
    /// precisely why they are bounded by VRAM.
    u64 streamableBytes = 0;
    /// KV cache at a 4096-token context. Exact for the supported variant,
    /// estimated for the rest from the published architecture.
    u64 kvBytesAt4K = 0;

    /// Set when the numbers above are derived rather than measured.
    bool sizesAreEstimates = false;

    /// What the variant would need from the runtime, when unsupported.
    std::string supportNote;

    /// Smallest device budget that can run this at all: everything resident
    /// for a dense model, or the core plus a minimal slot allocation for a
    /// mixture of experts.
    [[nodiscard]] u64 minimumDeviceBytes() const;

    /// Device budget at which nothing needs to stream.
    [[nodiscard]] u64 comfortableDeviceBytes() const;
};

/// Every Gemma 4 variant published as a 4-bit MLX checkpoint, largest last.
[[nodiscard]] std::span<const ModelVariant> modelCatalog();

[[nodiscard]] Result<const ModelVariant*> findModel(std::string_view id);

/// What the machine has available.
struct MachineProfile {
    u64 deviceBytes = 0;   ///< free VRAM
    u64 systemRamBytes = 0;
    u64 diskFreeBytes = 0;
    bool diskIsSolidState = true;
    bool hasGpu = true;
};

enum class FitQuality {
    /// Everything resident; no streaming at all.
    Comfortable,
    /// Runs, with routed experts streaming from disk.
    Streamed,
    /// Would run but leaves very little headroom.
    Tight,
    /// Will not fit.
    DoesNotFit,
};

[[nodiscard]] std::string_view toString(FitQuality fit) noexcept;

struct ModelFit {
    const ModelVariant* variant = nullptr;
    FitQuality fit = FitQuality::DoesNotFit;
    /// One sentence explaining the verdict.
    std::string rationale;
    /// Things that will work but disappoint, such as an install on a spinning
    /// disk or too little RAM to cache the expert set.
    std::vector<std::string> warnings;

    [[nodiscard]] bool runnable() const {
        return fit != FitQuality::DoesNotFit &&
               variant->support == RuntimeSupport::Supported;
    }
};

struct Recommendation {
    /// Largest variant this build can actually run on this machine, or null.
    const ModelVariant* best = nullptr;
    /// Every variant assessed, largest first.
    std::vector<ModelFit> assessed;

    [[nodiscard]] std::string format() const;
};

/// Ranks the catalog against a machine and picks the largest runnable variant.
///
/// Larger is preferred because quality tracks total parameters, and the
/// streaming design exists specifically so that a large mixture of experts can
/// run where a dense model of the same footprint could not.
[[nodiscard]] Recommendation recommendModel(const MachineProfile& machine);

/// Reads the machine's specifications. `installPath` selects which volume is
/// measured for free space and storage type.
[[nodiscard]] Result<MachineProfile> detectMachine(const std::filesystem::path& installPath);

}  // namespace tf::runtime
