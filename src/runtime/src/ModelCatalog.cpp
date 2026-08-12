#include "tf/runtime/ModelCatalog.h"

#include <algorithm>
#include <format>

#include "tf/core/io/Volume.h"

namespace tf::runtime {
namespace {

constexpr u64 kMiB = 1024ull * 1024;
constexpr u64 kGiB = 1024ull * kMiB;

/// Headroom for activations, scratch and the staging ring, matching the
/// residency planner's default.
constexpr u64 kScratchBytes = 64 * kMiB;

/// Slots a streamed layer needs at minimum, which is top-k for the MoE variant.
constexpr u64 kMinimumSlotsPerLayer = 16;

std::string gib(u64 bytes) {
    return std::format("{:.2f} GiB", static_cast<double>(bytes) / static_cast<double>(kGiB));
}

/// The Gemma 4 family as published by mlx-community in 4-bit MLX form.
///
/// Download sizes are the summed safetensors bytes read from each repository.
/// The 26B-A4B figures are measured from an actual install; the others are
/// derived from the published architecture and are flagged as estimates - they
/// are used for ranking, not for allocation.
const std::vector<ModelVariant>& catalog() {
    static const std::vector<ModelVariant> kVariants = [] {
        std::vector<ModelVariant> variants;

        variants.push_back(ModelVariant{
                .id = "e2b",
                .displayName = "Gemma 4 E2B IT (4-bit)",
                .repoId = "mlx-community/gemma-4-e2b-it-4bit",
                .revision = {},
                .family = ModelFamily::Dense,
                .support = RuntimeSupport::RequiresEdgeFeatures,
                .activeParamsB = 2.0,
                .totalParamsB = 2.3,
                .downloadBytes = 3'550'670'554,
                .installBytes = 3'550'670'554,
                .residentBytes = 3'550'670'554,
                .streamableBytes = 0,
                .kvBytesAt4K = 32 * kMiB,
                .sizesAreEstimates = true,
                .supportNote = "needs per-layer input gating and cross-layer KV sharing "
                               "(20 of its 35 layers reuse another layer's cache)"});

        variants.push_back(ModelVariant{
                .id = "e4b",
                .displayName = "Gemma 4 E4B IT (4-bit)",
                .repoId = "mlx-community/gemma-4-e4b-it-4bit",
                .revision = {},
                .family = ModelFamily::Dense,
                .support = RuntimeSupport::RequiresEdgeFeatures,
                .activeParamsB = 4.0,
                .totalParamsB = 4.5,
                .downloadBytes = 5'146'800'534,
                .installBytes = 5'146'800'534,
                .residentBytes = 5'146'800'534,
                .streamableBytes = 0,
                .kvBytesAt4K = 89 * kMiB,
                .sizesAreEstimates = true,
                .supportNote = "needs per-layer input gating and cross-layer KV sharing "
                               "(18 of its 42 layers reuse another layer's cache)"});

        variants.push_back(ModelVariant{
                .id = "12b",
                .displayName = "Gemma 4 12B IT (4-bit)",
                .repoId = "mlx-community/gemma-4-12B-it-4bit",
                .revision = {},
                .family = ModelFamily::Dense,
                .support = RuntimeSupport::RequiresDensePath,
                .activeParamsB = 12.0,
                .totalParamsB = 12.0,
                .downloadBytes = 6'741'039'511,
                .installBytes = 6'741'039'511,
                .residentBytes = 6'741'039'511,
                .streamableBytes = 0,
                .kvBytesAt4K = 424 * kMiB,
                .sizesAreEstimates = true,
                .supportNote = "dense, so every weight fires per token; needs a forward path "
                               "without the router and expert branches"});

        // The one this engine was built for, and the only mixture of experts in
        // the family. Every figure here is measured from a real install.
        variants.push_back(ModelVariant{
                .id = "26b-a4b",
                .displayName = "Gemma 4 26B-A4B IT (4-bit)",
                .repoId = "mlx-community/gemma-4-26b-a4b-it-4bit",
                .revision = "0d77464eeb233a2da68ebf9d7dc4edaac7db956d",
                .family = ModelFamily::MixtureOfExperts,
                .support = RuntimeSupport::Supported,
                .activeParamsB = 3.88,
                .totalParamsB = 26.0,
                .downloadBytes = 15'340'981'404,
                .installBytes = 14'207'594'168,
                .residentBytes = 1'353'689'148,
                .streamableBytes = 12'853'905'020,
                .kvBytesAt4K = 305 * kMiB,
                .sizesAreEstimates = false,
                .supportNote = {}});

        variants.push_back(ModelVariant{
                .id = "31b",
                .displayName = "Gemma 4 31B IT (4-bit)",
                .repoId = "mlx-community/gemma-4-31b-it-4bit",
                .revision = {},
                .family = ModelFamily::Dense,
                .support = RuntimeSupport::RequiresDensePath,
                .activeParamsB = 31.0,
                .totalParamsB = 31.0,
                .downloadBytes = 18'412'016'676,
                .installBytes = 18'412'016'676,
                .residentBytes = 18'412'016'676,
                .streamableBytes = 0,
                .kvBytesAt4K = 1220 * kMiB,
                .sizesAreEstimates = true,
                .supportNote = "dense, so every weight fires per token; needs a forward path "
                               "without the router and expert branches"});

        return variants;
    }();
    return kVariants;
}

/// Slot allocation for a streamed mixture of experts, derived from the expert
/// blob size rather than assumed.
u64 minimumSlotBytes(const ModelVariant& variant) {
    if (variant.family != ModelFamily::MixtureOfExperts) {
        return 0;
    }
    // 30 layers x 16 slots x the 3,346,432-byte stride.
    return 30 * kMinimumSlotsPerLayer * 3'346'432ull;
}

}  // namespace

std::string_view toString(ModelFamily family) noexcept {
    return family == ModelFamily::MixtureOfExperts ? "mixture-of-experts" : "dense";
}

std::string_view toString(RuntimeSupport support) noexcept {
    switch (support) {
        case RuntimeSupport::Supported:            return "supported";
        case RuntimeSupport::RequiresDensePath:    return "needs dense path";
        case RuntimeSupport::RequiresEdgeFeatures: return "needs edge features";
    }
    return "?";
}

std::string_view toString(FitQuality fit) noexcept {
    switch (fit) {
        case FitQuality::Comfortable: return "comfortable";
        case FitQuality::Streamed:    return "streamed";
        case FitQuality::Tight:       return "tight";
        case FitQuality::DoesNotFit:  return "does not fit";
    }
    return "?";
}

u64 ModelVariant::minimumDeviceBytes() const {
    // A dense model has nothing to stream, so its floor is the whole thing.
    // A mixture of experts only needs its core plus enough slots to hold the
    // experts one token actually selects.
    return residentBytes + kvBytesAt4K + kScratchBytes + minimumSlotBytes(*this);
}

u64 ModelVariant::comfortableDeviceBytes() const {
    return residentBytes + streamableBytes + kvBytesAt4K + kScratchBytes;
}

std::span<const ModelVariant> modelCatalog() { return catalog(); }

Result<const ModelVariant*> findModel(std::string_view id) {
    for (const auto& variant : catalog()) {
        if (variant.id == id) {
            return &variant;
        }
    }

    std::string known;
    for (const auto& variant : catalog()) {
        if (!known.empty()) {
            known += ", ";
        }
        known += variant.id;
    }
    return makeError(ErrorCode::NotFound, "unknown model '{}'; known variants are {}", id,
                     known);
}

Recommendation recommendModel(const MachineProfile& machine) {
    Recommendation recommendation;

    // Largest first, so the first runnable entry is the best answer.
    std::vector<const ModelVariant*> ordered;
    for (const auto& variant : catalog()) {
        ordered.push_back(&variant);
    }
    std::ranges::sort(ordered, [](const ModelVariant* a, const ModelVariant* b) {
        return a->totalParamsB > b->totalParamsB;
    });

    for (const ModelVariant* variant : ordered) {
        ModelFit fit;
        fit.variant = variant;

        if (!machine.hasGpu) {
            fit.fit = FitQuality::DoesNotFit;
            fit.rationale = "no GPU available";
            recommendation.assessed.push_back(std::move(fit));
            continue;
        }

        const u64 minimum = variant->minimumDeviceBytes();
        const u64 comfortable = variant->comfortableDeviceBytes();

        if (machine.deviceBytes >= comfortable) {
            fit.fit = FitQuality::Comfortable;
            fit.rationale = std::format("{} of VRAM holds all {} of weights with no streaming",
                                        gib(machine.deviceBytes), gib(variant->installBytes));
        } else if (variant->family == ModelFamily::MixtureOfExperts &&
                   machine.deviceBytes >= minimum) {
            fit.fit = FitQuality::Streamed;
            fit.rationale =
                    std::format("{} of VRAM holds the {} core; routed experts stream from disk",
                                gib(machine.deviceBytes), gib(variant->residentBytes));
        } else if (machine.deviceBytes >= minimum) {
            // Dense and it fits, but only just: `minimum` already equals the
            // whole model for a dense variant.
            fit.fit = FitQuality::Tight;
            fit.rationale = std::format("{} of weights against {} of VRAM leaves little room",
                                        gib(variant->installBytes),
                                        gib(machine.deviceBytes));
        } else {
            fit.fit = FitQuality::DoesNotFit;
            fit.rationale =
                    variant->family == ModelFamily::MixtureOfExperts
                            ? std::format("needs {} of VRAM even with everything streamed, "
                                          "but only {} is free",
                                          gib(minimum), gib(machine.deviceBytes))
                            : std::format("dense weights need {} of VRAM but only {} is free",
                                          gib(minimum), gib(machine.deviceBytes));
        }

        if (fit.fit != FitQuality::DoesNotFit) {
            if (machine.diskFreeBytes < variant->installBytes) {
                fit.fit = FitQuality::DoesNotFit;
                fit.rationale = std::format("the install needs {} but only {} is free on disk",
                                            gib(variant->installBytes),
                                            gib(machine.diskFreeBytes));
            } else if (machine.diskFreeBytes < variant->installBytes + variant->downloadBytes) {
                fit.warnings.push_back(std::format(
                        "only {} free: enough for the install, but the download needs another "
                        "{} unless it is streamed directly",
                        gib(machine.diskFreeBytes), gib(variant->downloadBytes)));
            }
        }

        if (fit.fit == FitQuality::Streamed) {
            if (!machine.diskIsSolidState) {
                fit.warnings.emplace_back(
                        "the install volume is a spinning disk; expert streaming issues many "
                        "scattered reads per token and will be far slower than on an SSD");
            }
            // Once the expert set fits in the page cache, streaming reads are
            // served from RAM and the SSD drops out of the steady state.
            if (machine.systemRamBytes < variant->streamableBytes * 3 / 2) {
                fit.warnings.push_back(std::format(
                        "{} of RAM cannot cache the {} expert set, so reads stay bound by "
                        "disk speed rather than falling back to memory",
                        gib(machine.systemRamBytes), gib(variant->streamableBytes)));
            }
        }

        if (variant->support != RuntimeSupport::Supported && fit.fit != FitQuality::DoesNotFit) {
            fit.warnings.push_back(
                    std::format("this build cannot run it yet: {}", variant->supportNote));
        }

        if (recommendation.best == nullptr && fit.runnable()) {
            recommendation.best = variant;
        }
        recommendation.assessed.push_back(std::move(fit));
    }

    return recommendation;
}

std::string Recommendation::format() const {
    usize nameWidth = 0;
    for (const auto& fit : assessed) {
        nameWidth = std::max(nameWidth, fit.variant->displayName.size());
    }

    std::string out;
    for (const auto& fit : assessed) {
        const char* marker = (fit.variant == best) ? "->" : "  ";
        out += std::format("{} {:<{}}  {:<12}  {}\n", marker, fit.variant->displayName,
                           nameWidth, toString(fit.fit), fit.rationale);
        for (const auto& warning : fit.warnings) {
            out += std::format("   {:<{}}  {:<12}  ! {}\n", "", nameWidth, "", warning);
        }
    }

    if (best != nullptr) {
        out += std::format("\nRecommended: {} ({})\n", best->displayName, best->repoId);
    } else {
        out += "\nNo variant in the catalog can run on this machine with this build.\n";
    }
    return out;
}

Result<MachineProfile> detectMachine(const std::filesystem::path& installPath) {
    MachineProfile machine;

    TF_TRY(const io::VolumeInfo volume, io::queryVolume(installPath));
    machine.diskFreeBytes = volume.freeBytes;
    machine.diskIsSolidState = volume.storage != io::StorageKind::Rotational;

    TF_TRY(const u64 ram, io::totalPhysicalMemory());
    machine.systemRamBytes = ram;

    // The caller fills deviceBytes and hasGpu from the GPU backend; this
    // function does not depend on tf::gpu so it stays usable without one.
    machine.hasGpu = false;
    return machine;
}

}  // namespace tf::runtime
