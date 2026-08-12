#pragma once

#include <array>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "tf/core/base/Error.h"
#include "tf/core/base/Types.h"
#include "tf/core/format/GTurbo.h"
#include "tf/gpu/Backend.h"
#include "tf/gpu/Kernels.h"
#include "tf/runtime/ExpertStreamer.h"
#include "tf/runtime/Residency.h"

namespace tf::runtime {

/// One quantized linear, bound to device memory.
struct QuantWeightRef {
    gpu::QuantizedWeights weights;

    [[nodiscard]] bool valid() const { return weights.packed.valid(); }
};

/// Everything one transformer layer needs, resolved to device addresses once at
/// load rather than looked up per token.
struct LayerWeights {
    // Attention. vProj is deliberately absent on full-attention layers, where
    // attention_k_eq_v means V branches off the raw K projection instead.
    QuantWeightRef qProj;
    QuantWeightRef kProj;
    QuantWeightRef vProj;
    QuantWeightRef oProj;
    gpu::DeviceView qNorm;  ///< bf16 [headDim]
    gpu::DeviceView kNorm;  ///< bf16 [headDim]

    // Dense (shared) expert.
    QuantWeightRef gateProj;
    QuantWeightRef upProj;
    QuantWeightRef downProj;

    // Router.
    QuantWeightRef routerProj;  ///< 8-bit
    /// router.scale premultiplied by hiddenSize^-0.5, folded once at load
    /// rather than recomputed per token.
    gpu::DeviceView routerScaleFolded;
    gpu::DeviceView perExpertScale;  ///< bf16 [numExperts]

    // The seven per-layer norms.
    gpu::DeviceView inputNorm;
    gpu::DeviceView postAttentionNorm;
    gpu::DeviceView preFeedforwardNorm;
    gpu::DeviceView preFeedforwardNorm2;
    gpu::DeviceView postFeedforwardNorm;
    gpu::DeviceView postFeedforwardNorm1;
    gpu::DeviceView postFeedforwardNorm2;

    /// Multiplies the whole layer output, residual included. Read to the host
    /// at load since it is a single scalar.
    float layerScalar = 1.0f;

    /// Set when this layer's experts live in VRAM permanently.
    bool expertsResident = false;
    /// Valid only when expertsResident: the whole layer's expert file.
    gpu::Buffer* residentExperts = nullptr;
};

/// A loaded .gturbo install: resident weights in VRAM, experts either resident
/// or streaming, and every tensor resolved to a device view.
class Model {
public:
    Model() = default;
    ~Model();

    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;
    Model(Model&&) noexcept;
    Model& operator=(Model&&) noexcept;

    struct LoadOptions {
        ResidencyBudget budget;
        EvictionPolicy evictionPolicy = EvictionPolicy::Lfu;
        bool unbufferedReads = false;
        u32 readThreads = 4;
    };

    /// Progress callback for the load, which moves gigabytes and is worth
    /// reporting. Return false to cancel.
    using LoadProgress = std::function<bool(std::string_view stage, u64 done, u64 total)>;

    [[nodiscard]] static Result<Model> load(gpu::IGpuBackend& backend,
                                            const std::filesystem::path& installDir,
                                            const LoadOptions& options = {},
                                            const LoadProgress& progress = {});

    [[nodiscard]] const ArchInfo& arch() const noexcept { return manifest_.arch; }
    [[nodiscard]] const gturbo::Manifest& manifest() const noexcept { return manifest_; }
    [[nodiscard]] const ResidencyPlan& residency() const noexcept { return residency_; }

    [[nodiscard]] const LayerWeights& layer(u64 index) const {
        return layers_[static_cast<usize>(index)];
    }

    /// The tied embedding table, used both for the input lookup and,
    /// transposed, as the output head.
    [[nodiscard]] const QuantWeightRef& embedding() const noexcept { return embedding_; }
    [[nodiscard]] gpu::DeviceView finalNorm() const noexcept { return finalNorm_; }

    [[nodiscard]] ExpertStreamer& streamer() noexcept { return streamer_; }
    [[nodiscard]] const ExpertStreamer& streamer() const noexcept { return streamer_; }

    /// Binds one expert's gate, up and down projections, wherever it lives:
    /// a streamer slot for a streamed layer, or the permanent allocation for a
    /// resident one.
    [[nodiscard]] Result<std::array<QuantWeightRef, 3>> expertWeights(u64 layer, u32 expert,
                                                                      u32 slot) const;

    [[nodiscard]] u64 deviceBytesUsed() const noexcept { return deviceBytes_; }

private:
    gpu::IGpuBackend* backend_ = nullptr;
    gturbo::Manifest manifest_;
    ResidencyPlan residency_;

    /// The whole resident weight file, uploaded once.
    gpu::BufferPtr resident_;
    /// Per-layer expert files for layers the planner made resident.
    std::vector<gpu::BufferPtr> residentExperts_;
    /// Folded router scales, one small buffer per layer.
    std::vector<gpu::BufferPtr> routerScales_;

    std::vector<LayerWeights> layers_;
    QuantWeightRef embedding_;
    gpu::DeviceView finalNorm_;

    ExpertStreamer streamer_;
    u64 deviceBytes_ = 0;
};

}  // namespace tf::runtime
