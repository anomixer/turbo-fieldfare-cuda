#include "tf/runtime/Model.h"

#include <algorithm>
#include <format>

#include "tf/core/io/File.h"
#include "tf/core/math/Float.h"

namespace tf::runtime {
namespace {

/// Resolves a resident tensor by name and turns it into a device view into the
/// single uploaded resident buffer.
class ResidentBinder {
public:
    ResidentBinder(const gturbo::ResidentIndex& index, gpu::Buffer& buffer,
                   const io::MappedFile& mapping)
        : index_(index), buffer_(buffer), mapping_(mapping) {}

    [[nodiscard]] Result<gpu::DeviceView> view(std::string_view name) const {
        TF_TRY(const gturbo::ResidentTensor* tensor, index_.require(name));
        return gpu::DeviceView{.buffer = &buffer_, .offset = tensor->range.offset};
    }

    /// Host bytes of a resident tensor, for the handful of values the loader
    /// needs on the CPU.
    [[nodiscard]] Result<ByteSpan> hostBytes(std::string_view name) const {
        TF_TRY(const gturbo::ResidentTensor* tensor, index_.require(name));
        return mapping_.range(tensor->range);
    }

    /// Binds a quantized linear from its three component tensors.
    [[nodiscard]] Result<QuantWeightRef> quantized(std::string_view base,
                                                   const QuantizedLinearLayout& layout) const {
        TF_CHECK(layout.validate());

        TF_TRY(const gpu::DeviceView packed, view(std::format("{}.weight", base)));
        TF_TRY(const gpu::DeviceView scales, view(std::format("{}.scales", base)));
        TF_TRY(const gpu::DeviceView biases, view(std::format("{}.biases", base)));

        // Cross-check the stored shapes against the architecture. A mismatch
        // here means the manifest and the config disagree, which would show up
        // downstream as arithmetic on the wrong stride.
        TF_TRY(const gturbo::ResidentTensor* weight, index_.require(std::format("{}.weight",
                                                                                base)));
        if (weight->range.length != layout.weightBytes()) {
            return makeError(ErrorCode::MalformedData,
                             "{}.weight is {} bytes but the architecture implies {}", base,
                             weight->range.length, layout.weightBytes());
        }

        return QuantWeightRef{.weights = gpu::QuantizedWeights{.packed = packed,
                                                               .scales = scales,
                                                               .biases = biases,
                                                               .layout = layout}};
    }

private:
    const gturbo::ResidentIndex& index_;
    gpu::Buffer& buffer_;
    const io::MappedFile& mapping_;
};

bool report(const Model::LoadProgress& progress, std::string_view stage, u64 done, u64 total) {
    return !progress || progress(stage, done, total);
}

/// Uploads a large host range in bounded chunks so progress can be reported and
/// a cancel can take effect part way through.
Status uploadChunked(gpu::IGpuBackend& backend, gpu::Stream& stream, gpu::Buffer& destination,
                     ByteSpan source, std::string_view stage,
                     const Model::LoadProgress& progress) {
    constexpr u64 kChunkBytes = 64ull * 1024 * 1024;

    u64 offset = 0;
    while (offset < source.size()) {
        const u64 chunk = std::min(kChunkBytes, source.size() - offset);
        TF_CHECK(backend.enqueueUpload(stream, destination, offset,
                                       source.subspan(static_cast<usize>(offset),
                                                      static_cast<usize>(chunk))));
        // The mapping is pageable, so each upload is synchronous anyway; the
        // explicit sync just bounds how much is in flight.
        TF_CHECK(stream.synchronize());
        offset += chunk;

        if (!report(progress, stage, offset, source.size())) {
            return makeError(ErrorCode::Cancelled, "cancelled while loading");
        }
    }
    return {};
}

}  // namespace

Model::~Model() = default;
Model::Model(Model&&) noexcept = default;
Model& Model::operator=(Model&&) noexcept = default;

Result<Model> Model::load(gpu::IGpuBackend& backend, const std::filesystem::path& installDir,
                          const LoadOptions& options, const LoadProgress& progress) {
    Model model;
    model.backend_ = &backend;

    TF_TRY(model.manifest_, gturbo::readManifest(installDir));
    const ArchInfo& arch = model.manifest_.arch;

    // ---- Residency plan -------------------------------------------------
    ResidencyBudget budget = options.budget;
    if (budget.deviceBytes == 0) {
        // Default to what the device actually has free, less a margin for the
        // desktop compositor and anything else that starts later.
        TF_TRY(const gpu::MemoryInfo memory, backend.memoryInfo());
        constexpr u64 kSafetyMargin = 512ull * 1024 * 1024;
        budget.deviceBytes = memory.freeBytes > kSafetyMargin
                                     ? memory.freeBytes - kSafetyMargin
                                     : memory.freeBytes;
    }

    TF_TRY(model.residency_,
           planResidency(arch, model.manifest_.experts,
                         model.manifest_.resident.totalBytes(), budget));

    // ---- Resident weights ------------------------------------------------
    const std::filesystem::path residentPath = installDir / gturbo::kResidentFile;
    TF_TRY(const io::MappedFile mapping, io::MappedFile::open(residentPath));

    if (mapping.size() != model.manifest_.resident.totalBytes()) {
        return makeError(ErrorCode::VerificationFailed,
                         "{} is {} bytes but the manifest records {}", residentPath.string(),
                         mapping.size(), model.manifest_.resident.totalBytes());
    }

    TF_TRY(model.resident_,
           backend.allocate(gpu::MemoryKind::Device, mapping.size(), "resident-weights"));
    model.deviceBytes_ += mapping.size();

    TF_TRY(gpu::StreamPtr loadStream, backend.createStream("model-load"));
    TF_CHECK(uploadChunked(backend, *loadStream, *model.resident_, mapping.bytes(),
                           "resident weights", progress));

    const ResidentBinder binder{model.manifest_.resident, *model.resident_, mapping};

    // ---- Global tensors --------------------------------------------------
    TF_TRY(model.embedding_, binder.quantized("embed_tokens", arch.embeddingLayout()));
    TF_TRY(model.finalNorm_, binder.view("norm.weight"));

    // ---- Per-layer binding ----------------------------------------------
    model.layers_.resize(static_cast<usize>(arch.numLayers));
    model.routerScales_.resize(static_cast<usize>(arch.numLayers));

    for (u64 index = 0; index < arch.numLayers; ++index) {
        LayerWeights& layer = model.layers_[static_cast<usize>(index)];
        const std::string prefix = std::format("layers.{}", index);

        TF_TRY(layer.qProj,
               binder.quantized(std::format("{}.self_attn.q_proj", prefix),
                                arch.qProjLayout(index)));
        TF_TRY(layer.kProj,
               binder.quantized(std::format("{}.self_attn.k_proj", prefix),
                                arch.kvProjLayout(index)));
        // Full-attention layers carry no v_proj: V comes from the raw K
        // projection instead, so leaving this unbound is correct.
        if (arch.hasSeparateVProj(index)) {
            TF_TRY(layer.vProj,
                   binder.quantized(std::format("{}.self_attn.v_proj", prefix),
                                    arch.kvProjLayout(index)));
        }
        TF_TRY(layer.oProj,
               binder.quantized(std::format("{}.self_attn.o_proj", prefix),
                                arch.oProjLayout(index)));

        TF_TRY(layer.qNorm, binder.view(std::format("{}.self_attn.q_norm.weight", prefix)));
        TF_TRY(layer.kNorm, binder.view(std::format("{}.self_attn.k_norm.weight", prefix)));

        TF_TRY(layer.gateProj,
               binder.quantized(std::format("{}.mlp.gate_proj", prefix),
                                arch.sharedGateUpLayout()));
        TF_TRY(layer.upProj, binder.quantized(std::format("{}.mlp.up_proj", prefix),
                                              arch.sharedGateUpLayout()));
        TF_TRY(layer.downProj, binder.quantized(std::format("{}.mlp.down_proj", prefix),
                                                arch.sharedDownLayout()));

        TF_TRY(layer.routerProj,
               binder.quantized(std::format("{}.router.proj", prefix), arch.routerLayout()));
        TF_TRY(layer.perExpertScale,
               binder.view(std::format("{}.router.per_expert_scale", prefix)));

        TF_TRY(layer.inputNorm, binder.view(std::format("{}.input_layernorm.weight", prefix)));
        TF_TRY(layer.postAttentionNorm,
               binder.view(std::format("{}.post_attention_layernorm.weight", prefix)));
        TF_TRY(layer.preFeedforwardNorm,
               binder.view(std::format("{}.pre_feedforward_layernorm.weight", prefix)));
        TF_TRY(layer.preFeedforwardNorm2,
               binder.view(std::format("{}.pre_feedforward_layernorm_2.weight", prefix)));
        TF_TRY(layer.postFeedforwardNorm,
               binder.view(std::format("{}.post_feedforward_layernorm.weight", prefix)));
        TF_TRY(layer.postFeedforwardNorm1,
               binder.view(std::format("{}.post_feedforward_layernorm_1.weight", prefix)));
        TF_TRY(layer.postFeedforwardNorm2,
               binder.view(std::format("{}.post_feedforward_layernorm_2.weight", prefix)));

        // layer_scalar is a single bf16 that multiplies the whole layer output.
        // Reading it to the host removes an indirection from every token.
        TF_TRY(const ByteSpan scalarBytes,
               binder.hostBytes(std::format("{}.layer_scalar", prefix)));
        if (scalarBytes.size() < sizeof(bf16)) {
            return makeError(ErrorCode::MalformedData, "{}.layer_scalar is empty", prefix);
        }
        layer.layerScalar = toFloat(asBf16(scalarBytes)[0]);

        // The router normalizes with router.scale * hiddenSize^-0.5. Folding
        // the constant here saves a multiply per element per token.
        TF_TRY(const ByteSpan routerScaleBytes,
               binder.hostBytes(std::format("{}.router.scale", prefix)));
        const auto routerScale = asBf16(routerScaleBytes);

        std::vector<bf16> folded(routerScale.size());
        const auto rootSize =
                static_cast<float>(1.0 / std::sqrt(static_cast<double>(arch.hiddenSize)));
        for (usize i = 0; i < routerScale.size(); ++i) {
            folded[i] = toBf16(toFloat(routerScale[i]) * rootSize);
        }

        TF_TRY(gpu::BufferPtr foldedBuffer,
               backend.allocate(gpu::MemoryKind::Device, folded.size() * sizeof(bf16),
                                std::format("router-scale-{:02}", index)));
        TF_CHECK(backend.enqueueUpload(
                *loadStream, *foldedBuffer, 0,
                ByteSpan{reinterpret_cast<const u8*>(folded.data()),
                         folded.size() * sizeof(bf16)}));
        model.deviceBytes_ += folded.size() * sizeof(bf16);

        layer.routerScaleFolded = gpu::DeviceView{.buffer = foldedBuffer.get(), .offset = 0};
        model.routerScales_[static_cast<usize>(index)] = std::move(foldedBuffer);

        layer.expertsResident =
                model.residency_.layers[static_cast<usize>(index)].expertsResident;
    }
    TF_CHECK(loadStream->synchronize());

    // ---- Experts ---------------------------------------------------------
    const gturbo::ExpertLayout& expertLayout = model.manifest_.experts;
    const std::filesystem::path expertsDir = installDir / gturbo::kExpertsDir;

    std::vector<u64> streamedLayers;
    model.residentExperts_.resize(static_cast<usize>(arch.numLayers));

    u64 residentExpertBytes = 0;
    for (u64 index = 0; index < arch.numLayers; ++index) {
        if (model.residency_.layers[static_cast<usize>(index)].expertsResident) {
            residentExpertBytes += expertLayout.layerFileBytes();
        } else {
            streamedLayers.push_back(index);
        }
    }

    u64 uploadedExpertBytes = 0;
    for (u64 index = 0; index < arch.numLayers; ++index) {
        if (!model.residency_.layers[static_cast<usize>(index)].expertsResident) {
            continue;
        }

        const std::filesystem::path layerPath =
                expertsDir / expertLayout.layerFiles[static_cast<usize>(index)];
        TF_TRY(const io::MappedFile layerMapping, io::MappedFile::open(layerPath));

        TF_TRY(gpu::BufferPtr buffer,
               backend.allocate(gpu::MemoryKind::Device, layerMapping.size(),
                                std::format("resident-experts-{:02}", index)));

        TF_CHECK(backend.enqueueUpload(*loadStream, *buffer, 0, layerMapping.bytes()));
        TF_CHECK(loadStream->synchronize());

        uploadedExpertBytes += layerMapping.size();
        if (!report(progress, "resident experts", uploadedExpertBytes, residentExpertBytes)) {
            return makeError(ErrorCode::Cancelled, "cancelled while loading experts");
        }

        model.deviceBytes_ += layerMapping.size();
        model.layers_[static_cast<usize>(index)].residentExperts = buffer.get();
        model.residentExperts_[static_cast<usize>(index)] = std::move(buffer);
    }

    if (!streamedLayers.empty()) {
        TF_TRY(model.streamer_,
               ExpertStreamer::create(
                       backend, installDir, expertLayout, streamedLayers,
                       StreamerOptions{
                               .slotsPerLayer = budget.slotsPerStreamedLayer,
                               .policy = options.evictionPolicy,
                               .unbufferedReads = options.unbufferedReads,
                               .readThreads = options.readThreads}));
        model.deviceBytes_ +=
                streamedLayers.size() * budget.slotsPerStreamedLayer * expertLayout.stride;
    }

    return model;
}

Result<std::array<QuantWeightRef, 3>> Model::expertWeights(u64 layer, u32 expert,
                                                           u32 slot) const {
    const ArchInfo& arch = manifest_.arch;
    const gturbo::ExpertLayout& expertLayout = manifest_.experts;
    const LayerWeights& weights = layers_[static_cast<usize>(layer)];

    // Where the blob lives depends on whether the planner made this layer
    // resident; the component offsets inside it are identical either way.
    gpu::Buffer* buffer = nullptr;
    u64 base = 0;

    if (weights.expertsResident) {
        buffer = weights.residentExperts;
        if (buffer == nullptr) {
            return makeError(ErrorCode::Unknown, "layer {} is resident but has no experts",
                             layer);
        }
        TF_TRY(const ByteRange range, expertLayout.expertRange(layer, expert));
        base = range.offset;
    } else {
        const gpu::DeviceView slotView = streamer_.slotView(layer, slot);
        if (!slotView.valid()) {
            return makeError(ErrorCode::InvalidArgument,
                             "layer {} slot {} is not backed by the streamer", layer, slot);
        }
        buffer = slotView.buffer;
        base = slotView.offset;
    }

    const auto component = [&](std::string_view role) -> Result<u64> {
        for (const auto& entry : expertLayout.components) {
            if (entry.role == role) {
                return base + entry.offset;
            }
        }
        return makeError(ErrorCode::NotFound, "expert component '{}' is not in the layout",
                         role);
    };

    const auto bind = [&](std::string_view prefix,
                          const QuantizedLinearLayout& layout) -> Result<QuantWeightRef> {
        TF_TRY(const u64 packed, component(std::format("{}.weight", prefix)));
        TF_TRY(const u64 scales, component(std::format("{}.scales", prefix)));
        TF_TRY(const u64 biases, component(std::format("{}.biases", prefix)));

        return QuantWeightRef{
                .weights = gpu::QuantizedWeights{
                        .packed = gpu::DeviceView{.buffer = buffer, .offset = packed},
                        .scales = gpu::DeviceView{.buffer = buffer, .offset = scales},
                        .biases = gpu::DeviceView{.buffer = buffer, .offset = biases},
                        .layout = layout}};
    };

    std::array<QuantWeightRef, 3> result;
    TF_TRY(result[0], bind("gate", arch.expertGateUpLayout()));
    TF_TRY(result[1], bind("up", arch.expertGateUpLayout()));
    TF_TRY(result[2], bind("down", arch.expertDownLayout()));
    return result;
}

}  // namespace tf::runtime
