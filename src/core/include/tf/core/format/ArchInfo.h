#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "tf/core/base/Error.h"
#include "tf/core/base/Types.h"
#include "tf/core/format/Quantization.h"

namespace tf {

enum class AttentionKind {
    /// Sliding-window attention over the last `slidingWindow` tokens.
    Sliding,
    /// Full causal attention over the whole prompt.
    Full,
};

/// Architecture of the text model, read from the checkpoint's config.json.
///
/// Gemma 4 26B-A4B is not uniform across layers: sliding and full-attention
/// layers differ in head dimension, KV head count, RoPE parameters, and whether
/// a separate V projection exists at all. Every dimension the runtime needs is
/// therefore derived per layer rather than assumed global.
struct ArchInfo {
    // ---- Core dimensions -------------------------------------------------
    u64 hiddenSize = 0;
    u64 numLayers = 0;
    u64 vocabSize = 0;
    u64 maxPositionEmbeddings = 0;

    // ---- Attention -------------------------------------------------------
    u64 numHeads = 0;
    /// KV heads on sliding-attention layers.
    u64 numKVHeads = 0;
    /// KV heads on full-attention layers, typically far fewer.
    u64 numGlobalKVHeads = 0;
    /// Head dimension on sliding-attention layers.
    u64 headDim = 0;
    /// Head dimension on full-attention layers.
    u64 globalHeadDim = 0;
    u64 slidingWindow = 0;
    /// When true, full-attention layers reuse K as V and carry no v_proj.
    bool attentionKEqV = false;

    // ---- Feed-forward ----------------------------------------------------
    /// Shared (dense) expert width.
    u64 intermediateSize = 0;
    /// Per routed-expert width.
    u64 moeIntermediateSize = 0;
    u64 numExperts = 0;
    u64 topKExperts = 0;
    std::string hiddenActivation;

    // ---- Normalization and output ---------------------------------------
    double rmsNormEps = 0.0;
    double finalLogitSoftcap = 0.0;
    bool tieWordEmbeddings = false;

    // ---- RoPE ------------------------------------------------------------
    double slidingRopeTheta = 0.0;
    double fullRopeTheta = 0.0;
    /// Fraction of each head dimension that is rotated on full-attention
    /// layers; the remainder passes through unrotated.
    double partialRotaryFactor = 0.0;

    // ---- Tokens ----------------------------------------------------------
    u32 bosTokenId = 0;
    u32 padTokenId = 0;
    std::vector<u32> eosTokenIds;

    // ---- Quantization ----------------------------------------------------
    QuantSpec weightQuant = kWeightQuant;
    QuantSpec routerQuant = kRouterQuant;

    /// Attention kind per layer, length `numLayers`.
    std::vector<AttentionKind> layerTypes;

    // ---- Derived per-layer geometry -------------------------------------

    [[nodiscard]] bool isFullAttention(u64 layer) const {
        return layer < layerTypes.size() && layerTypes[layer] == AttentionKind::Full;
    }

    [[nodiscard]] u64 headDimFor(u64 layer) const {
        return isFullAttention(layer) ? globalHeadDim : headDim;
    }

    [[nodiscard]] u64 kvHeadsFor(u64 layer) const {
        return isFullAttention(layer) ? numGlobalKVHeads : numKVHeads;
    }

    [[nodiscard]] double ropeThetaFor(u64 layer) const {
        return isFullAttention(layer) ? fullRopeTheta : slidingRopeTheta;
    }

    /// Output width of q_proj: all heads concatenated.
    [[nodiscard]] u64 qProjOutFeatures(u64 layer) const {
        return numHeads * headDimFor(layer);
    }

    /// Output width of k_proj, and of v_proj where one exists.
    [[nodiscard]] u64 kvProjOutFeatures(u64 layer) const {
        return kvHeadsFor(layer) * headDimFor(layer);
    }

    /// True when the layer stores its own v_proj. False on full-attention
    /// layers when attentionKEqV is set, where V aliases K.
    [[nodiscard]] bool hasSeparateVProj(u64 layer) const {
        return !(attentionKEqV && isFullAttention(layer));
    }

    [[nodiscard]] u64 slidingLayerCount() const;
    [[nodiscard]] u64 fullAttentionLayerCount() const;

    // ---- Derived quantized layouts --------------------------------------

    [[nodiscard]] QuantizedLinearLayout qProjLayout(u64 layer) const {
        return {.outFeatures = qProjOutFeatures(layer),
                .inFeatures = hiddenSize,
                .spec = weightQuant};
    }

    [[nodiscard]] QuantizedLinearLayout kvProjLayout(u64 layer) const {
        return {.outFeatures = kvProjOutFeatures(layer),
                .inFeatures = hiddenSize,
                .spec = weightQuant};
    }

    [[nodiscard]] QuantizedLinearLayout oProjLayout(u64 layer) const {
        return {.outFeatures = hiddenSize,
                .inFeatures = qProjOutFeatures(layer),
                .spec = weightQuant};
    }

    [[nodiscard]] QuantizedLinearLayout sharedGateUpLayout() const {
        return {.outFeatures = intermediateSize,
                .inFeatures = hiddenSize,
                .spec = weightQuant};
    }

    [[nodiscard]] QuantizedLinearLayout sharedDownLayout() const {
        return {.outFeatures = hiddenSize,
                .inFeatures = intermediateSize,
                .spec = weightQuant};
    }

    /// Gate and up projections of a single routed expert.
    [[nodiscard]] QuantizedLinearLayout expertGateUpLayout() const {
        return {.outFeatures = moeIntermediateSize,
                .inFeatures = hiddenSize,
                .spec = weightQuant};
    }

    [[nodiscard]] QuantizedLinearLayout expertDownLayout() const {
        return {.outFeatures = hiddenSize,
                .inFeatures = moeIntermediateSize,
                .spec = weightQuant};
    }

    [[nodiscard]] QuantizedLinearLayout routerLayout() const {
        return {.outFeatures = numExperts, .inFeatures = hiddenSize, .spec = routerQuant};
    }

    [[nodiscard]] QuantizedLinearLayout embeddingLayout() const {
        return {.outFeatures = vocabSize, .inFeatures = hiddenSize, .spec = weightQuant};
    }

    /// Unpadded bytes of one routed expert: gate + up + down.
    [[nodiscard]] u64 expertBlobBytes() const {
        return 2 * expertGateUpLayout().totalBytes() + expertDownLayout().totalBytes();
    }

    /// Total routed-expert bytes across the whole model, before stride padding.
    /// This is what the residency planner budgets against.
    [[nodiscard]] u64 totalExpertBytes() const {
        return expertBlobBytes() * numExperts * numLayers;
    }

    /// Rejects a config that is internally inconsistent or that the runtime
    /// cannot support, so failures surface at load rather than as bad output.
    [[nodiscard]] Status validate() const;

    /// Parses the `text_config` and `quantization` sections of a checkpoint
    /// config.json. Vision fields are ignored: the port is text-only.
    [[nodiscard]] static Result<ArchInfo> parseConfigJson(std::string_view json);

    [[nodiscard]] static Result<ArchInfo> readFromCheckpoint(
            const std::filesystem::path& checkpointDir);
};

[[nodiscard]] std::string_view toString(AttentionKind kind) noexcept;

}  // namespace tf
