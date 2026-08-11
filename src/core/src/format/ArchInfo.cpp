#include "tf/core/format/ArchInfo.h"

#include <algorithm>

#include "tf/core/io/File.h"
#include "tf/core/json/Json.h"

namespace tf {
namespace {

Result<u64> requireUInt(const json::Value& root, std::string_view path) {
    TF_TRY(const json::Value* node, root.path(path));
    auto value = node->asUInt();
    if (!value) {
        return std::unexpected(value.error().wrap(path));
    }
    return *value;
}

Result<double> requireDouble(const json::Value& root, std::string_view path) {
    TF_TRY(const json::Value* node, root.path(path));
    auto value = node->asDouble();
    if (!value) {
        return std::unexpected(value.error().wrap(path));
    }
    return *value;
}

/// Optional fields fall back rather than failing: config.json varies slightly
/// between exporters, and a missing `partial_rotary_factor` is not corruption.
u64 optionalUInt(const json::Value& root, std::string_view path, u64 fallback) {
    if (const auto node = root.path(path); node.has_value()) {
        if (const auto value = (*node)->asUInt(); value.has_value()) {
            return *value;
        }
    }
    return fallback;
}

double optionalDouble(const json::Value& root, std::string_view path, double fallback) {
    if (const auto node = root.path(path); node.has_value()) {
        if (const auto value = (*node)->asDouble(); value.has_value()) {
            return *value;
        }
    }
    return fallback;
}

bool optionalBool(const json::Value& root, std::string_view path, bool fallback) {
    if (const auto node = root.path(path); node.has_value()) {
        if (const auto value = (*node)->asBool(); value.has_value()) {
            return *value;
        }
    }
    return fallback;
}

std::string optionalString(const json::Value& root, std::string_view path,
                           std::string_view fallback) {
    if (const auto node = root.path(path); node.has_value()) {
        if (const auto value = (*node)->asString(); value.has_value()) {
            return std::string{*value};
        }
    }
    return std::string{fallback};
}

/// `eos_token_id` is a scalar in some exports and an array in others.
std::vector<u32> parseTokenIdList(const json::Value* node) {
    std::vector<u32> ids;
    if (node == nullptr) {
        return ids;
    }
    if (const auto single = node->asUInt(); single.has_value()) {
        ids.push_back(static_cast<u32>(*single));
        return ids;
    }
    if (const auto array = node->asArray(); array.has_value()) {
        for (const auto& element : **array) {
            if (const auto value = element.asUInt(); value.has_value()) {
                ids.push_back(static_cast<u32>(*value));
            }
        }
    }
    return ids;
}

/// Reads the MLX quantization block. The base spec applies to everything;
/// per-tensor overrides appear as nested objects keyed by tensor path, and the
/// router is the only one that differs in the pinned checkpoint.
Result<std::pair<QuantSpec, QuantSpec>> parseQuantization(const json::Value& root) {
    const json::Value* block = root.find("quantization");
    if (block == nullptr) {
        block = root.find("quantization_config");
    }
    if (block == nullptr) {
        return makeError(ErrorCode::Unsupported,
                         "config.json has no quantization block; this runtime only loads "
                         "MLX affine quantized checkpoints");
    }

    const auto mode = optionalString(*block, "mode", "affine");
    if (mode != "affine") {
        return makeError(ErrorCode::Unsupported,
                         "quantization mode '{}' is not supported (expected 'affine')", mode);
    }

    QuantSpec base{.bits = static_cast<u32>(optionalUInt(*block, "bits", 4)),
                   .groupSize = static_cast<u32>(optionalUInt(*block, "group_size", 64))};

    // Scan for a router override. Every layer carries an identical entry, so the
    // first one found describes them all.
    QuantSpec router = base;
    TF_TRY(const json::Object* entries, block->asObject());
    for (const auto& [key, value] : *entries) {
        if (!key.ends_with("router.proj") || !value.isObject()) {
            continue;
        }
        router = QuantSpec{
                .bits = static_cast<u32>(optionalUInt(value, "bits", base.bits)),
                .groupSize = static_cast<u32>(optionalUInt(value, "group_size", base.groupSize))};
        break;
    }

    return std::pair{base, router};
}

}  // namespace

std::string_view toString(AttentionKind kind) noexcept {
    return kind == AttentionKind::Full ? "full_attention" : "sliding_attention";
}

u64 ArchInfo::slidingLayerCount() const {
    return static_cast<u64>(std::ranges::count(layerTypes, AttentionKind::Sliding));
}

u64 ArchInfo::fullAttentionLayerCount() const {
    return static_cast<u64>(std::ranges::count(layerTypes, AttentionKind::Full));
}

Status ArchInfo::validate() const {
    if (numLayers == 0 || hiddenSize == 0 || vocabSize == 0) {
        return makeError(ErrorCode::MalformedData, "architecture has a zero core dimension");
    }
    if (layerTypes.size() != numLayers) {
        return makeError(ErrorCode::MalformedData,
                         "layer_types has {} entries but num_hidden_layers is {}",
                         layerTypes.size(), numLayers);
    }
    if (numHeads == 0 || numKVHeads == 0) {
        return makeError(ErrorCode::MalformedData, "attention head counts must be non-zero");
    }
    if (numHeads % numKVHeads != 0) {
        return makeError(ErrorCode::MalformedData,
                         "num_attention_heads {} is not divisible by num_key_value_heads {}",
                         numHeads, numKVHeads);
    }
    if (numGlobalKVHeads != 0 && numHeads % numGlobalKVHeads != 0) {
        return makeError(ErrorCode::MalformedData,
                         "num_attention_heads {} is not divisible by "
                         "num_global_key_value_heads {}",
                         numHeads, numGlobalKVHeads);
    }
    if (topKExperts == 0 || topKExperts > numExperts) {
        return makeError(ErrorCode::MalformedData,
                         "top_k_experts {} is out of range for {} experts", topKExperts,
                         numExperts);
    }
    if (eosTokenIds.empty()) {
        return makeError(ErrorCode::MalformedData, "no end-of-sequence token ids");
    }
    if (hiddenActivation != "gelu_pytorch_tanh") {
        return makeError(ErrorCode::Unsupported,
                         "hidden activation '{}' is not implemented (expected "
                         "gelu_pytorch_tanh)",
                         hiddenActivation);
    }

    // Every quantized layout the runtime will build must divide evenly. Checking
    // here means a bad config fails at load with a named projection rather than
    // producing a subtly wrong stride during repack.
    TF_CHECK(embeddingLayout().validate());
    TF_CHECK(routerLayout().validate());
    TF_CHECK(sharedGateUpLayout().validate());
    TF_CHECK(sharedDownLayout().validate());
    TF_CHECK(expertGateUpLayout().validate());
    TF_CHECK(expertDownLayout().validate());

    for (u64 layer = 0; layer < numLayers; ++layer) {
        const auto describe = [&](Error error) {
            return std::unexpected(error.wrap(std::format("layer {}", layer)));
        };
        if (auto status = qProjLayout(layer).validate(); !status) {
            return describe(status.error());
        }
        if (auto status = kvProjLayout(layer).validate(); !status) {
            return describe(status.error());
        }
        if (auto status = oProjLayout(layer).validate(); !status) {
            return describe(status.error());
        }
    }

    return {};
}

Result<ArchInfo> ArchInfo::parseConfigJson(std::string_view text) {
    TF_TRY(const json::Value root, json::parse(text));

    // Multimodal checkpoints nest the language model under text_config. A
    // text-only export would place the same fields at the top level.
    const json::Value* textConfig = root.find("text_config");
    const json::Value& cfg = textConfig != nullptr ? *textConfig : root;

    ArchInfo arch;

    TF_TRY(arch.hiddenSize, requireUInt(cfg, "hidden_size"));
    TF_TRY(arch.numLayers, requireUInt(cfg, "num_hidden_layers"));
    TF_TRY(arch.vocabSize, requireUInt(cfg, "vocab_size"));
    TF_TRY(arch.numHeads, requireUInt(cfg, "num_attention_heads"));
    TF_TRY(arch.numKVHeads, requireUInt(cfg, "num_key_value_heads"));
    TF_TRY(arch.headDim, requireUInt(cfg, "head_dim"));
    TF_TRY(arch.intermediateSize, requireUInt(cfg, "intermediate_size"));
    TF_TRY(arch.moeIntermediateSize, requireUInt(cfg, "moe_intermediate_size"));
    TF_TRY(arch.numExperts, requireUInt(cfg, "num_experts"));
    TF_TRY(arch.topKExperts, requireUInt(cfg, "top_k_experts"));
    TF_TRY(arch.rmsNormEps, requireDouble(cfg, "rms_norm_eps"));

    arch.numGlobalKVHeads = optionalUInt(cfg, "num_global_key_value_heads", arch.numKVHeads);
    arch.globalHeadDim = optionalUInt(cfg, "global_head_dim", arch.headDim);
    arch.slidingWindow = optionalUInt(cfg, "sliding_window", 0);
    arch.maxPositionEmbeddings = optionalUInt(cfg, "max_position_embeddings", 0);
    arch.attentionKEqV = optionalBool(cfg, "attention_k_eq_v", false);
    arch.tieWordEmbeddings = optionalBool(cfg, "tie_word_embeddings", false);
    arch.finalLogitSoftcap = optionalDouble(cfg, "final_logit_softcapping", 0.0);
    arch.hiddenActivation = optionalString(cfg, "hidden_activation", "gelu_pytorch_tanh");

    // RoPE is configured per attention kind.
    arch.slidingRopeTheta =
            optionalDouble(cfg, "rope_parameters.sliding_attention.rope_theta", 10000.0);
    arch.fullRopeTheta =
            optionalDouble(cfg, "rope_parameters.full_attention.rope_theta",
                           optionalDouble(cfg, "rope_theta", arch.slidingRopeTheta));
    arch.partialRotaryFactor =
            optionalDouble(cfg, "rope_parameters.full_attention.partial_rotary_factor",
                           optionalDouble(cfg, "partial_rotary_factor", 1.0));

    arch.bosTokenId = static_cast<u32>(optionalUInt(cfg, "bos_token_id",
                                                    optionalUInt(root, "bos_token_id", 2)));
    arch.padTokenId = static_cast<u32>(optionalUInt(cfg, "pad_token_id",
                                                    optionalUInt(root, "pad_token_id", 0)));

    arch.eosTokenIds = parseTokenIdList(root.find("eos_token_id"));
    if (arch.eosTokenIds.empty()) {
        arch.eosTokenIds = parseTokenIdList(cfg.find("eos_token_id"));
    }

    TF_TRY(const auto quant, parseQuantization(root));
    arch.weightQuant = quant.first;
    arch.routerQuant = quant.second;

    // layer_types drives everything that varies per layer, so an absent or
    // wrong-length list is fatal rather than something to guess around.
    TF_TRY(const json::Value* typesNode, cfg.at("layer_types"));
    TF_TRY(const json::Array* types, typesNode->asArray());
    arch.layerTypes.reserve(types->size());
    for (usize i = 0; i < types->size(); ++i) {
        const auto name = (*types)[i].asString();
        if (!name) {
            return std::unexpected(name.error().wrap(std::format("layer_types[{}]", i)));
        }
        if (*name == "full_attention") {
            arch.layerTypes.push_back(AttentionKind::Full);
        } else if (*name == "sliding_attention") {
            arch.layerTypes.push_back(AttentionKind::Sliding);
        } else {
            return makeError(ErrorCode::Unsupported, "layer_types[{}] is '{}', which is unknown",
                             i, *name);
        }
    }

    TF_CHECK(arch.validate());
    return arch;
}

Result<ArchInfo> ArchInfo::readFromCheckpoint(const std::filesystem::path& checkpointDir) {
    const std::filesystem::path path = checkpointDir / "config.json";
    TF_TRY(const std::string text, io::readTextFile(path));

    auto arch = parseConfigJson(text);
    if (!arch) {
        return std::unexpected(arch.error().wrap(path.string()));
    }
    return arch;
}

}  // namespace tf
