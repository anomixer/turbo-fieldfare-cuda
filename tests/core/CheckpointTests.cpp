// Integration tests against the real pinned checkpoint.
//
// These read only safetensors headers and the JSON sidecars - tens of KB out of
// 14.3 GB - so they are fast enough to run in the normal suite. They exist to
// keep our understanding of the format pinned to the actual data rather than to
// upstream's prose.
//
// Skipped, not failed, when the checkpoint is absent. Point TF_CHECKPOINT_DIR at
// a checkout to override the default location.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <cstdlib>
#include <filesystem>
#include <format>
#include <set>
#include <string>

#include "tf/core/format/Quantization.h"
#include "tf/core/format/Safetensors.h"
#include "tf/core/io/File.h"
#include "tf/core/json/Json.h"

using namespace tf;

namespace {

std::filesystem::path checkpointDir() {
    if (const char* override = std::getenv("TF_CHECKPOINT_DIR")) {
        return std::filesystem::path{override};
    }
    if (const char* home = std::getenv("USERPROFILE")) {
        return std::filesystem::path{home} / "model-data" / "gemma-4-26b-a4b-it-4bit";
    }
    return {};
}

bool checkpointAvailable() {
    const auto dir = checkpointDir();
    return !dir.empty() && std::filesystem::exists(dir / "model.safetensors.index.json");
}

#define REQUIRE_CHECKPOINT()                                                        \
    if (!checkpointAvailable()) {                                                   \
        SKIP("checkpoint not present at " << checkpointDir().string()               \
                                          << " - run scripts/fetch-checkpoint.ps1"); \
    }

constexpr u64 kNumLayers = 30;
constexpr u64 kNumExperts = 128;
constexpr u64 kHiddenSize = 2816;
constexpr u64 kMoeIntermediate = 704;

/// Full attention lands on every sixth layer; the rest use sliding-window
/// attention. Read from config.json's layer_types.
bool isFullAttentionLayer(u64 layer) { return layer % 6 == 5; }

}  // namespace

TEST_CASE("checkpoint index matches expectations", "[checkpoint]") {
    REQUIRE_CHECKPOINT();

    const auto index = readSafetensorsIndex(checkpointDir());
    REQUIRE(index.has_value());

    CHECK(index->weightMap().size() == 1697);
    CHECK(index->shardFiles().size() == 3);
    CHECK(index->totalSize() == 15340981404ull);
}

TEST_CASE("every indexed tensor resolves in its shard", "[checkpoint]") {
    REQUIRE_CHECKPOINT();

    const auto index = readSafetensorsIndex(checkpointDir());
    REQUIRE(index.has_value());

    // Parse each shard header once, then confirm the index and headers agree.
    std::vector<std::pair<std::string, SafetensorsHeader>> shards;
    for (const auto& shard : index->shardFiles()) {
        auto header = readSafetensorsHeader(checkpointDir() / shard);
        REQUIRE(header.has_value());
        shards.emplace_back(shard, std::move(*header));
    }

    u64 resolved = 0;
    for (const auto& [tensorName, shardName] : index->weightMap()) {
        const SafetensorsHeader* header = nullptr;
        for (const auto& [name, parsed] : shards) {
            if (name == shardName) {
                header = &parsed;
                break;
            }
        }
        REQUIRE(header != nullptr);

        INFO("tensor: " << tensorName << " in " << shardName);
        REQUIRE(header->find(tensorName) != nullptr);
        ++resolved;
    }
    CHECK(resolved == 1697);
}

TEST_CASE("architecture constants match config.json", "[checkpoint]") {
    REQUIRE_CHECKPOINT();

    const auto text = io::readTextFile(checkpointDir() / "config.json");
    REQUIRE(text.has_value());
    const auto config = json::parse(*text);
    REQUIRE(config.has_value());

    const auto readUInt = [&](std::string_view path) -> u64 {
        const auto node = config->path(path);
        REQUIRE(node.has_value());
        const auto value = (*node)->asUInt();
        REQUIRE(value.has_value());
        return *value;
    };

    CHECK(readUInt("text_config.hidden_size") == kHiddenSize);
    CHECK(readUInt("text_config.num_hidden_layers") == kNumLayers);
    CHECK(readUInt("text_config.num_attention_heads") == 16);
    CHECK(readUInt("text_config.num_key_value_heads") == 8);
    CHECK(readUInt("text_config.num_global_key_value_heads") == 2);
    CHECK(readUInt("text_config.head_dim") == 256);
    CHECK(readUInt("text_config.global_head_dim") == 512);
    CHECK(readUInt("text_config.intermediate_size") == 2112);
    CHECK(readUInt("text_config.moe_intermediate_size") == kMoeIntermediate);
    CHECK(readUInt("text_config.num_experts") == kNumExperts);
    CHECK(readUInt("text_config.top_k_experts") == 8);
    CHECK(readUInt("text_config.vocab_size") == 262144);
    CHECK(readUInt("text_config.sliding_window") == 1024);

    // Quantization: 4-bit affine at group 64.
    CHECK(readUInt("quantization.bits") == 4);
    CHECK(readUInt("quantization.group_size") == 64);
    CHECK(*(*config->path("quantization.mode"))->asString() == "affine");

    // K and V are shared on full-attention layers, which is why v_proj appears
    // only 25 times rather than 30.
    CHECK(*(*config->path("text_config.attention_k_eq_v"))->asBool());
    CHECK(*(*config->path("text_config.tie_word_embeddings"))->asBool());
}

TEST_CASE("layer_types place full attention on every sixth layer", "[checkpoint]") {
    REQUIRE_CHECKPOINT();

    const auto text = io::readTextFile(checkpointDir() / "config.json");
    const auto config = json::parse(*text);
    REQUIRE(config.has_value());

    const auto node = config->path("text_config.layer_types");
    REQUIRE(node.has_value());
    const auto types = (*node)->asArray();
    REQUIRE(types.has_value());
    REQUIRE((*types)->size() == kNumLayers);

    u64 fullCount = 0;
    for (u64 layer = 0; layer < kNumLayers; ++layer) {
        const auto kind = (**types)[layer].asString();
        REQUIRE(kind.has_value());
        const bool isFull = (*kind == "full_attention");
        INFO("layer " << layer << " is " << *kind);
        CHECK(isFull == isFullAttentionLayer(layer));
        fullCount += isFull ? 1 : 0;
    }
    CHECK(fullCount == 5);
}

TEST_CASE("expert tensor shapes match the computed layout", "[checkpoint]") {
    REQUIRE_CHECKPOINT();

    const auto index = readSafetensorsIndex(checkpointDir());
    REQUIRE(index.has_value());

    const auto shard = index->requireShardFor(
            "language_model.model.layers.0.experts.switch_glu.gate_proj.weight");
    REQUIRE(shard.has_value());
    const auto header = readSafetensorsHeader(checkpointDir() / *shard);
    REQUIRE(header.has_value());

    const QuantizedLinearLayout gate{
            .outFeatures = kMoeIntermediate, .inFeatures = kHiddenSize, .spec = kWeightQuant};
    const QuantizedLinearLayout down{
            .outFeatures = kHiddenSize, .inFeatures = kMoeIntermediate, .spec = kWeightQuant};

    const auto expectShape = [&](std::string_view name, const std::vector<u64>& shape) {
        INFO("tensor: " << name);
        const TensorEntry* entry = header->find(name);
        REQUIRE(entry != nullptr);
        CHECK(entry->shape == shape);
    };

    // Experts are stored stacked: one tensor per layer holding all 128.
    expectShape("language_model.model.layers.0.experts.switch_glu.gate_proj.weight",
                {kNumExperts, kMoeIntermediate, gate.packedWordsPerRow()});
    expectShape("language_model.model.layers.0.experts.switch_glu.gate_proj.scales",
                {kNumExperts, kMoeIntermediate, gate.groupsPerRow()});
    expectShape("language_model.model.layers.0.experts.switch_glu.down_proj.weight",
                {kNumExperts, kHiddenSize, down.packedWordsPerRow()});
    expectShape("language_model.model.layers.0.experts.switch_glu.down_proj.scales",
                {kNumExperts, kHiddenSize, down.groupsPerRow()});

    // The per-expert blob size the streamer reads and the residency planner
    // budgets against.
    const u64 perExpert = 2 * gate.totalBytes() + down.totalBytes();
    CHECK(perExpert == 3345408);
}

TEST_CASE("v_proj is present only on sliding-attention layers", "[checkpoint]") {
    REQUIRE_CHECKPOINT();

    const auto index = readSafetensorsIndex(checkpointDir());
    REQUIRE(index.has_value());

    std::set<std::string> present;
    for (const auto& [tensorName, shardName] : index->weightMap()) {
        present.insert(tensorName);
    }

    u64 slidingWithV = 0;
    for (u64 layer = 0; layer < kNumLayers; ++layer) {
        const std::string vProj =
                std::format("language_model.model.layers.{}.self_attn.v_proj.weight", layer);
        const bool hasV = present.contains(vProj);

        INFO("layer " << layer << (isFullAttentionLayer(layer) ? " (full)" : " (sliding)"));
        // attention_k_eq_v: full-attention layers reuse K as V, so they carry no
        // separate v_proj. Getting this wrong would silently corrupt attention
        // on 5 of 30 layers.
        CHECK(hasV == !isFullAttentionLayer(layer));
        slidingWithV += hasV ? 1 : 0;
    }
    CHECK(slidingWithV == 25);
}

TEST_CASE("attention projection widths differ by layer type", "[checkpoint]") {
    REQUIRE_CHECKPOINT();

    const auto index = readSafetensorsIndex(checkpointDir());
    REQUIRE(index.has_value());

    std::vector<std::pair<std::string, SafetensorsHeader>> shards;
    for (const auto& shard : index->shardFiles()) {
        auto header = readSafetensorsHeader(checkpointDir() / shard);
        REQUIRE(header.has_value());
        shards.emplace_back(shard, std::move(*header));
    }

    const auto lookup = [&](const std::string& name) -> const TensorEntry* {
        for (const auto& [shardName, header] : shards) {
            if (const TensorEntry* entry = header.find(name)) {
                return entry;
            }
        }
        return nullptr;
    };

    // Sliding layer 0: 16 heads x 256 = 4096 for Q, 8 kv heads x 256 = 2048 for K.
    const TensorEntry* q0 = lookup("language_model.model.layers.0.self_attn.q_proj.weight");
    REQUIRE(q0 != nullptr);
    CHECK(q0->shape == std::vector<u64>{4096, 352});

    const TensorEntry* k0 = lookup("language_model.model.layers.0.self_attn.k_proj.weight");
    REQUIRE(k0 != nullptr);
    CHECK(k0->shape == std::vector<u64>{2048, 352});

    // Full layer 5: 16 heads x 512 = 8192 for Q, 2 global kv heads x 512 = 1024.
    const TensorEntry* q5 = lookup("language_model.model.layers.5.self_attn.q_proj.weight");
    REQUIRE(q5 != nullptr);
    CHECK(q5->shape == std::vector<u64>{8192, 352});

    const TensorEntry* k5 = lookup("language_model.model.layers.5.self_attn.k_proj.weight");
    REQUIRE(k5 != nullptr);
    CHECK(k5->shape == std::vector<u64>{1024, 352});

    // Per-head norms follow the layer's head dimension.
    CHECK(lookup("language_model.model.layers.0.self_attn.q_norm.weight")->shape ==
          std::vector<u64>{256});
    CHECK(lookup("language_model.model.layers.5.self_attn.q_norm.weight")->shape ==
          std::vector<u64>{512});
}

TEST_CASE("embeddings are 4-bit quantized and tied to the head", "[checkpoint]") {
    REQUIRE_CHECKPOINT();

    const auto index = readSafetensorsIndex(checkpointDir());
    REQUIRE(index.has_value());

    const auto shard = index->requireShardFor("language_model.model.embed_tokens.weight");
    REQUIRE(shard.has_value());
    const auto header = readSafetensorsHeader(checkpointDir() / *shard);
    REQUIRE(header.has_value());

    const QuantizedLinearLayout embed{
            .outFeatures = 262144, .inFeatures = kHiddenSize, .spec = kWeightQuant};

    const TensorEntry* weight = header->find("language_model.model.embed_tokens.weight");
    REQUIRE(weight != nullptr);
    CHECK(weight->dtype == DType::U32);
    CHECK(weight->shape == std::vector<u64>{262144, embed.packedWordsPerRow()});

    const TensorEntry* scales = header->find("language_model.model.embed_tokens.scales");
    REQUIRE(scales != nullptr);
    CHECK(scales->dtype == DType::BF16);
    CHECK(scales->shape == std::vector<u64>{262144, embed.groupsPerRow()});

    // tie_word_embeddings means there is no separate lm_head tensor.
    CHECK(index->shardFor("lm_head.weight") == nullptr);
    CHECK(index->shardFor("language_model.lm_head.weight") == nullptr);
}

TEST_CASE("the vision tower is present and will be skipped at repack", "[checkpoint]") {
    REQUIRE_CHECKPOINT();

    const auto index = readSafetensorsIndex(checkpointDir());
    REQUIRE(index.has_value());

    u64 visionTensors = 0;
    u64 languageTensors = 0;
    for (const auto& [tensorName, shardName] : index->weightMap()) {
        if (tensorName.starts_with("vision_tower.") || tensorName.starts_with("embed_vision.")) {
            ++visionTensors;
        } else if (tensorName.starts_with("language_model.")) {
            ++languageTensors;
        }
    }

    // The checkpoint is a Gemma4ForConditionalGeneration wrapper. The port is
    // text-only, so these tensors are the difference between the 15.3 GB source
    // and the 14.3 GB install.
    CHECK(visionTensors > 0);
    CHECK(languageTensors > 0);
    CHECK(visionTensors + languageTensors == 1697);
}
