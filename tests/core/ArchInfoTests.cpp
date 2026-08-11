#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <cstdlib>
#include <filesystem>
#include <string>

#include "tf/core/format/ArchInfo.h"

using namespace tf;

namespace {

/// A minimal but valid Gemma-4-shaped config, small enough to mutate per test.
/// Six layers with full attention last, mirroring the real every-sixth pattern.
std::string miniConfig() {
    return R"({
      "eos_token_id": [1, 106, 50],
      "quantization": {
        "group_size": 64,
        "bits": 4,
        "mode": "affine",
        "language_model.model.layers.0.router.proj": {"group_size": 64, "bits": 8}
      },
      "text_config": {
        "hidden_size": 2816,
        "num_hidden_layers": 6,
        "vocab_size": 262144,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "num_global_key_value_heads": 2,
        "head_dim": 256,
        "global_head_dim": 512,
        "intermediate_size": 2112,
        "moe_intermediate_size": 704,
        "num_experts": 128,
        "top_k_experts": 8,
        "sliding_window": 1024,
        "rms_norm_eps": 1e-06,
        "final_logit_softcapping": 30.0,
        "tie_word_embeddings": true,
        "attention_k_eq_v": true,
        "hidden_activation": "gelu_pytorch_tanh",
        "max_position_embeddings": 262144,
        "bos_token_id": 2,
        "pad_token_id": 0,
        "rope_parameters": {
          "full_attention": {
            "partial_rotary_factor": 0.25,
            "rope_theta": 1000000.0,
            "rope_type": "proportional"
          },
          "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"}
        },
        "layer_types": [
          "sliding_attention", "sliding_attention", "sliding_attention",
          "sliding_attention", "sliding_attention", "full_attention"
        ]
      }
    })";
}

std::filesystem::path checkpointDir() {
    if (const char* override = std::getenv("TF_CHECKPOINT_DIR")) {
        return std::filesystem::path{override};
    }
    if (const char* home = std::getenv("USERPROFILE")) {
        return std::filesystem::path{home} / "model-data" / "gemma-4-26b-a4b-it-4bit";
    }
    return {};
}

}  // namespace

TEST_CASE("config parses into the expected architecture", "[arch]") {
    const auto arch = ArchInfo::parseConfigJson(miniConfig());
    REQUIRE(arch.has_value());

    CHECK(arch->hiddenSize == 2816);
    CHECK(arch->numLayers == 6);
    CHECK(arch->numExperts == 128);
    CHECK(arch->topKExperts == 8);
    CHECK(arch->vocabSize == 262144);
    CHECK(arch->attentionKEqV);
    CHECK(arch->tieWordEmbeddings);
    CHECK_THAT(arch->rmsNormEps, Catch::Matchers::WithinRel(1e-06, 1e-12));
    CHECK_THAT(arch->finalLogitSoftcap, Catch::Matchers::WithinRel(30.0, 1e-12));

    // The router override is picked up from the per-tensor quantization block.
    CHECK(arch->weightQuant == QuantSpec{.bits = 4, .groupSize = 64});
    CHECK(arch->routerQuant == QuantSpec{.bits = 8, .groupSize = 64});

    CHECK(arch->eosTokenIds == std::vector<u32>{1, 106, 50});
    CHECK(arch->bosTokenId == 2);
}

TEST_CASE("per-layer geometry differs by attention kind", "[arch]") {
    const auto arch = ArchInfo::parseConfigJson(miniConfig());
    REQUIRE(arch.has_value());

    CHECK(arch->slidingLayerCount() == 5);
    CHECK(arch->fullAttentionLayerCount() == 1);

    SECTION("sliding layer 0") {
        CHECK_FALSE(arch->isFullAttention(0));
        CHECK(arch->headDimFor(0) == 256);
        CHECK(arch->kvHeadsFor(0) == 8);
        CHECK(arch->qProjOutFeatures(0) == 4096);   // 16 heads x 256
        CHECK(arch->kvProjOutFeatures(0) == 2048);  // 8 kv heads x 256
        CHECK(arch->hasSeparateVProj(0));
        CHECK_THAT(arch->ropeThetaFor(0), Catch::Matchers::WithinRel(10000.0, 1e-12));
    }

    SECTION("full-attention layer 5") {
        CHECK(arch->isFullAttention(5));
        CHECK(arch->headDimFor(5) == 512);
        CHECK(arch->kvHeadsFor(5) == 2);
        CHECK(arch->qProjOutFeatures(5) == 8192);   // 16 heads x 512
        CHECK(arch->kvProjOutFeatures(5) == 1024);  // 2 global kv heads x 512
        CHECK_THAT(arch->ropeThetaFor(5), Catch::Matchers::WithinRel(1000000.0, 1e-12));

        // attention_k_eq_v: no separate V projection on full-attention layers.
        CHECK_FALSE(arch->hasSeparateVProj(5));
    }
}

TEST_CASE("derived layouts match the real tensor shapes", "[arch]") {
    const auto arch = ArchInfo::parseConfigJson(miniConfig());
    REQUIRE(arch.has_value());

    // Each expectation is a shape read out of the actual safetensors headers.
    CHECK(arch->qProjLayout(0).packedWordsPerRow() == 352);
    CHECK(arch->oProjLayout(0).packedWordsPerRow() == 512);
    CHECK(arch->oProjLayout(0).groupsPerRow() == 64);
    CHECK(arch->oProjLayout(5).packedWordsPerRow() == 1024);
    CHECK(arch->oProjLayout(5).groupsPerRow() == 128);
    CHECK(arch->sharedDownLayout().packedWordsPerRow() == 264);
    CHECK(arch->sharedDownLayout().groupsPerRow() == 33);
    CHECK(arch->expertGateUpLayout().packedWordsPerRow() == 352);
    CHECK(arch->expertDownLayout().packedWordsPerRow() == 88);
    CHECK(arch->routerLayout().packedWordsPerRow() == 704);  // 8-bit: 4 per word
    CHECK(arch->embeddingLayout().packedWordsPerRow() == 352);

    CHECK(arch->expertBlobBytes() == 3345408);
    CHECK(arch->totalExpertBytes() == 3345408ull * 128 * 6);
}

TEST_CASE("inconsistent configs are rejected", "[arch]") {
    SECTION("layer_types length must match num_hidden_layers") {
        std::string text = miniConfig();
        const auto pos = text.find("\"num_hidden_layers\": 6");
        REQUIRE(pos != std::string::npos);
        text.replace(pos, std::string("\"num_hidden_layers\": 6").size(),
                     "\"num_hidden_layers\": 7");

        const auto arch = ArchInfo::parseConfigJson(text);
        REQUIRE_FALSE(arch.has_value());
        CHECK_THAT(arch.error().message(),
                   Catch::Matchers::ContainsSubstring("layer_types has 6 entries"));
    }

    SECTION("unknown layer type") {
        std::string text = miniConfig();
        // "full_attention" also appears as a rope_parameters key, so anchor the
        // search inside layer_types.
        const auto listStart = text.find("\"layer_types\"");
        REQUIRE(listStart != std::string::npos);
        const auto pos = text.find("\"full_attention\"", listStart);
        REQUIRE(pos != std::string::npos);
        text.replace(pos, std::string("\"full_attention\"").size(), "\"linear_attention\"");

        const auto arch = ArchInfo::parseConfigJson(text);
        REQUIRE_FALSE(arch.has_value());
        CHECK(arch.error().code() == ErrorCode::Unsupported);
        CHECK_THAT(arch.error().message(),
                   Catch::Matchers::ContainsSubstring("linear_attention"));
    }

    SECTION("unsupported activation") {
        std::string text = miniConfig();
        const auto pos = text.find("gelu_pytorch_tanh");
        REQUIRE(pos != std::string::npos);
        text.replace(pos, std::string("gelu_pytorch_tanh").size(), "silu");

        const auto arch = ArchInfo::parseConfigJson(text);
        REQUIRE_FALSE(arch.has_value());
        CHECK(arch.error().code() == ErrorCode::Unsupported);
    }

    SECTION("non-affine quantization") {
        std::string text = miniConfig();
        const auto pos = text.find("\"mode\": \"affine\"");
        REQUIRE(pos != std::string::npos);
        text.replace(pos, std::string("\"mode\": \"affine\"").size(), "\"mode\": \"nf4\"");

        const auto arch = ArchInfo::parseConfigJson(text);
        REQUIRE_FALSE(arch.has_value());
        CHECK(arch.error().code() == ErrorCode::Unsupported);
        CHECK_THAT(arch.error().message(), Catch::Matchers::ContainsSubstring("nf4"));
    }

    SECTION("heads not divisible by kv heads") {
        std::string text = miniConfig();
        const auto pos = text.find("\"num_key_value_heads\": 8");
        REQUIRE(pos != std::string::npos);
        text.replace(pos, std::string("\"num_key_value_heads\": 8").size(),
                     "\"num_key_value_heads\": 7");

        const auto arch = ArchInfo::parseConfigJson(text);
        REQUIRE_FALSE(arch.has_value());
        CHECK_THAT(arch.error().message(), Catch::Matchers::ContainsSubstring("divisible"));
    }

    SECTION("a missing required field names itself") {
        std::string text = miniConfig();
        const auto pos = text.find("\"moe_intermediate_size\"");
        REQUIRE(pos != std::string::npos);
        text.replace(pos, std::string("\"moe_intermediate_size\"").size(), "\"unused_field\"");

        const auto arch = ArchInfo::parseConfigJson(text);
        REQUIRE_FALSE(arch.has_value());
        CHECK_THAT(arch.error().message(),
                   Catch::Matchers::ContainsSubstring("moe_intermediate_size"));
    }
}

TEST_CASE("the real checkpoint config parses and validates", "[arch][checkpoint]") {
    const auto dir = checkpointDir();
    if (dir.empty() || !std::filesystem::exists(dir / "config.json")) {
        SKIP("checkpoint not present at " << dir.string());
    }

    const auto arch = ArchInfo::readFromCheckpoint(dir);
    REQUIRE(arch.has_value());

    CHECK(arch->numLayers == 30);
    CHECK(arch->slidingLayerCount() == 25);
    CHECK(arch->fullAttentionLayerCount() == 5);

    // Full attention on every sixth layer.
    for (u64 layer = 0; layer < arch->numLayers; ++layer) {
        INFO("layer " << layer);
        CHECK(arch->isFullAttention(layer) == (layer % 6 == 5));
    }

    // The two totals the residency planner depends on.
    CHECK(arch->expertBlobBytes() == 3345408);
    CHECK(arch->totalExpertBytes() == 12846366720ull);
}
