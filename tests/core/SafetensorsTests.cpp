#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <array>
#include <bit>
#include <cstring>
#include <string>
#include <vector>

#include "tf/core/format/Quantization.h"
#include "tf/core/format/Safetensors.h"
#include "tf/core/json/Json.h"

using namespace tf;

namespace {

/// Builds the 8-byte little-endian length prefix a safetensors file starts with.
std::array<u8, 8> lengthPrefix(u64 value) {
    if constexpr (std::endian::native == std::endian::big) {
        value = std::byteswap(value);
    }
    std::array<u8, 8> bytes{};
    std::memcpy(bytes.data(), &value, sizeof(value));
    return bytes;
}

}  // namespace

TEST_CASE("dtype names round-trip and widths are right", "[safetensors]") {
    CHECK(*parseDType("BF16") == DType::BF16);
    CHECK(*parseDType("U32") == DType::U32);
    CHECK(byteWidth(DType::BF16) == 2);
    CHECK(byteWidth(DType::U32) == 4);
    CHECK(toString(DType::BF16) == "BF16");

    const auto unknown = parseDType("FP8_E4M3");
    REQUIRE_FALSE(unknown.has_value());
    CHECK(unknown.error().code() == ErrorCode::Unsupported);
}

TEST_CASE("header length prefix is validated", "[safetensors]") {
    CHECK(*SafetensorsHeader::readHeaderLength(lengthPrefix(64091)) == 64091);

    const std::array<u8, 4> tooShort{};
    CHECK_FALSE(SafetensorsHeader::readHeaderLength(tooShort).has_value());

    CHECK_FALSE(SafetensorsHeader::readHeaderLength(lengthPrefix(0)).has_value());

    // A corrupt prefix must not be trusted into a huge allocation.
    const auto absurd = SafetensorsHeader::readHeaderLength(lengthPrefix(1ull << 60));
    REQUIRE_FALSE(absurd.has_value());
    CHECK_THAT(absurd.error().message(), Catch::Matchers::ContainsSubstring("sanity limit"));
}

TEST_CASE("a minimal header parses", "[safetensors]") {
    const std::string body =
            R"({"a.weight":{"dtype":"U32","shape":[4,8],"data_offsets":[0,128]},)"
            R"("a.scales":{"dtype":"BF16","shape":[4,2],"data_offsets":[128,144]}})";

    const auto header = SafetensorsHeader::parse(body, body.size());
    REQUIRE(header.has_value());

    CHECK(header->dataOffset() == 8 + body.size());
    REQUIRE(header->tensors().size() == 2);

    const TensorEntry* weight = header->find("a.weight");
    REQUIRE(weight != nullptr);
    CHECK(weight->dtype == DType::U32);
    CHECK(weight->shape == std::vector<u64>{4, 8});
    CHECK(weight->elementCount() == 32);
    CHECK(weight->expectedBytes() == 128);
    CHECK(weight->dataRange == ByteRange{.offset = 0, .length = 128});
    CHECK(weight->shapeString() == "[4,8]");

    // fileRange shifts by the data section offset.
    const auto range = header->fileRange("a.scales");
    REQUIRE(range.has_value());
    CHECK(range->offset == header->dataOffset() + 128);
    CHECK(range->length == 16);

    CHECK(header->find("absent") == nullptr);
    CHECK_FALSE(header->require("absent").has_value());
}

TEST_CASE("shape and data_offsets must agree", "[safetensors]") {
    // [4,8] U32 is 128 bytes, but the offsets claim 100.
    const std::string body =
            R"({"a":{"dtype":"U32","shape":[4,8],"data_offsets":[0,100]}})";
    const auto header = SafetensorsHeader::parse(body, body.size());
    REQUIRE_FALSE(header.has_value());
    CHECK(header.error().code() == ErrorCode::MalformedData);
    CHECK_THAT(header.error().message(), Catch::Matchers::ContainsSubstring("implies 128 bytes"));
    // The failing tensor is named, not just the failure.
    CHECK_THAT(header.error().message(), Catch::Matchers::ContainsSubstring("a:"));
}

TEST_CASE("reversed data_offsets are rejected", "[safetensors]") {
    const std::string body = R"({"a":{"dtype":"U8","shape":[4],"data_offsets":[100,0]}})";
    const auto header = SafetensorsHeader::parse(body, body.size());
    REQUIRE_FALSE(header.has_value());
    CHECK_THAT(header.error().message(), Catch::Matchers::ContainsSubstring("precedes begin"));
}

TEST_CASE("__metadata__ is captured, not treated as a tensor", "[safetensors]") {
    const std::string body =
            R"({"__metadata__":{"format":"pt"},)"
            R"("a":{"dtype":"U8","shape":[2],"data_offsets":[0,2]}})";
    const auto header = SafetensorsHeader::parse(body, body.size());
    REQUIRE(header.has_value());
    CHECK(header->tensors().size() == 1);
    REQUIRE(header->metadata().size() == 1);
    CHECK(header->metadata()[0].first == "format");
    CHECK(header->metadata()[0].second == "pt");
}

TEST_CASE("index maps tensors to shards and dedupes shard names", "[safetensors]") {
    const std::string text = R"({
        "metadata": {"total_size": 15340981404},
        "weight_map": {
            "a": "model-00001-of-00003.safetensors",
            "b": "model-00001-of-00003.safetensors",
            "c": "model-00002-of-00003.safetensors"
        }
    })";

    const auto index = SafetensorsIndex::parse(text);
    REQUIRE(index.has_value());
    CHECK(index->totalSize() == 15340981404ull);
    CHECK(index->weightMap().size() == 3);
    CHECK(index->shardFiles().size() == 2);
    CHECK(index->shardFiles()[0] == "model-00001-of-00003.safetensors");
    CHECK(*index->requireShardFor("c") == "model-00002-of-00003.safetensors");
    CHECK(index->shardFor("missing") == nullptr);
    CHECK_FALSE(index->requireShardFor("missing").has_value());
}

// ---------------------------------------------------------------------------
// Quantization layout
// ---------------------------------------------------------------------------

TEST_CASE("MLX affine layout matches the pinned checkpoint", "[quant]") {
    // Every expectation below was read out of the real safetensors headers, so
    // this pins our understanding of the format against the actual data.

    SECTION("routed expert gate/up projection") {
        const QuantizedLinearLayout gate{
                .outFeatures = 704,   // moe_intermediate_size
                .inFeatures = 2816,   // hidden_size
                .spec = kWeightQuant};
        REQUIRE(gate.validate().has_value());
        CHECK(gate.valuesPerWord() == 8);        // 4-bit: 8 nibbles per u32
        CHECK(gate.packedWordsPerRow() == 352);  // matches [128,704,352]
        CHECK(gate.groupsPerRow() == 44);        // matches [128,704,44]
        CHECK(gate.weightBytes() == 991232);
        CHECK(gate.scaleBytes() == 61952);
        CHECK(gate.totalBytes() == 1115136);
    }

    SECTION("routed expert down projection") {
        const QuantizedLinearLayout down{
                .outFeatures = 2816, .inFeatures = 704, .spec = kWeightQuant};
        REQUIRE(down.validate().has_value());
        CHECK(down.packedWordsPerRow() == 88);  // matches [128,2816,88]
        CHECK(down.groupsPerRow() == 11);       // matches [128,2816,11]
        CHECK(down.totalBytes() == 1115136);
    }

    SECTION("one expert blob is 3,345,408 bytes") {
        const QuantizedLinearLayout gate{
                .outFeatures = 704, .inFeatures = 2816, .spec = kWeightQuant};
        const QuantizedLinearLayout down{
                .outFeatures = 2816, .inFeatures = 704, .spec = kWeightQuant};
        const u64 perExpert = 2 * gate.totalBytes() + down.totalBytes();
        CHECK(perExpert == 3345408);

        // 128 experts x 30 layers is the ~12 GiB the streamer manages.
        CHECK(perExpert * 128 * 30 == 12846366720ull);

        // 4 KiB alignment costs 1 KiB per expert. 2 MiB alignment would round
        // each 3.19 MiB blob to 4 MiB - a 25% penalty - which is why the packed
        // layout aligns to a sector, not a large page.
        CHECK(alignUp(perExpert, align::kSector) - perExpert == 1024);
        CHECK(alignUp(perExpert, align::kLargePage) - perExpert == 848896);
    }

    SECTION("router projection is 8-bit") {
        const QuantizedLinearLayout router{
                .outFeatures = 128, .inFeatures = 2816, .spec = kRouterQuant};
        REQUIRE(router.validate().has_value());
        CHECK(router.valuesPerWord() == 4);      // 8-bit: 4 bytes per u32
        CHECK(router.packedWordsPerRow() == 704);  // matches [128,704]
        CHECK(router.groupsPerRow() == 44);        // matches [128,44]
    }

    SECTION("sliding vs full attention projections") {
        // Layer 0 (sliding): 16 heads x 256 head_dim = 4096
        const QuantizedLinearLayout qSliding{
                .outFeatures = 4096, .inFeatures = 2816, .spec = kWeightQuant};
        CHECK(qSliding.packedWordsPerRow() == 352);
        CHECK(qSliding.groupsPerRow() == 44);

        // Layer 5 (full): 16 heads x 512 global_head_dim = 8192
        const QuantizedLinearLayout qFull{
                .outFeatures = 8192, .inFeatures = 2816, .spec = kWeightQuant};
        CHECK(qFull.packedWordsPerRow() == 352);

        // o_proj consumes the attention output, so its inFeatures differ too.
        const QuantizedLinearLayout oFull{
                .outFeatures = 2816, .inFeatures = 8192, .spec = kWeightQuant};
        CHECK(oFull.packedWordsPerRow() == 1024);  // matches [2816,1024]
        CHECK(oFull.groupsPerRow() == 128);        // matches [2816,128]
    }
}

TEST_CASE("indivisible quantization dimensions are rejected", "[quant]") {
    SECTION("not a whole number of packed words") {
        // 2810 % 8 != 0, so the packed row length would truncate.
        const QuantizedLinearLayout bad{
                .outFeatures = 704, .inFeatures = 2810, .spec = kWeightQuant};
        const auto status = bad.validate();
        REQUIRE_FALSE(status.has_value());
        CHECK_THAT(status.error().message(),
                   Catch::Matchers::ContainsSubstring("values per 32-bit word"));
    }

    SECTION("not a whole number of quantization groups") {
        // 2808 packs evenly into u32 words (2808 / 8 = 351) but is not a
        // multiple of 64, so the scale/bias count would truncate instead.
        const QuantizedLinearLayout bad{
                .outFeatures = 704, .inFeatures = 2808, .spec = kWeightQuant};
        const auto status = bad.validate();
        REQUIRE_FALSE(status.has_value());
        CHECK_THAT(status.error().message(), Catch::Matchers::ContainsSubstring("group size"));
    }

    SECTION("unsupported bit width") {
        const QuantizedLinearLayout bad{
                .outFeatures = 16, .inFeatures = 64, .spec = QuantSpec{.bits = 3, .groupSize = 64}};
        CHECK(bad.validate().error().code() == ErrorCode::Unsupported);
    }

    SECTION("zero dimensions") {
        const QuantizedLinearLayout bad{
                .outFeatures = 0, .inFeatures = 2816, .spec = kWeightQuant};
        CHECK(bad.validate().error().code() == ErrorCode::InvalidArgument);
    }
}
