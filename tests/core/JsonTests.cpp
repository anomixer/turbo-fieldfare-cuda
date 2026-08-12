#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <string>

#include "tf/core/json/Json.h"

using namespace tf;
using namespace tf::json;

namespace {

Value parseOrFail(std::string_view text) {
    auto parsed = parse(text);
    REQUIRE(parsed.has_value());
    return std::move(*parsed);
}

}  // namespace

TEST_CASE("scalars round-trip", "[json]") {
    CHECK(parseOrFail("null").isNull());
    CHECK(*parseOrFail("true").asBool());
    CHECK_FALSE(*parseOrFail("false").asBool());
    CHECK(*parseOrFail("42").asInt() == 42);
    CHECK(*parseOrFail("-17").asInt() == -17);
    CHECK(*parseOrFail("1e-06").asDouble() == 1e-06);
    CHECK(*parseOrFail("30.0").asDouble() == 30.0);
    CHECK(*parseOrFail("\"hello\"").asString() == "hello");
}

TEST_CASE("integers keep full precision", "[json]") {
    // Safetensors data_offsets run into the billions. Storing them as double
    // would be lossless today but is exactly the sort of thing that silently
    // breaks on a larger checkpoint, so integers stay integers.
    const Value value = parseOrFail("15340981404");
    REQUIRE(value.type() == Value::Type::Int);
    CHECK(*value.asUInt() == 15340981404ull);
    CHECK(value.dump() == "15340981404");
}

TEST_CASE("an integral double still reads as an int", "[json]") {
    const Value value = parseOrFail("30.0");
    REQUIRE(value.type() == Value::Type::Double);
    CHECK(*value.asInt() == 30);
}

TEST_CASE("negative values are rejected by asUInt", "[json]") {
    const auto result = parseOrFail("-1").asUInt();
    REQUIRE_FALSE(result.has_value());
    CHECK(result.error().code() == ErrorCode::MalformedData);
}

TEST_CASE("objects preserve insertion order for reproducible manifests", "[json]") {
    // Key order must be stable: manifest.json is hashed, so a reordering would
    // invalidate every previously written install receipt.
    const Value value = parseOrFail(R"({"zebra":1,"apple":2,"mango":3})");
    CHECK(value.dump() == R"({"zebra":1,"apple":2,"mango":3})");

    Value built = Value::makeObject();
    built.set("version", 1);
    built.set("arch", "gemma4");
    built.set("version", 2);  // updates in place, does not move to the end
    CHECK(built.dump() == R"({"version":2,"arch":"gemma4"})");
}

TEST_CASE("nested navigation by dotted path", "[json]") {
    const Value config = parseOrFail(R"({
        "text_config": {
            "num_experts": 128,
            "rope_parameters": { "sliding_attention": { "rope_theta": 10000.0 } }
        }
    })");

    const auto experts = config.path("text_config.num_experts");
    REQUIRE(experts.has_value());
    CHECK(*(*experts)->asUInt() == 128);

    const auto theta = config.path("text_config.rope_parameters.sliding_attention.rope_theta");
    REQUIRE(theta.has_value());
    CHECK(*(*theta)->asDouble() == 10000.0);

    const auto missing = config.path("text_config.nope");
    REQUIRE_FALSE(missing.has_value());
    // The error names both the full path and the offending segment.
    CHECK_THAT(missing.error().message(),
               Catch::Matchers::ContainsSubstring("text_config.nope"));
    CHECK_THAT(missing.error().message(), Catch::Matchers::ContainsSubstring("nope"));
}

TEST_CASE("arrays and mixed types", "[json]") {
    const Value value = parseOrFail(R"([1, "two", 3.5, true, null, [], {}])");
    const auto array = value.asArray();
    REQUIRE(array.has_value());
    REQUIRE((*array)->size() == 7);
    CHECK(*(**array)[0].asInt() == 1);
    CHECK(*(**array)[1].asString() == "two");
    CHECK(*(**array)[3].asBool());
    CHECK((**array)[4].isNull());
    CHECK((**array)[5].isArray());
    CHECK((**array)[6].isObject());
}

TEST_CASE("string escapes decode, including surrogate pairs", "[json]") {
    CHECK(*parseOrFail(R"("a\nb\tc")").asString() == "a\nb\tc");
    CHECK(*parseOrFail(R"("quote\"backslash\\slash\/")").asString() == "quote\"backslash\\slash/");
    // U+00E9
    CHECK(*parseOrFail(R"("é")").asString() == "\xc3\xa9");
    // U+1F600 as a surrogate pair - the tokenizer's added_tokens contain emoji.
    CHECK(*parseOrFail(R"("😀")").asString() == "\xf0\x9f\x98\x80");
}

TEST_CASE("escapes re-serialize safely", "[json]") {
    Value value = Value::makeObject();
    value.set("text", std::string{"line\nbreak\ttab\"quote\\slash"});
    const std::string dumped = value.dump();
    CHECK(dumped == R"({"text":"line\nbreak\ttab\"quote\\slash"})");

    // Round-trips back to the original bytes.
    const Value reparsed = parseOrFail(dumped);
    CHECK(*(*reparsed.at("text"))->asString() == "line\nbreak\ttab\"quote\\slash");
}

TEST_CASE("pretty printing is stable", "[json]") {
    Value root = Value::makeObject();
    root.set("bits", 4);
    Value nested = Value::makeObject();
    nested.set("group_size", 64);
    root.set("quant", std::move(nested));

    CHECK(root.dump(2) ==
          "{\n"
          "  \"bits\": 4,\n"
          "  \"quant\": {\n"
          "    \"group_size\": 64\n"
          "  }\n"
          "}");
}

TEST_CASE("malformed input is rejected with a byte offset", "[json]") {
    const auto cases = {
        R"({"a":1,})",       // trailing comma
        R"({"a" 1})",        // missing colon
        R"([1,2)",           // unterminated array
        R"("unterminated)",  // unterminated string
        R"({a:1})",          // unquoted key
        "{} extra",          // trailing content
        R"("\ud83d")",       // unpaired high surrogate
        R"("\q")",           // invalid escape
    };

    for (const auto* text : cases) {
        const auto result = parse(text);
        INFO("input: " << text);
        REQUIRE_FALSE(result.has_value());
        CHECK(result.error().code() == ErrorCode::MalformedData);
        CHECK_THAT(result.error().message(), Catch::Matchers::ContainsSubstring("json at byte"));
    }
}

TEST_CASE("deep nesting is bounded rather than blowing the stack", "[json]") {
    std::string deep(200, '[');
    deep.append(200, ']');
    const auto result = parse(deep);
    REQUIRE_FALSE(result.has_value());
    CHECK_THAT(result.error().message(), Catch::Matchers::ContainsSubstring("nesting too deep"));
}

TEST_CASE("a UTF-8 BOM is tolerated", "[json]") {
    CHECK(parseOrFail("\xEF\xBB\xBF{\"a\":1}").find("a") != nullptr);
}

TEST_CASE("typed accessors report what they found", "[json]") {
    const Value value = parseOrFail(R"({"count":"12"})");
    const auto asInt = (*value.at("count"))->asInt();
    REQUIRE_FALSE(asInt.has_value());
    CHECK_THAT(asInt.error().message(), Catch::Matchers::ContainsSubstring("expected int"));
    CHECK_THAT(asInt.error().message(), Catch::Matchers::ContainsSubstring("string"));

    const auto missing = value.at("absent");
    REQUIRE_FALSE(missing.has_value());
    CHECK_THAT(missing.error().message(), Catch::Matchers::ContainsSubstring("absent"));
}
