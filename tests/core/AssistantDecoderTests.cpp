// Separating the assistant's answer from its channel markup.
//
// This is what stands between the user and "<|channel>thought\n<channel|>Paris"
// where they asked for "Paris". The markers are real tokens in this checkpoint,
// so the tests drive the decoder with the ids the tokenizer reports rather than
// with hard-coded numbers.
//
// Skipped when no install is present.

#include <catch2/catch_test_macros.hpp>

#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#include "tf/core/tokenizer/AssistantDecoder.h"
#include "tf/core/tokenizer/Tokenizer.h"

using namespace tf;

namespace {

std::filesystem::path tokenizerPath() {
    if (const char* override = std::getenv("TF_GTURBO_DIR")) {
        return std::filesystem::path{override} / "tokenizer" / "tokenizer.json";
    }
    if (const char* home = std::getenv("USERPROFILE")) {
        return std::filesystem::path{home} / "model-data" / "gemma4.gturbo" / "tokenizer" /
               "tokenizer.json";
    }
    return {};
}

const Tokenizer* sharedTokenizer() {
    static auto loaded = [] {
        const auto path = tokenizerPath();
        if (path.empty() || !std::filesystem::exists(path)) {
            return Result<Tokenizer>{
                    std::unexpected(Error{ErrorCode::NotFound, "tokenizer.json not present"})};
        }
        return Tokenizer::loadFromFile(path);
    }();
    return loaded.has_value() ? &*loaded : nullptr;
}

#define REQUIRE_TOKENIZER()                                    \
    const Tokenizer* tokenizerPtr = sharedTokenizer();         \
    if (tokenizerPtr == nullptr) {                             \
        SKIP("no tokenizer.json available");                   \
    }                                                          \
    const Tokenizer& tokenizer = *tokenizerPtr

/// Feeds a token sequence through the decoder and concatenates each side.
struct Split {
    std::string content;
    std::string thinking;
};

[[nodiscard]] Split run(const Tokenizer& tokenizer, const std::vector<u32>& tokens) {
    AssistantDecoder decoder{tokenizer};
    Split split;
    for (const u32 token : tokens) {
        const auto update = decoder.push(token);
        split.content += update.content;
        split.thinking += update.thinking;
    }
    return split;
}

}  // namespace

TEST_CASE("the channel markers are present and recognized", "[assistant]") {
    REQUIRE_TOKENIZER();

    AssistantDecoder decoder{tokenizer};
    CHECK(decoder.structured());
    CHECK_FALSE(decoder.inChannel());

    // Named lookup, so the decoder does not depend on ids staying put.
    CHECK(tokenizer.idFor("<|channel>").has_value());
    CHECK(tokenizer.idFor("<channel|>").has_value());
}

TEST_CASE("an empty thought channel is stripped from the answer", "[assistant]") {
    REQUIRE_TOKENIZER();

    // Exactly what the model produces with thinking disabled: it still opens
    // and closes the channel, and the answer follows.
    std::vector<u32> tokens{*tokenizer.idFor("<|channel>")};
    for (const u32 id : tokenizer.encode("thought\n")) {
        tokens.push_back(id);
    }
    tokens.push_back(*tokenizer.idFor("<channel|>"));
    for (const u32 id : tokenizer.encode("Paris")) {
        tokens.push_back(id);
    }

    const Split split = run(tokenizer, tokens);
    CHECK(split.content == "Paris");
    CHECK(split.thinking.find("thought") != std::string::npos);
}

TEST_CASE("thinking is routed away from the answer, not into it", "[assistant]") {
    REQUIRE_TOKENIZER();

    std::vector<u32> tokens{*tokenizer.idFor("<|channel>")};
    for (const u32 id : tokenizer.encode("thought\nThe user wants a city.")) {
        tokens.push_back(id);
    }
    tokens.push_back(*tokenizer.idFor("<channel|>"));
    for (const u32 id : tokenizer.encode("Paris.")) {
        tokens.push_back(id);
    }

    const Split split = run(tokenizer, tokens);
    CHECK(split.content == "Paris.");
    CHECK(split.thinking.find("The user wants a city.") != std::string::npos);
    // The whole point: nothing from the channel reaches the answer.
    CHECK(split.content.find("user wants") == std::string::npos);
}

TEST_CASE("text with no channel at all passes straight through", "[assistant]") {
    REQUIRE_TOKENIZER();

    const Split split = run(tokenizer, tokenizer.encode("Just an answer."));
    CHECK(split.content == "Just an answer.");
    CHECK(split.thinking.empty());
}

TEST_CASE("the decoder tracks whether it is inside a channel", "[assistant]") {
    REQUIRE_TOKENIZER();

    AssistantDecoder decoder{tokenizer};
    CHECK_FALSE(decoder.inChannel());

    static_cast<void>(decoder.push(*tokenizer.idFor("<|channel>")));
    CHECK(decoder.inChannel());

    static_cast<void>(decoder.push(*tokenizer.idFor("<channel|>")));
    CHECK_FALSE(decoder.inChannel());

    decoder.reset();
    CHECK_FALSE(decoder.inChannel());
}

TEST_CASE("marker tokens themselves emit nothing", "[assistant]") {
    REQUIRE_TOKENIZER();

    AssistantDecoder decoder{tokenizer};
    for (const std::string_view marker : {"<|channel>", "<channel|>"}) {
        INFO("marker " << marker);
        const auto update = decoder.push(*tokenizer.idFor(marker));
        CHECK(update.content.empty());
        CHECK(update.thinking.empty());
    }
}

TEST_CASE("other special tokens do not leak into the answer", "[assistant]") {
    REQUIRE_TOKENIZER();

    // An end-of-turn token reaching the decoder - because a caller did not list
    // it as a stop token, say - must not print as "<turn|>".
    AssistantDecoder decoder{tokenizer};
    const auto update = decoder.push(*tokenizer.idFor("<turn|>"));
    CHECK(update.content.empty());
    CHECK(update.thinking.empty());
}
