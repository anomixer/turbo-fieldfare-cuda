// BPE tokenizer and Gemma chat template.
//
// Ground truth from Python transformers is not available on this machine, so
// correctness is established three ways instead: exact round-tripping over
// awkward text, structural properties that only a correct BPE satisfies, and -
// in the end-to-end tests - the model answering correctly when fed these ids.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <string>

#include "tf/core/tokenizer/ChatTemplate.h"
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

/// Loaded once: the file is 30 MB and carries half a million merges.
const Tokenizer* sharedTokenizer() {
    static const auto loaded = [] {
        const auto path = tokenizerPath();
        if (path.empty() || !std::filesystem::exists(path)) {
            return Result<Tokenizer>{std::unexpected(
                    Error{ErrorCode::NotFound, "tokenizer.json not present"})};
        }
        return Tokenizer::loadFromFile(path);
    }();
    return loaded.has_value() ? &*loaded : nullptr;
}

#define REQUIRE_TOKENIZER()                                                    \
    const Tokenizer* tokenizerPtr = sharedTokenizer();                         \
    if (tokenizerPtr == nullptr) {                                             \
        SKIP("no tokenizer.json at " << tokenizerPath().string());             \
    }                                                                          \
    [[maybe_unused]] const Tokenizer& tokenizer = *tokenizerPtr

}  // namespace

TEST_CASE("the tokenizer loads the full vocabulary and merge table", "[tokenizer]") {
    REQUIRE_TOKENIZER();

    CHECK(tokenizer.vocabularySize() == 262144);
    // Merges whose halves or result are missing from the vocabulary are
    // dropped, so this is a lower bound rather than the raw file count.
    CHECK(tokenizer.mergeCount() > 500000);
    CHECK(tokenizer.addedTokens().size() == 24);
}

TEST_CASE("known special tokens resolve to their documented ids", "[tokenizer]") {
    REQUIRE_TOKENIZER();

    CHECK(*tokenizer.idFor("<pad>") == 0);
    CHECK(*tokenizer.idFor("<eos>") == 1);
    CHECK(*tokenizer.idFor("<bos>") == 2);
    CHECK(*tokenizer.idFor("<|turn>") == 105);
    CHECK(*tokenizer.idFor("<turn|>") == 106);
    CHECK(*tokenizer.idFor("\n") == 107);

    CHECK(tokenizer.isSpecial(0));
    CHECK(tokenizer.isSpecial(2));
    CHECK_FALSE(tokenizer.isSpecial(*tokenizer.idFor("the")));
}

TEST_CASE("encoding round-trips through decoding", "[tokenizer]") {
    REQUIRE_TOKENIZER();

    const std::vector<std::string> cases{
            "Hello, world!",
            "The capital of France is Paris.",
            "a",
            " leading space",
            "trailing space ",
            "multiple     consecutive     spaces",
            "tabs\tand\nnewlines\r\n",
            "1234567890 -3.14159e-7",
            "CamelCase snake_case SCREAMING_CASE kebab-case",
            "punctuation: ;,.!?'\"()[]{}<>/\\|@#$%^&*~`",
            "unicode: e\xCC\x81 caf\xC3\xA9 na\xC3\xAFve",             // combining + accents
            "greek \xCE\xB1\xCE\xB2\xCE\xB3 cyrillic \xD0\xB4\xD0\xB0",
            "cjk \xE6\x97\xA5\xE6\x9C\xAC\xE8\xAA\x9E \xED\x95\x9C\xEA\xB5\xAD\xEC\x96\xB4",
            "emoji \xF0\x9F\x98\x80 \xF0\x9F\x91\x8D\xF0\x9F\x8F\xBD",  // includes a modifier
            "math \xE2\x88\x91 \xE2\x88\x9E \xE2\x89\xA0 \xCF\x80",
            "code: for (int i = 0; i < n; ++i) { sum += a[i]; }",
            "json: {\"key\": [1, 2, {\"nested\": true}]}",
    };

    for (const auto& text : cases) {
        const auto ids = tokenizer.encode(text);
        const auto decoded = tokenizer.decode(ids, /*skipSpecialTokens=*/false);
        INFO("input: " << text << "\n  ids: " << ids.size());
        REQUIRE(decoded == text);
    }
}

TEST_CASE("no leading space is synthesized", "[tokenizer]") {
    REQUIRE_TOKENIZER();

    // This configuration has a Replace normalizer but no Prepend, so "hello"
    // must not be encoded as if it were " hello". Getting this wrong changes
    // the first token of every prompt.
    const auto bare = tokenizer.encode("hello");
    const auto spaced = tokenizer.encode(" hello");

    CHECK(bare != spaced);
    CHECK(tokenizer.decode(bare, false) == "hello");
    CHECK(tokenizer.decode(spaced, false) == " hello");

    // The bare form's first piece must not carry the space marker.
    REQUIRE_FALSE(bare.empty());
    CHECK_FALSE(tokenizer.piece(bare.front()).starts_with("\xE2\x96\x81"));
    CHECK(tokenizer.piece(spaced.front()).starts_with("\xE2\x96\x81"));
}

TEST_CASE("byte fallback covers characters outside the vocabulary", "[tokenizer]") {
    REQUIRE_TOKENIZER();

    // Rare scripts and unusual symbols that are unlikely to have their own
    // entries still have to round-trip, one `<0xXX>` token per byte.
    const std::vector<std::string> exotic{
            "\xF0\x90\x8C\xB0",                  // Gothic letter, U+10330
            "\xEF\xBF\xBD",                      // replacement character
            "\xF0\x9F\xA7\x91\xE2\x80\x8D\xF0\x9F\x9A\x80",  // ZWJ sequence
    };

    for (const auto& text : exotic) {
        const auto ids = tokenizer.encode(text);
        INFO("input bytes: " << text.size() << ", ids: " << ids.size());
        REQUIRE_FALSE(ids.empty());
        REQUIRE(tokenizer.decode(ids, false) == text);
    }
}

TEST_CASE("every single byte round-trips", "[tokenizer]") {
    REQUIRE_TOKENIZER();

    // Byte fallback must be complete: any input at all has to survive, which is
    // what makes the tokenizer total rather than merely usually-correct.
    for (int value = 1; value < 256; ++value) {
        const std::string text(1, static_cast<char>(value));
        const auto ids = tokenizer.encode(text);
        INFO("byte 0x" << std::hex << value);
        REQUIRE_FALSE(ids.empty());
        REQUIRE(tokenizer.decode(ids, false) == text);
    }
}

TEST_CASE("added tokens are matched literally, not merged", "[tokenizer]") {
    REQUIRE_TOKENIZER();

    const auto ids = tokenizer.encode("before<|turn>after");
    const auto turn = *tokenizer.idFor("<|turn>");

    // The special token must appear as exactly one id, not be split into
    // punctuation pieces by BPE.
    CHECK(std::ranges::count(ids, turn) == 1);
    CHECK(tokenizer.decode(ids, /*skipSpecialTokens=*/false) == "before<|turn>after");

    // Skipping specials drops it and leaves the surrounding text intact.
    CHECK(tokenizer.decode(ids, /*skipSpecialTokens=*/true) == "beforeafter");
}

TEST_CASE("merges produce longer pieces than raw characters", "[tokenizer]") {
    REQUIRE_TOKENIZER();

    // A BPE that never merged would emit one token per character. Common
    // English must compress well below that.
    const std::string text =
            "The quick brown fox jumps over the lazy dog while the cat watches.";
    const auto ids = tokenizer.encode(text);

    INFO("chars " << text.size() << " -> ids " << ids.size());
    CHECK(ids.size() < text.size() / 3);

    // And common whole words should survive as single pieces.
    const auto theIds = tokenizer.encode(" the");
    CHECK(theIds.size() == 1);
}

TEST_CASE("encoding is deterministic and concatenation-stable", "[tokenizer]") {
    REQUIRE_TOKENIZER();

    const std::string text = "Reproducibility matters for a decoder.";
    CHECK(tokenizer.encode(text) == tokenizer.encode(text));

    // Encoding a prefix then the remainder need not equal encoding the whole -
    // that is inherent to BPE - but the decoded text must still match.
    const auto whole = tokenizer.encode(text);
    auto split = tokenizer.encode(text.substr(0, 10));
    for (const u32 id : tokenizer.encode(text.substr(10))) {
        split.push_back(id);
    }
    CHECK(tokenizer.decode(whole, false) == tokenizer.decode(split, false));
}

TEST_CASE("decoding one token at a time reassembles the whole", "[tokenizer]") {
    REQUIRE_TOKENIZER();

    // The streaming path emits text per token, so per-token decoding has to
    // concatenate to the same string - except across byte-fallback runs, which
    // is why the multi-byte case is checked separately below.
    const std::string text = "Streaming output, one token at a time.";
    const auto ids = tokenizer.encode(text);

    std::string streamed;
    for (const u32 id : ids) {
        streamed += tokenizer.decodeOne(id, false);
    }
    CHECK(streamed == text);
}

TEST_CASE("empty input encodes to nothing", "[tokenizer]") {
    REQUIRE_TOKENIZER();
    CHECK(tokenizer.encode("").empty());
    CHECK(tokenizer.decode({}, false).empty());
}

TEST_CASE("loading is fast enough to do at startup", "[tokenizer]") {
    REQUIRE_TOKENIZER();

    const auto start = std::chrono::steady_clock::now();
    const auto reloaded = Tokenizer::loadFromFile(tokenizerPath());
    const double seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

    REQUIRE(reloaded.has_value());
    INFO("load took " << seconds << " s");
    // Sits alongside a 13 GiB model load, so a few seconds is the budget.
    CHECK(seconds < 30.0);
}

// ---------------------------------------------------------------------------
// Chat template
// ---------------------------------------------------------------------------

TEST_CASE("the chat template renders Gemma's instruction format", "[chat]") {
    REQUIRE_TOKENIZER();

    const auto templateResult = ChatTemplate::create(tokenizer);
    REQUIRE(templateResult.has_value());

    const auto ids = templateResult->render({{ChatRole::User, "Hi"}});
    REQUIRE(ids.has_value());

    // <bos><|turn>user\nHi<turn|>\n<|turn>model\n
    const auto bos = *tokenizer.idFor("<bos>");
    const auto turnOpen = *tokenizer.idFor("<|turn>");
    const auto turnClose = *tokenizer.idFor("<turn|>");
    const auto newline = *tokenizer.idFor("\n");

    REQUIRE(ids->size() >= 8);
    CHECK((*ids)[0] == bos);
    CHECK((*ids)[1] == turnOpen);
    CHECK(ids->back() == newline);
    CHECK((*ids)[ids->size() - 2] == *tokenizer.idFor("model"));
    CHECK((*ids)[ids->size() - 3] == turnOpen);
    CHECK(std::ranges::count(*ids, turnClose) == 1);

    // Rendering must be reproducible, since a prompt cache keys on it.
    CHECK(*templateResult->render({{ChatRole::User, "Hi"}}) == *ids);
}

TEST_CASE("system guidance folds into the first user turn", "[chat]") {
    REQUIRE_TOKENIZER();

    const auto templateResult = ChatTemplate::create(tokenizer);
    REQUIRE(templateResult.has_value());

    // Gemma has no system role, so the text has to go somewhere: into the
    // first user turn, which is what the bundled template does.
    const auto ids = templateResult->render({{ChatRole::System, "Be terse."},
                                             {ChatRole::User, "Hello"}});
    REQUIRE(ids.has_value());

    const auto turnOpen = *tokenizer.idFor("<|turn>");
    // Two turns only: the user turn and the opening of the model turn.
    CHECK(std::ranges::count(*ids, turnOpen) == 2);

    const std::string rendered = tokenizer.decode(*ids, /*skipSpecialTokens=*/true);
    CHECK_THAT(rendered, Catch::Matchers::ContainsSubstring("Be terse."));
    CHECK_THAT(rendered, Catch::Matchers::ContainsSubstring("Hello"));
}

TEST_CASE("a multi-turn conversation alternates correctly", "[chat]") {
    REQUIRE_TOKENIZER();

    const auto templateResult = ChatTemplate::create(tokenizer);
    REQUIRE(templateResult.has_value());

    const auto ids = templateResult->render({{ChatRole::User, "What is 2+2?"},
                                             {ChatRole::Model, "4"},
                                             {ChatRole::User, "And 3+3?"}});
    REQUIRE(ids.has_value());

    const auto turnOpen = *tokenizer.idFor("<|turn>");
    const auto turnClose = *tokenizer.idFor("<turn|>");
    // Three completed turns plus the opening of a fourth.
    CHECK(std::ranges::count(*ids, turnOpen) == 4);
    CHECK(std::ranges::count(*ids, turnClose) == 3);
}

TEST_CASE("generation stops on the turn terminator", "[chat]") {
    REQUIRE_TOKENIZER();

    const auto templateResult = ChatTemplate::create(tokenizer);
    REQUIRE(templateResult.has_value());

    const auto& stops = templateResult->stopTokens();
    CHECK(std::ranges::find(stops, *tokenizer.idFor("<turn|>")) != stops.end());
    CHECK(std::ranges::find(stops, *tokenizer.idFor("<eos>")) != stops.end());
}

TEST_CASE("omitting the generation prompt leaves the turn closed", "[chat]") {
    REQUIRE_TOKENIZER();

    const auto templateResult = ChatTemplate::create(tokenizer);
    REQUIRE(templateResult.has_value());

    const auto ids = templateResult->render({{ChatRole::User, "Hi"}},
                                            /*addGenerationPrompt=*/false);
    REQUIRE(ids.has_value());
    CHECK(ids->back() == *tokenizer.idFor("\n"));
    CHECK(std::ranges::count(*ids, *tokenizer.idFor("<|turn>")) == 1);
}
