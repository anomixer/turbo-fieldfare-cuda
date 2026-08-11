// Command-line parsing.
//
// Argument handling is where a CLI accumulates quiet bugs: a missing value read
// as zero, a typo'd flag silently ignored, a number parsed out of "12abc". Args
// is a pure value with no side effects precisely so this can be checked here
// rather than by running the binary.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <string>
#include <string_view>
#include <vector>

#include "Args.h"

using namespace tf;
using namespace tf::cli;

namespace {

[[nodiscard]] Result<Args> parse(std::vector<std::string_view> arguments) {
    return Args::parse(arguments);
}

/// The smallest argument list that validates, so a test can add one thing to it
/// and check that one thing.
[[nodiscard]] std::vector<std::string_view> minimal() {
    return {"--model", "C:\\models\\gemma4.gturbo", "--prompt", "hello"};
}

}  // namespace

TEST_CASE("a minimal command line parses and validates", "[cli]") {
    auto args = parse(minimal());
    REQUIRE(args.has_value());
    CHECK(args->prompt == "hello");
    CHECK(args->chat);
    CHECK(args->validate().has_value());
}

TEST_CASE("defaults match the documented sampling settings", "[cli]") {
    auto args = parse(minimal());
    REQUIRE(args.has_value());
    CHECK(args->sampling.temperature == 0.2f);
    CHECK(args->sampling.topK == 64);
    CHECK(args->sampling.topP == 0.95f);
    CHECK(args->maxTokens == 512);
    CHECK(args->contextLength == 4096);
    CHECK(args->prefillChunk == 128);
}

TEST_CASE("help and version short-circuit the rest of the line", "[cli]") {
    // Neither should need a model, and neither should trip over what follows.
    auto help = parse({"--help", "--model"});
    REQUIRE(help.has_value());
    CHECK(help->showHelp);
    CHECK(help->validate().has_value());

    auto version = parse({"--version"});
    REQUIRE(version.has_value());
    CHECK(version->showVersion);
    CHECK(version->validate().has_value());
}

TEST_CASE("an unknown flag is an error, not silence", "[cli]") {
    auto args = parse({"--model", "x", "--prompt", "y", "--tempurature", "0.5"});
    REQUIRE_FALSE(args.has_value());
    CHECK(args.error().code() == ErrorCode::InvalidArgument);
    CHECK_THAT(args.error().message(), Catch::Matchers::ContainsSubstring("tempurature"));
}

TEST_CASE("a flag missing its value is an error, not a zero", "[cli]") {
    auto args = parse({"--model", "x", "--prompt", "y", "--tokens"});
    REQUIRE_FALSE(args.has_value());
    CHECK_THAT(args.error().message(), Catch::Matchers::ContainsSubstring("needs a value"));
}

TEST_CASE("numbers with trailing characters are rejected", "[cli]") {
    // strtoull would read this as 12, which is worse than refusing it: the user
    // asked for something the parser did not understand.
    auto tokens = parse({"--model", "x", "--prompt", "y", "--tokens", "12abc"});
    REQUIRE_FALSE(tokens.has_value());
    CHECK_THAT(tokens.error().message(), Catch::Matchers::ContainsSubstring("whole number"));

    auto temperature = parse({"--model", "x", "--prompt", "y", "--temperature", "0.5x"});
    REQUIRE_FALSE(temperature.has_value());
    CHECK_THAT(temperature.error().message(), Catch::Matchers::ContainsSubstring("number"));

    // And a value that is not a number at all does not become zero.
    auto nonsense = parse({"--model", "x", "--prompt", "y", "--tokens", "lots"});
    REQUIRE_FALSE(nonsense.has_value());
}

TEST_CASE("stop strings accumulate and understand escapes", "[cli]") {
    auto args = parse({"--model", "x", "--prompt", "y", "--stop", "\\n\\nUser:", "--stop",
                       "<<END>>"});
    REQUIRE(args.has_value());
    REQUIRE(args->stopStrings.size() == 2);
    // A shell will not turn \n into a newline, so the parser does.
    CHECK(args->stopStrings[0] == "\n\nUser:");
    CHECK(args->stopStrings[1] == "<<END>>");
}

TEST_CASE("a backslash that is not an escape is left alone", "[cli]") {
    auto args = parse({"--model", "x", "--prompt", "y", "--stop", "C:\\path\\x"});
    REQUIRE(args.has_value());
    // \p and \x are not escapes, so they survive; \\ collapses to one.
    CHECK(args->stopStrings[0] == "C:\\path\\x");
}

TEST_CASE("--raw turns off the instruction template", "[cli]") {
    auto args = parse({"--model", "x", "--prompt", "y", "--raw"});
    REQUIRE(args.has_value());
    CHECK_FALSE(args->chat);
    CHECK(args->validate().has_value());
}

TEST_CASE("a prompt is required, from one source only", "[cli]") {
    auto none = parse({"--model", "x"});
    REQUIRE(none.has_value());
    auto valid = none->validate();
    REQUIRE_FALSE(valid.has_value());
    CHECK_THAT(valid.error().message(), Catch::Matchers::ContainsSubstring("--stdin"));

    auto both = parse({"--model", "x", "--prompt", "y", "--stdin"});
    REQUIRE(both.has_value());
    CHECK_FALSE(both->validate().has_value());

    auto stdinOnly = parse({"--model", "x", "--stdin"});
    REQUIRE(stdinOnly.has_value());
    CHECK(stdinOnly->validate().has_value());
}

TEST_CASE("contradictory or impossible combinations are refused", "[cli]") {
    const auto rejects = [](std::vector<std::string_view> arguments) {
        auto args = parse(std::move(arguments));
        REQUIRE(args.has_value());
        return !args->validate().has_value();
    };

    CHECK(rejects({"--model", "x", "--prompt", "y", "--verbose", "--quiet"}));
    CHECK(rejects({"--model", "x", "--prompt", "y", "--tokens", "0"}));
    CHECK(rejects({"--model", "x", "--prompt", "y", "--context", "0"}));
    CHECK(rejects({"--model", "x", "--prompt", "y", "--prefill-chunk", "0"}));
    // A chunk larger than the context can never be filled.
    CHECK(rejects({"--model", "x", "--prompt", "y", "--context", "64", "--prefill-chunk",
                   "128"}));
    // A system message needs the template that has somewhere to put it.
    CHECK(rejects({"--model", "x", "--prompt", "y", "--raw", "--system", "be brief"}));
    // Sampling validation is reached through Args::validate.
    CHECK(rejects({"--model", "x", "--prompt", "y", "--temperature", "-1"}));
    CHECK(rejects({"--model", "x", "--prompt", "y", "--top-p", "0"}));
}

TEST_CASE("a missing model is caught at validation, not at load", "[cli]") {
    // Only when the environment does not supply one; the parser reads
    // TF_GTURBO_DIR, so this asserts on the message rather than the outcome.
    auto args = parse({"--prompt", "y"});
    REQUIRE(args.has_value());
    if (args->modelDir.empty()) {
        auto valid = args->validate();
        REQUIRE_FALSE(valid.has_value());
        CHECK_THAT(valid.error().message(), Catch::Matchers::ContainsSubstring("--model"));
    }
}

TEST_CASE("the VRAM budget is read in gibibytes", "[cli]") {
    auto args = parse({"--model", "x", "--prompt", "y", "--vram-budget", "6.5"});
    REQUIRE(args.has_value());
    CHECK(args->vramBudget == static_cast<u64>(6.5 * 1024.0 * 1024.0 * 1024.0));
}

TEST_CASE("usage text mentions every flag the parser accepts", "[cli]") {
    // Cheap guard against a flag being added and never documented.
    const std::string text = usage();
    for (const std::string_view flag :
         {"--model", "--prompt", "--stdin", "--system", "--raw", "--temperature", "--top-k",
          "--top-p", "--repeat-penalty", "--repeat-window", "--seed", "--tokens", "--stop",
          "--context", "--vram-budget", "--expert-slots", "--prefill-chunk", "--verbose",
          "--quiet", "--dry-run", "--version", "--help"}) {
        INFO("flag " << flag);
        CHECK(text.find(flag) != std::string::npos);
    }
}
