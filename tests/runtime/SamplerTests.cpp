// Sampling and stop matching.
//
// Both are pure host logic, so these run without a GPU or an install. They are
// also where a subtle bug is least likely to be noticed: a sampler that quietly
// ignores top-p still produces fluent text, and a stop matcher that misses a
// sequence spanning two tokens only fails on some prompts.

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <map>
#include <string>
#include <vector>

#include "tf/runtime/Sampler.h"
#include "tf/runtime/StopMatcher.h"

using namespace tf;
using namespace tf::runtime;

namespace {

/// Logits that put a known ordering on a small vocabulary.
[[nodiscard]] std::vector<float> ramp(u32 count) {
    std::vector<float> logits(count);
    for (u32 i = 0; i < count; ++i) {
        // Descending, so token 0 is the most likely.
        logits[i] = static_cast<float>(count - i);
    }
    return logits;
}

/// Runs the sampler many times and reports how often each token came up.
[[nodiscard]] std::map<u32, u32> histogram(SamplingParams params, std::span<const float> logits,
                                           u32 draws) {
    Sampler sampler{params};
    std::map<u32, u32> counts;
    for (u32 i = 0; i < draws; ++i) {
        std::vector<float> copy{logits.begin(), logits.end()};
        ++counts[sampler.sample(copy, {})];
    }
    return counts;
}

}  // namespace

// ---------------------------------------------------------------------------
// Sampler
// ---------------------------------------------------------------------------

TEST_CASE("temperature zero is greedy", "[sampler]") {
    SamplingParams params{.temperature = 0.0f};
    REQUIRE(params.isGreedy());

    std::vector<float> logits{1.0f, 5.0f, 3.0f, 5.0f};
    Sampler sampler{params};
    // Ties go to the lowest index, matching the GPU argmax kernel. The two
    // paths pick between them by whether the logits were downloaded, so they
    // must not disagree.
    CHECK(sampler.sample(logits, {}) == 1);
}

TEST_CASE("greedy ignores the distribution entirely", "[sampler]") {
    const auto logits = ramp(1000);
    const auto counts = histogram(SamplingParams{.temperature = 0.0f}, logits, 20);
    REQUIRE(counts.size() == 1);
    CHECK(counts.begin()->first == 0);
}

TEST_CASE("top-k bounds which tokens can appear", "[sampler]") {
    const auto logits = ramp(1000);

    // A high temperature would otherwise spread the mass across the whole
    // vocabulary, so anything above index 4 appearing means top-k did nothing.
    const auto counts = histogram(
            SamplingParams{.temperature = 5.0f, .topK = 5, .topP = 1.0f, .seed = 7}, logits,
            2000);

    REQUIRE_FALSE(counts.empty());
    CHECK(counts.rbegin()->first < 5);
    // And it really is sampling, not collapsing to the top token.
    CHECK(counts.size() > 1);
}

TEST_CASE("top-p bounds the mass, not the count", "[sampler]") {
    // Token 0 alone holds well over half the probability at this temperature.
    std::vector<float> logits{10.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};

    const auto counts = histogram(
            SamplingParams{.temperature = 1.0f, .topK = 0, .topP = 0.5f, .seed = 11}, logits,
            500);

    // With the nucleus at 0.5 and the leader above it, only the leader
    // survives - even though top-k is disabled and seven other tokens exist.
    REQUIRE(counts.size() == 1);
    CHECK(counts.begin()->first == 0);
}

TEST_CASE("top-p always keeps at least one token", "[sampler]") {
    const auto logits = ramp(50);
    // A nucleus below the leading token's probability must not empty the
    // candidate set; the prefix is inclusive of the token that crosses it.
    const auto counts = histogram(
            SamplingParams{.temperature = 1.0f, .topK = 0, .topP = 0.0001f, .seed = 3}, logits,
            50);
    REQUIRE(counts.size() == 1);
    CHECK(counts.begin()->first == 0);
}

TEST_CASE("a seed makes a run reproducible", "[sampler]") {
    const auto logits = ramp(200);
    const SamplingParams params{.temperature = 1.5f, .topK = 40, .topP = 0.95f, .seed = 4242};

    std::vector<u32> first;
    std::vector<u32> second;
    {
        Sampler sampler{params};
        for (u32 i = 0; i < 40; ++i) {
            std::vector<float> copy = logits;
            first.push_back(sampler.sample(copy, {}));
        }
    }
    {
        Sampler sampler{params};
        for (u32 i = 0; i < 40; ++i) {
            std::vector<float> copy = logits;
            second.push_back(sampler.sample(copy, {}));
        }
    }
    CHECK(first == second);
    // And a different seed does not produce the same sequence, or the seed is
    // not reaching the generator.
    Sampler other{SamplingParams{.temperature = 1.5f, .topK = 40, .topP = 0.95f, .seed = 99}};
    std::vector<u32> third;
    for (u32 i = 0; i < 40; ++i) {
        std::vector<float> copy = logits;
        third.push_back(other.sample(copy, {}));
    }
    CHECK(third != first);
}

TEST_CASE("an unseeded sampler still reports the seed it used", "[sampler]") {
    Sampler sampler{SamplingParams{.temperature = 1.0f, .seed = 0}};
    // Zero would make the run unrepeatable even in principle, so the resolved
    // seed is what gets reported.
    CHECK(sampler.seed() != 0);
}

TEST_CASE("the repetition penalty pushes both positive and negative logits down",
          "[sampler]") {
    // The classic bug: dividing by the penalty lowers a positive logit but
    // *raises* a negative one, making an unlikely repeated token more likely.
    std::vector<float> logits{2.0f, -2.0f, 0.5f};
    const std::vector<u32> history{0, 1};

    Sampler sampler{SamplingParams{.temperature = 0.0f, .repetitionPenalty = 2.0f}};
    static_cast<void>(sampler.sample(logits, history));

    CHECK(logits[0] == 1.0f);   // positive: divided
    CHECK(logits[1] == -4.0f);  // negative: multiplied, so still pushed down
    CHECK(logits[2] == 0.5f);   // untouched
}

TEST_CASE("the repetition penalty only looks back over its window", "[sampler]") {
    std::vector<float> logits{2.0f, 2.0f, 2.0f};
    const std::vector<u32> history{0, 1, 2};

    Sampler sampler{
            SamplingParams{.temperature = 0.0f, .repetitionPenalty = 2.0f,
                           .repetitionWindow = 2}};
    static_cast<void>(sampler.sample(logits, history));

    CHECK(logits[0] == 2.0f);  // outside the window
    CHECK(logits[1] == 1.0f);
    CHECK(logits[2] == 1.0f);
}

TEST_CASE("sampling survives the softcapped logit range at a low temperature",
          "[sampler]") {
    // Logits arrive softcapped to +-30. At temperature 0.05 that is a division
    // by 20, which overflows expf unless the maximum is subtracted first.
    std::vector<float> logits{30.0f, -30.0f, 29.0f, 0.0f};
    Sampler sampler{SamplingParams{.temperature = 0.05f, .topK = 0, .topP = 1.0f, .seed = 5}};

    const u32 chosen = sampler.sample(logits, {});
    CHECK(chosen < 4);
}

TEST_CASE("sampling parameters are validated", "[sampler]") {
    CHECK(SamplingParams{}.validate().has_value());
    CHECK_FALSE(SamplingParams{.temperature = -1.0f}.validate().has_value());
    CHECK_FALSE(SamplingParams{.topP = 0.0f}.validate().has_value());
    CHECK_FALSE(SamplingParams{.repetitionPenalty = 0.0f}.validate().has_value());
    // Zero top-k means "disabled", not an error.
    CHECK(SamplingParams{.topK = 0}.validate().has_value());
}

// ---------------------------------------------------------------------------
// StopMatcher
// ---------------------------------------------------------------------------

TEST_CASE("no stop strings emits everything immediately", "[stop]") {
    StopMatcher matcher;
    REQUIRE(matcher.empty());

    const auto update = matcher.push("hello world");
    CHECK(update.emit == "hello world");
    CHECK_FALSE(update.stopped);
    CHECK(matcher.flush().empty());
}

TEST_CASE("a stop string inside one piece truncates it", "[stop]") {
    StopMatcher matcher{{"STOP"}};

    const auto update = matcher.push("before STOP after");
    CHECK(update.emit == "before ");
    CHECK(update.stopped);
}

TEST_CASE("a stop string spanning pieces is still caught", "[stop]") {
    StopMatcher matcher{{"\n\nUser:"}};

    // This is the case that motivates the whole class: the sequence arrives as
    // three tokens and no single piece contains it.
    auto first = matcher.push("answer.\n\n");
    CHECK(first.emit == "answer.");
    CHECK_FALSE(first.stopped);

    auto second = matcher.push("User");
    CHECK(second.emit.empty());
    CHECK_FALSE(second.stopped);

    auto third = matcher.push(":");
    CHECK(third.emit.empty());
    CHECK(third.stopped);
}

TEST_CASE("text that only looked like a stop string is released", "[stop]") {
    StopMatcher matcher{{"\n\nUser:"}};

    auto held = matcher.push("done.\n\nUse");
    // "\n\nUse" is a prefix of the stop string, so it must not be shown yet.
    CHECK(held.emit == "done.");
    CHECK_FALSE(held.stopped);

    auto released = matcher.push("ful notes follow");
    // It turned out to be "Useful", so the held text belongs to the answer.
    CHECK(released.emit == "\n\nUseful notes follow");
    CHECK_FALSE(released.stopped);
}

TEST_CASE("held-back text is flushed when generation ends without a stop", "[stop]") {
    StopMatcher matcher{{"<<END>>"}};

    auto update = matcher.push("trailing <<EN");
    CHECK(update.emit == "trailing ");
    CHECK_FALSE(update.stopped);

    // A token limit ends the run; the partial match was ordinary text.
    CHECK(matcher.flush() == "<<EN");
    CHECK(matcher.flush().empty());
}

TEST_CASE("the earliest of several stop strings wins", "[stop]") {
    StopMatcher matcher{{"BBB", "A"}};

    const auto update = matcher.push("xxAyyBBB");
    CHECK(update.emit == "xx");
    CHECK(update.stopped);
}

TEST_CASE("empty stop strings are ignored rather than stopping immediately",
          "[stop]") {
    // An empty stop string is a prefix of everything; taken literally it would
    // end generation before the first token.
    StopMatcher matcher{{"", "END"}};
    REQUIRE_FALSE(matcher.empty());

    const auto update = matcher.push("some text");
    CHECK(update.emit == "some text");
    CHECK_FALSE(update.stopped);
}

TEST_CASE("a stop string arriving one character at a time is caught", "[stop]") {
    StopMatcher matcher{{"abc"}};

    std::string emitted;
    bool stopped = false;
    for (const char c : std::string{"zzabc"}) {
        const auto update = matcher.push(std::string_view{&c, 1});
        emitted += update.emit;
        if (update.stopped) {
            stopped = true;
            break;
        }
    }
    CHECK(emitted == "zz");
    CHECK(stopped);
}
