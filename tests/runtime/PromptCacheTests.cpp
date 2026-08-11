// Reusing the KV cache across requests.
//
// A chat client resends the whole conversation every turn, so without this the
// server reprocesses the entire history to add one message. With it, the reused
// prefix must produce *exactly* what a cold run would - anything else is a
// cache that silently changes the answer, which is worse than no cache.
//
// Skipped when no install is present.

#include <catch2/catch_test_macros.hpp>

#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#include "tf/core/tokenizer/Tokenizer.h"
#include "tf/gpu/Backend.h"
#include "tf/runtime/Generator.h"
#include "tf/runtime/Model.h"

using namespace tf;

namespace {

std::filesystem::path installDir() {
    if (const char* override = std::getenv("TF_GTURBO_DIR")) {
        return std::filesystem::path{override};
    }
    if (const char* home = std::getenv("USERPROFILE")) {
        return std::filesystem::path{home} / "model-data" / "gemma4.gturbo";
    }
    return {};
}

struct Fixture {
    gpu::BackendPtr backend;
    runtime::Model model;
    Tokenizer tokenizer;
};

/// Loads the model once for the whole file: 13 GiB and several seconds each
/// time otherwise.
Fixture* sharedFixture() {
    static auto loaded = [] -> std::unique_ptr<Fixture> {
        const auto dir = installDir();
        if (dir.empty() || !std::filesystem::exists(dir / "manifest.json")) {
            return nullptr;
        }
        if (gpu::compiledBackends().empty()) {
            return nullptr;
        }
        auto backend = gpu::createBackend(gpu::compiledBackends().front());
        if (!backend) {
            return nullptr;
        }
        auto model = runtime::Model::load(
                **backend, dir,
                runtime::Model::LoadOptions{.budget = {.contextLength = 2048}});
        if (!model) {
            return nullptr;
        }
        auto tokenizer = Tokenizer::loadFromFile(dir / "tokenizer" / "tokenizer.json");
        if (!tokenizer) {
            return nullptr;
        }
        auto fixture = std::make_unique<Fixture>();
        fixture->backend = std::move(*backend);
        fixture->model = std::move(*model);
        fixture->tokenizer = std::move(*tokenizer);
        return fixture;
    }();
    return loaded.get();
}

#define REQUIRE_MODEL()                                             \
    Fixture* fixture = sharedFixture();                             \
    if (fixture == nullptr) {                                       \
        SKIP("no .gturbo install or GPU available");                \
    }

[[nodiscard]] runtime::GenerationOptions greedy(u64 tokens) {
    runtime::GenerationOptions options;
    options.sampling.temperature = 0.0f;
    options.sampling.topK = 0;
    options.maxTokens = tokens;
    return options;
}

/// Runs one prompt and returns the text.
[[nodiscard]] std::string run(runtime::Generator& generator, std::span<const u32> prompt,
                              u64 tokens, runtime::GenerationStats& stats) {
    std::string text;
    auto result = generator.generate(prompt, greedy(tokens), [&](std::string_view piece) {
        text += piece;
        return true;
    });
    REQUIRE(result.has_value());
    stats = *result;
    return text;
}

}  // namespace

TEST_CASE("a reused prefix produces the same answer as a cold run", "[promptcache]") {
    REQUIRE_MODEL();

    // Two prompts sharing a long prefix, which is what a second chat turn looks
    // like.
    const std::string shared =
            "The following is a list of facts. Paris is the capital of France. "
            "Rome is the capital of Italy. Berlin is the capital of Germany. ";
    const auto first = fixture->tokenizer.encode(shared + "What is the capital of France?");
    const auto second = fixture->tokenizer.encode(shared + "What is the capital of Italy?");

    runtime::GenerationStats coldStats;
    runtime::GenerationStats warmStats;

    // Cold: a generator that has never seen the prefix.
    std::string cold;
    {
        auto generator = runtime::Generator::create(*fixture->backend, fixture->model,
                                                    fixture->tokenizer, 2048);
        REQUIRE(generator.has_value());
        cold = run(*generator, second, 12, coldStats);
    }
    CHECK(coldStats.cachedPromptTokens == 0);

    // Warm: the same second prompt, after the first has populated the cache.
    std::string warm;
    {
        auto generator = runtime::Generator::create(*fixture->backend, fixture->model,
                                                    fixture->tokenizer, 2048);
        REQUIRE(generator.has_value());

        runtime::GenerationStats ignored;
        static_cast<void>(run(*generator, first, 8, ignored));
        warm = run(*generator, second, 12, warmStats);
    }

    // The prefix really was reused - otherwise this test proves nothing.
    INFO("reused " << warmStats.cachedPromptTokens << " of " << warmStats.promptTokens);
    CHECK(warmStats.cachedPromptTokens > 20);

    // And the answer is identical. Greedy decoding makes this exact rather than
    // approximate: the reused KV rows are the same rows a cold run would have
    // written, so any difference means the cache changed the computation.
    CHECK(warm == cold);
}

TEST_CASE("continuing a conversation reuses almost all of it", "[promptcache]") {
    REQUIRE_MODEL();

    auto generator = runtime::Generator::create(*fixture->backend, fixture->model,
                                                fixture->tokenizer, 2048);
    REQUIRE(generator.has_value());

    const auto turnOne = fixture->tokenizer.encode("Count from one to five.");
    runtime::GenerationStats firstStats;
    static_cast<void>(run(*generator, turnOne, 20, firstStats));

    // What a chat client sends next: everything so far plus a new message. The
    // cache holds the prompt and the generated reply, so only the new tokens
    // should need processing.
    std::vector<u32> turnTwo{generator->cachedTokens().begin(),
                             generator->cachedTokens().end()};
    for (const u32 id : fixture->tokenizer.encode(" Now count backwards.")) {
        turnTwo.push_back(id);
    }

    runtime::GenerationStats secondStats;
    static_cast<void>(run(*generator, turnTwo, 20, secondStats));

    INFO("reused " << secondStats.cachedPromptTokens << " of " << secondStats.promptTokens);
    // Everything but the appended message and the one token the decode loop
    // needs to step on.
    CHECK(secondStats.cachedPromptTokens >= turnTwo.size() - 8);
}

TEST_CASE("a different prompt reuses only what it shares", "[promptcache]") {
    REQUIRE_MODEL();

    auto generator = runtime::Generator::create(*fixture->backend, fixture->model,
                                                fixture->tokenizer, 2048);
    REQUIRE(generator.has_value());

    runtime::GenerationStats ignored;
    static_cast<void>(
            run(*generator, fixture->tokenizer.encode("Tell me about trees."), 8, ignored));

    // Nothing in common past the first token or two, so nothing meaningful can
    // be reused. A cache that reported otherwise would be serving rows from the
    // wrong sequence.
    runtime::GenerationStats stats;
    static_cast<void>(run(*generator,
                          fixture->tokenizer.encode("Explain quantum tunnelling briefly."), 8,
                          stats));
    INFO("reused " << stats.cachedPromptTokens);
    CHECK(stats.cachedPromptTokens < 4);
}

TEST_CASE("reset drops the prompt cache", "[promptcache]") {
    REQUIRE_MODEL();

    auto generator = runtime::Generator::create(*fixture->backend, fixture->model,
                                                fixture->tokenizer, 2048);
    REQUIRE(generator.has_value());

    const auto prompt = fixture->tokenizer.encode("Name a colour.");
    runtime::GenerationStats first;
    static_cast<void>(run(*generator, prompt, 6, first));
    CHECK_FALSE(generator->cachedTokens().empty());

    generator->reset();
    CHECK(generator->cachedTokens().empty());

    runtime::GenerationStats second;
    static_cast<void>(run(*generator, prompt, 6, second));
    // After a reset the same prompt must be reprocessed, not silently reused.
    CHECK(second.cachedPromptTokens == 0);
}
