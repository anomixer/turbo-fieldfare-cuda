// tf-cli - the command-line entry point.
//
// Thin on purpose: argument parsing lives in Args, and everything from the
// prompt onward is tf::runtime::Generator, which the server in M10 and the GUI
// in M11 use unchanged.

#include <cstdio>
#include <iostream>
#include <iterator>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "Args.h"
#include "tf/core/tokenizer/ChatTemplate.h"
#include "tf/core/tokenizer/Tokenizer.h"
#include "tf/gpu/Backend.h"
#include "tf/gpu/Preflight.h"
#include "tf/runtime/Generator.h"
#include "tf/runtime/Model.h"

using namespace tf;

namespace {

/// Diagnostics go to stderr so that `tf-cli ... > answer.txt` captures the
/// answer and nothing else.
void note(bool quiet, std::string_view text) {
    if (!quiet) {
        std::fputs(text.data(), stderr);
    }
}

template <class... Args>
void notef(bool quiet, std::format_string<Args...> format, Args&&... arguments) {
    if (!quiet) {
        const std::string text = std::format(format, std::forward<Args>(arguments)...);
        std::fwrite(text.data(), 1, text.size(), stderr);
    }
}

[[nodiscard]] std::string gib(u64 bytes) {
    return std::format("{:.2f} GiB", static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0));
}

[[nodiscard]] std::string readStdin() {
    return std::string{std::istreambuf_iterator<char>{std::cin},
                       std::istreambuf_iterator<char>{}};
}

int fail(const Error& error) {
    std::fprintf(stderr, "error: %s\n", error.toString().c_str());
    return 1;
}

}  // namespace

int main(int argc, char** argv) {
    std::vector<std::string_view> arguments;
    arguments.reserve(static_cast<usize>(argc > 0 ? argc - 1 : 0));
    for (int i = 1; i < argc; ++i) {
        arguments.emplace_back(argv[i]);
    }

    auto parsed = cli::Args::parse(arguments);
    if (!parsed) {
        std::fprintf(stderr, "error: %s\n\n%s\n", parsed.error().toString().c_str(),
                     cli::usage().c_str());
        return 2;
    }
    cli::Args& args = *parsed;

    if (args.showHelp) {
        std::puts(cli::usage().c_str());
        return 0;
    }
    if (args.showVersion) {
        std::printf("tf-cli, TurboFieldfare for Windows\nbuilt %s %s\n", __DATE__, __TIME__);
        return 0;
    }
    if (const auto valid = args.validate(); !valid) {
        std::fprintf(stderr, "error: %s\n\n%s\n", valid.error().toString().c_str(),
                     cli::usage().c_str());
        return 2;
    }

    if (args.promptFromStdin) {
        args.prompt = readStdin();
        if (args.prompt.empty()) {
            std::fprintf(stderr, "error: --stdin was given but no input arrived\n");
            return 2;
        }
    }

    if (args.dryRun) {
        std::printf("model          %s\n", args.modelDir.string().c_str());
        std::printf("prompt         %zu characters\n", args.prompt.size());
        std::printf("template       %s\n", args.chat ? "instruction" : "raw completion");
        std::printf("temperature    %.3f\n", static_cast<double>(args.sampling.temperature));
        std::printf("top-k / top-p  %u / %.3f\n", args.sampling.topK,
                    static_cast<double>(args.sampling.topP));
        std::printf("repeat penalty %.3f over %llu tokens\n",
                    static_cast<double>(args.sampling.repetitionPenalty),
                    static_cast<unsigned long long>(args.sampling.repetitionWindow));
        std::printf("tokens         %llu\n", static_cast<unsigned long long>(args.maxTokens));
        std::printf("context        %llu\n",
                    static_cast<unsigned long long>(args.contextLength));
        std::printf("prefill chunk  %u\n", args.prefillChunk);
        std::printf("stop strings   %zu\n", args.stopStrings.size());
        return 0;
    }

    // ---- Preflight -------------------------------------------------------
    //
    // Checked before anything is loaded, because a missing driver or toolkit
    // should produce an explanation rather than a failure deep inside the
    // backend. This is the whole reason the preflight exists.
    const gpu::PreflightReport preflight = gpu::runPreflight();
    if (!preflight.canRun()) {
        std::fprintf(stderr, "%s\n", preflight.format().c_str());
        return 1;
    }
    if (preflight.hasWarnings() && args.verbose) {
        std::fprintf(stderr, "%s\n", preflight.format().c_str());
    }

    const auto backends = gpu::compiledBackends();
    if (backends.empty()) {
        std::fprintf(stderr, "error: this build has no GPU backend\n");
        return 1;
    }

    gpu::BackendKind chosen = backends.front();
    if (!args.backend.empty()) {
        const auto matches = [&](gpu::BackendKind kind) {
            std::string name{gpu::toString(kind)};
            std::ranges::transform(name, name.begin(),
                                   [](unsigned char c) { return std::tolower(c); });
            return name == args.backend;
        };
        const auto found = std::ranges::find_if(backends, matches);
        if (found == backends.end()) {
            std::fprintf(stderr, "error: no backend called '%s' in this build\n",
                         args.backend.c_str());
            return 2;
        }
        chosen = *found;
    }

    auto backend = gpu::createBackend(chosen);
    if (!backend) {
        return fail(backend.error());
    }
    notef(args.quiet, "Device: {}\n", (*backend)->info().name);

    // ---- Load ------------------------------------------------------------
    runtime::Model::LoadOptions options;
    options.budget.deviceBytes = args.vramBudget;
    options.budget.contextLength = args.contextLength;
    if (args.expertSlots > 0) {
        options.budget.slotsPerStreamedLayer = args.expertSlots;
    }
    options.budget.maxPrefillTokens = args.prefillChunk;
    options.unbufferedReads = args.unbufferedReads;
    options.readThreads = args.readThreads;

    auto model = runtime::Model::load(
            **backend, args.modelDir, options,
            [&](std::string_view stage, u64 done, u64 total) {
                if (total > 0) {
                    notef(args.quiet, "\r  loading {:<18} {:5.1f}%", stage,
                          100.0 * static_cast<double>(done) / static_cast<double>(total));
                }
                return true;
            });
    if (!model) {
        return fail(model.error());
    }
    note(args.quiet, "\r                                        \r");

    if (args.verbose) {
        notef(args.quiet, "\nResidency plan\n{}", model->residency().describe());
        notef(args.quiet, "Loaded {} of device memory\n\n", gib(model->deviceBytesUsed()));
    }

    auto tokenizer =
            Tokenizer::loadFromFile(args.modelDir / "tokenizer" / "tokenizer.json");
    if (!tokenizer) {
        return fail(tokenizer.error());
    }

    // ---- Prompt ----------------------------------------------------------
    std::vector<u32> promptIds;
    runtime::GenerationOptions generation;
    generation.sampling = args.sampling;
    generation.maxTokens = args.maxTokens;
    generation.stopStrings = args.stopStrings;
    generation.prefillChunk = args.prefillChunk;

    if (args.chat) {
        auto chat = ChatTemplate::create(*tokenizer);
        if (!chat) {
            return fail(chat.error());
        }
        std::vector<ChatMessage> messages;
        if (!args.system.empty()) {
            messages.push_back(ChatMessage{ChatRole::System, args.system});
        }
        messages.push_back(ChatMessage{ChatRole::User, args.prompt});

        auto rendered = chat->render(messages);
        if (!rendered) {
            return fail(rendered.error());
        }
        promptIds = std::move(*rendered);
        generation.stopTokens = chat->stopTokens();
    } else {
        promptIds = tokenizer->encode(args.prompt);
        promptIds.insert(promptIds.begin(), model->arch().bosTokenId);
        generation.stopTokens.assign(model->arch().eosTokenIds.begin(),
                                     model->arch().eosTokenIds.end());
    }

    // ---- Generate --------------------------------------------------------
    auto generator = runtime::Generator::create(**backend, *model, *tokenizer,
                                                args.contextLength, args.prefillChunk);
    if (!generator) {
        return fail(generator.error());
    }

    if (args.showThinking) {
        // To stderr, so redirecting stdout still captures only the answer.
        generator->setThinkingCallback([](std::string_view piece) {
            std::fwrite(piece.data(), 1, piece.size(), stderr);
            std::fflush(stderr);
        });
    }

    const auto stats = generator->generate(promptIds, generation, [](std::string_view piece) {
        // Written and flushed as it arrives: the point of streaming is that the
        // first words appear while the rest is still being produced.
        std::fwrite(piece.data(), 1, piece.size(), stdout);
        std::fflush(stdout);
        return true;
    });
    if (!stats) {
        std::fputc('\n', stdout);
        return fail(stats.error());
    }

    std::fputc('\n', stdout);
    if (args.verbose) {
        notef(args.quiet, "\n{}\n", stats->describe());
        if (!model->residency().isFullyResident()) {
            const auto& streamer = model->streamer().stats();
            notef(args.quiet, "Expert cache: {:.1f}% hit rate over {} requests, {} read\n",
                  streamer.hitRate() * 100.0, streamer.requests, gib(streamer.bytesRead));
        }
    }
    return 0;
}
