// tf-generate - runs the decode path against a real install.
//
// A diagnostic front end, superseded by the real CLI in M9. Uses the M2 BPE
// tokenizer and chat template.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

#include "tf/core/io/File.h"
#include "tf/core/json/Json.h"
#include "tf/core/tokenizer/ChatTemplate.h"
#include "tf/core/tokenizer/Tokenizer.h"
#include "tf/gpu/Backend.h"
#include "tf/runtime/ForwardRunner.h"
#include "tf/runtime/KVCache.h"
#include "tf/runtime/Model.h"

using namespace tf;

namespace {

void printUsage() {
    std::puts(R"(tf-generate - generate tokens from a .gturbo install

Usage:
  tf-generate --model <dir.gturbo> [options]

Options:
  --model <dir>          Install to load (or TF_GTURBO_DIR)
  --prompt <text>        Prompt text, encoded with the BPE tokenizer
  --chat                 Wrap the prompt in Gemma's instruction template
  --tokens <n>           Tokens to generate, default 32
  --start <id>           Starting token id, default the model's BOS
  --context <n>          KV context length, default 4096
  --vram-budget <GiB>    Constrain the residency planner below the hardware.
                         Use this to exercise the streamed path on a card that
                         would otherwise hold the whole model.
  --expert-slots <n>     Slots per streamed layer, default 16
  -h, --help             This message)");
}

std::string gib(u64 bytes) {
    return std::format("{:.2f} GiB", static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0));
}

}  // namespace

int main(int argc, char** argv) {
    std::filesystem::path installDir;
    if (const char* env = std::getenv("TF_GTURBO_DIR")) {
        installDir = env;
    }

    u32 tokensToGenerate = 32;
    u32 startToken = 0;
    bool startTokenSet = false;
    u64 contextLength = 4096;
    u64 vramBudget = 0;
    u32 expertSlots = 16;
    u32 prefillChunk = 128;
    bool debugRms = false;
    u64 layerLimit = runtime::ForwardRunner::kNoLayerLimit;
    std::string prompt;
    bool chatTemplate = false;

    for (int i = 1; i < argc; ++i) {
        const std::string_view flag = argv[i];
        const auto value = [&]() -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "%.*s needs a value\n", static_cast<int>(flag.size()),
                             flag.data());
                std::exit(2);
            }
            return argv[++i];
        };

        if (flag == "-h" || flag == "--help") {
            printUsage();
            return 0;
        }
        if (flag == "--model") {
            installDir = value();
        } else if (flag == "--tokens") {
            tokensToGenerate = static_cast<u32>(std::strtoul(value(), nullptr, 10));
        } else if (flag == "--start") {
            startToken = static_cast<u32>(std::strtoul(value(), nullptr, 10));
            startTokenSet = true;
        } else if (flag == "--context") {
            contextLength = std::strtoull(value(), nullptr, 10);
        } else if (flag == "--vram-budget") {
            vramBudget = static_cast<u64>(std::strtod(value(), nullptr) * 1024.0 * 1024.0 *
                                          1024.0);
        } else if (flag == "--expert-slots") {
            expertSlots = static_cast<u32>(std::strtoul(value(), nullptr, 10));
        } else if (flag == "--prefill-chunk") {
            prefillChunk = static_cast<u32>(std::strtoul(value(), nullptr, 10));
        } else if (flag == "--prompt") {
            prompt = value();
        } else if (flag == "--chat") {
            chatTemplate = true;
        } else if (flag == "--layers") {
            layerLimit = std::strtoull(value(), nullptr, 10);
        } else if (flag == "--debug-rms") {
            debugRms = true;
        } else {
            std::fprintf(stderr, "unknown argument '%.*s'\n\n", static_cast<int>(flag.size()),
                         flag.data());
            printUsage();
            return 2;
        }
    }

    if (installDir.empty()) {
        std::fprintf(stderr, "--model is required\n\n");
        printUsage();
        return 2;
    }

    const auto backends = gpu::compiledBackends();
    if (backends.empty()) {
        std::fprintf(stderr, "this build has no GPU backend\n");
        return 1;
    }
    auto backend = gpu::createBackend(backends.front());
    if (!backend) {
        std::fprintf(stderr, "GPU unavailable: %s\n", backend.error().toString().c_str());
        return 1;
    }
    std::printf("Device: %s\n", (*backend)->info().name.c_str());

    // ---- Load ------------------------------------------------------------
    runtime::Model::LoadOptions options;
    options.budget.deviceBytes = vramBudget;
    options.budget.contextLength = contextLength;
    options.budget.slotsPerStreamedLayer = expertSlots;
    // The planner has to reserve for the prefill scratch before it decides how
    // many layers stay resident; the runner allocating it comes later.
    options.budget.maxPrefillTokens = prefillChunk;

    const auto loadStart = std::chrono::steady_clock::now();
    auto model = runtime::Model::load(
            **backend, installDir, options,
            [](std::string_view stage, u64 done, u64 total) {
                if (total > 0) {
                    std::printf("\r  loading %-18.*s %5.1f%%", static_cast<int>(stage.size()),
                                stage.data(),
                                100.0 * static_cast<double>(done) /
                                        static_cast<double>(total));
                    std::fflush(stdout);
                }
                return true;
            });
    if (!model) {
        std::fprintf(stderr, "\nload failed: %s\n", model.error().toString().c_str());
        return 1;
    }
    const double loadSeconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - loadStart).count();
    std::printf("\r%-50s\r", "");

    std::printf("\nResidency plan\n%s\n", model->residency().describe().c_str());
    std::printf("Loaded %s of device memory in %.1fs\n", gib(model->deviceBytesUsed()).c_str(),
                loadSeconds);

    auto cache = runtime::KVCacheManager::create(**backend, model->arch(), contextLength,
                                                prefillChunk);
    if (!cache) {
        std::fprintf(stderr, "KV cache: %s\n", cache.error().toString().c_str());
        return 1;
    }

    auto runner = runtime::ForwardRunner::create(**backend, *model, *cache, prefillChunk);
    if (!runner) {
        std::fprintf(stderr, "runner: %s\n", runner.error().toString().c_str());
        return 1;
    }

    std::printf("Prefill scratch: %s at a %u token chunk\n\n",
                gib(runner->scratchBytes()).c_str(), prefillChunk);

    runner->setLayerLimit(layerLimit);

    if (debugRms) {
        // Localizes a layer that saturates, collapses, or goes non-finite.
        runner->setLayerObserver([](u64 layer, std::span<const float> hidden) {
            double sumSquares = 0.0;
            float lowest = hidden.empty() ? 0.0f : hidden[0];
            float highest = lowest;
            u64 nonFinite = 0;
            for (const float value : hidden) {
                if (!std::isfinite(value)) {
                    ++nonFinite;
                    continue;
                }
                sumSquares += static_cast<double>(value) * value;
                lowest = std::min(lowest, value);
                highest = std::max(highest, value);
            }
            const double rms = std::sqrt(sumSquares / static_cast<double>(hidden.size()));
            std::printf("  layer %2llu  rms %10.4f  min %10.4f  max %10.4f%s\n",
                        static_cast<unsigned long long>(layer), rms, lowest, highest,
                        nonFinite > 0 ? "  <- NON-FINITE" : "");
        });
    }

    // The real BPE tokenizer, replacing the greedy longest-match placeholder
    // this tool used before M2.
    auto tokenizerResult =
            Tokenizer::loadFromFile(installDir / "tokenizer" / "tokenizer.json");
    if (!tokenizerResult) {
        std::fprintf(stderr, "tokenizer: %s\n", tokenizerResult.error().toString().c_str());
        return 1;
    }
    const Tokenizer& tokenizer = *tokenizerResult;

    // ---- Prompt ----------------------------------------------------------
    std::vector<u32> promptIds;
    std::vector<u32> stopTokens{model->arch().eosTokenIds.begin(),
                                model->arch().eosTokenIds.end()};

    if (!prompt.empty()) {
        if (chatTemplate) {
            auto templateResult = ChatTemplate::create(tokenizer);
            if (!templateResult) {
                std::fprintf(stderr, "chat template: %s\n",
                             templateResult.error().toString().c_str());
                return 1;
            }
            auto rendered = templateResult->render({{ChatRole::User, prompt}});
            if (!rendered) {
                std::fprintf(stderr, "render: %s\n", rendered.error().toString().c_str());
                return 1;
            }
            promptIds = std::move(*rendered);
            stopTokens = templateResult->stopTokens();
        } else {
            promptIds = tokenizer.encode(prompt);
            promptIds.insert(promptIds.begin(), model->arch().bosTokenId);
        }

        std::printf("Prompt: \"%s\" -> %zu tokens\n  ", prompt.c_str(), promptIds.size());
        for (const u32 id : promptIds) {
            std::printf("[%u:%s] ", id, tokenizer.decodeOne(id, false).c_str());
        }
        std::printf("\n");
    }

    // ---- Generate --------------------------------------------------------
    if (!startTokenSet) {
        startToken = model->arch().bosTokenId;
    }

    u32 current = promptIds.empty() ? startToken : promptIds.front();
    std::string text;

    // Prefill in chunks. The last prompt token is left for the decode loop,
    // which needs a token to step on; every earlier one goes through the batched
    // path, where the weights are read once per chunk instead of once per token.
    //
    // No head runs here at all: it multiplies against the whole tied embedding
    // table, 396 MiB of the resident set, and prompt tokens produce logits
    // nobody reads.
    const auto prefillStart = std::chrono::steady_clock::now();
    if (promptIds.size() > 1) {
        const usize toPrefill = promptIds.size() - 1;
        for (usize offset = 0; offset < toPrefill; offset += prefillChunk) {
            const usize count = std::min(static_cast<usize>(prefillChunk), toPrefill - offset);
            const auto status = runner->prefillChunk(
                    std::span<const u32>{promptIds}.subspan(offset, count), cache->position());
            if (!status) {
                std::fprintf(stderr, "prefill failed: %s\n", status.error().toString().c_str());
                return 1;
            }
            cache->advance(count);
        }
        current = promptIds.back();

        const double prefillSeconds =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - prefillStart)
                        .count();
        std::printf("Prefill: %zu tokens in %.2fs = %.1f tok/s\n", toPrefill, prefillSeconds,
                    static_cast<double>(toPrefill) / prefillSeconds);
    }

    if (!promptIds.empty()) {
        std::printf("Completion:%s", "");
    } else {
        std::printf("Generating %u tokens from id %u\n\n", tokensToGenerate, startToken);
    }

    const auto generateStart = std::chrono::steady_clock::now();
    u32 produced = 0;

    for (u32 step = 0; step < tokensToGenerate; ++step) {
        const auto status = runner->decodeStep(current, cache->position());
        if (!status) {
            std::fprintf(stderr, "\ndecode failed at step %u: %s\n", step,
                         status.error().toString().c_str());
            return 1;
        }
        cache->advance(1);

        if (debugRms && step == 0) {
            if (auto values = runner->readLogits(); values.has_value()) {
                u64 nonFinite = 0;
                u64 atCap = 0;
                float lowest = (*values)[0];
                float highest = lowest;
                for (const float value : *values) {
                    if (!std::isfinite(value)) {
                        ++nonFinite;
                        continue;
                    }
                    if (std::abs(std::abs(value) - 30.0f) < 1e-3f) {
                        ++atCap;
                    }
                    lowest = std::min(lowest, value);
                    highest = std::max(highest, value);
                }
                std::printf(
                        "  logits: min %.4f max %.4f, %llu non-finite, %llu saturated at "
                        "the cap of %zu\n",
                        lowest, highest, static_cast<unsigned long long>(nonFinite),
                        static_cast<unsigned long long>(atCap), values->size());
            }
        }

        auto next = runner->greedyToken();
        if (!next) {
            std::fprintf(stderr, "\nargmax failed: %s\n", next.error().toString().c_str());
            return 1;
        }
        ++produced;

        // Streaming decode, one token at a time, keeping special markers
        // visible so the structure of the reply is observable.
        const std::string piece = tokenizer.decodeOne(*next, /*skipSpecialTokens=*/false);
        text += piece;
        std::printf("%s", piece.c_str());
        std::fflush(stdout);

        if (std::ranges::find(stopTokens, *next) != stopTokens.end()) {
            std::printf("\n\n(end of sequence)\n");
            break;
        }
        current = *next;
    }

    const double seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - generateStart)
                    .count();

    std::printf("\n\n%u tokens in %.2fs = %.2f tok/s\n", produced, seconds,
                static_cast<double>(produced) / seconds);

    const auto& stats = model->streamer().stats();
    if (stats.requests > 0) {
        std::printf("Expert cache: %.1f%% hit rate over %llu requests, %s read\n",
                    stats.hitRate() * 100.0, static_cast<unsigned long long>(stats.requests),
                    gib(stats.bytesRead).c_str());
    }
    return 0;
}
