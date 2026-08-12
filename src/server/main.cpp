// tf-server - an OpenAI-compatible endpoint for a local .gturbo install.
//
// Point any OpenAI client at http://127.0.0.1:8080/v1 and it works: the wire
// format is the compatibility surface, not a bespoke protocol.

#include <windows.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

#include "Engine.h"
#include "Http.h"
#include "tf/gpu/Preflight.h"

using namespace tf;

namespace {

struct Arguments {
    std::filesystem::path modelDir;
    std::string host = "127.0.0.1";
    u16 port = 8080;
    u64 contextLength = 4096;
    u32 prefillChunk = 128;
    u64 vramBudget = 0;
    u32 expertSlots = 16;
    std::string modelName;
    bool verbose = false;
    bool showHelp = false;
};

void printUsage() {
    std::puts(R"(tf-server - OpenAI-compatible endpoint for a local Gemma 4 install

Usage:
  tf-server --model <dir.gturbo> [options]

Then point any OpenAI client at http://127.0.0.1:8080/v1 with any API key.

Endpoints:
  POST /v1/chat/completions   streaming and non-streaming
  POST /v1/completions        raw completion, no instruction template
  GET  /v1/models
  GET  /health                counters, including prompt cache effectiveness

Options:
  --model <dir>        Install to serve, or $TF_GTURBO_DIR
  --host <addr>        Interface to bind, default 127.0.0.1. This process holds
                       a 13 GiB model and does not authenticate, so binding it
                       anywhere else should be a deliberate act.
  --port <n>           Default 8080. Zero picks a free port and prints it.
  --context <n>        KV context length, default 4096
  --model-name <name>  Reported by /v1/models, default the directory name
  --vram-budget <GiB>  Constrain residency below the hardware
  --expert-slots <n>   Slots per streamed layer, default 16
  --prefill-chunk <n>  Prompt tokens batched per pass, default 128
  -v, --verbose        Log every request
  -h, --help           This message)");
}

Result<Arguments> parseArguments(int argc, char** argv) {
    Arguments args;
    if (const char* env = std::getenv("TF_GTURBO_DIR")) {
        args.modelDir = env;
    }

    for (int i = 1; i < argc; ++i) {
        const std::string_view flag = argv[i];

        if (flag == "-h" || flag == "--help") {
            args.showHelp = true;
            return args;
        }
        if (flag == "-v" || flag == "--verbose") {
            args.verbose = true;
            continue;
        }

        if (i + 1 >= argc) {
            return makeError(ErrorCode::InvalidArgument, "{} needs a value", flag);
        }
        const std::string_view value = argv[++i];

        if (flag == "--model") {
            args.modelDir = value;
        } else if (flag == "--host") {
            args.host = value;
        } else if (flag == "--model-name") {
            args.modelName = value;
        } else if (flag == "--port") {
            args.port = static_cast<u16>(std::strtoul(std::string{value}.c_str(), nullptr, 10));
        } else if (flag == "--context") {
            args.contextLength = std::strtoull(std::string{value}.c_str(), nullptr, 10);
        } else if (flag == "--prefill-chunk") {
            args.prefillChunk =
                    static_cast<u32>(std::strtoul(std::string{value}.c_str(), nullptr, 10));
        } else if (flag == "--expert-slots") {
            args.expertSlots =
                    static_cast<u32>(std::strtoul(std::string{value}.c_str(), nullptr, 10));
        } else if (flag == "--vram-budget") {
            args.vramBudget = static_cast<u64>(
                    std::strtod(std::string{value}.c_str(), nullptr) * 1024.0 * 1024.0 * 1024.0);
        } else {
            return makeError(ErrorCode::InvalidArgument, "unknown argument '{}'", flag);
        }
    }

    if (!args.showHelp && args.modelDir.empty()) {
        return makeError(ErrorCode::InvalidArgument, "--model is required, or set TF_GTURBO_DIR");
    }
    if (args.contextLength == 0 || args.prefillChunk == 0) {
        return makeError(ErrorCode::InvalidArgument,
                         "--context and --prefill-chunk must be positive");
    }
    return args;
}

/// Set by the console handler so the accept loop can be woken from it.
server::Listener* g_listener = nullptr;
std::atomic<bool> g_stopping{false};

BOOL WINAPI consoleHandler(DWORD signal) {
    if (signal == CTRL_C_EVENT || signal == CTRL_CLOSE_EVENT || signal == CTRL_BREAK_EVENT) {
        // Only signal-safe work here: flip a flag and close the listening
        // socket, which is what breaks accept() out of its block.
        g_stopping.store(true, std::memory_order_relaxed);
        if (g_listener != nullptr) {
            g_listener->stop();
        }
        return TRUE;
    }
    return FALSE;
}

}  // namespace

int main(int argc, char** argv) {
    auto parsed = parseArguments(argc, argv);
    if (!parsed) {
        std::fprintf(stderr, "error: %s\n\n", parsed.error().toString().c_str());
        printUsage();
        return 2;
    }
    if (parsed->showHelp) {
        printUsage();
        return 0;
    }
    const Arguments& args = *parsed;

    const gpu::PreflightReport preflight = gpu::runPreflight();
    if (!preflight.canRun()) {
        std::fprintf(stderr, "%s\n", preflight.format().c_str());
        return 1;
    }

    std::printf("Loading %s\n", args.modelDir.string().c_str());
    auto engine = server::Engine::create(server::EngineOptions{
            .modelDir = args.modelDir,
            .contextLength = args.contextLength,
            .prefillChunk = args.prefillChunk,
            .vramBudget = args.vramBudget,
            .expertSlots = args.expertSlots,
            .modelName = args.modelName});
    if (!engine) {
        std::fprintf(stderr, "error: %s\n", engine.error().toString().c_str());
        return 1;
    }

    auto listener = server::Listener::bind(
            server::ListenerOptions{.address = args.host, .port = args.port});
    if (!listener) {
        std::fprintf(stderr, "error: %s\n", listener.error().toString().c_str());
        return 1;
    }

    g_listener = &*listener;
    ::SetConsoleCtrlHandler(consoleHandler, TRUE);

    std::printf("\n%s ready at http://%s:%u/v1\n", (*engine)->modelName().c_str(),
                args.host.c_str(), listener->port());
    std::printf("Context %llu tokens. Ctrl-C to stop.\n\n",
                static_cast<unsigned long long>(args.contextLength));
    std::fflush(stdout);

    const auto served = listener->serve([&](server::Connection connection) {
        // Keep-alive: an OpenAI client reuses one connection for a whole
        // conversation, and reconnecting per request would add a handshake to
        // every turn.
        while (connection.isOpen() && !g_stopping.load(std::memory_order_relaxed)) {
            auto request = connection.readRequest();
            if (!request) {
                // A clean close between requests is ordinary, not an error.
                break;
            }
            if (args.verbose) {
                std::printf("  %s %s\n", request->method.c_str(), request->target.c_str());
                std::fflush(stdout);
            }

            server::handleRequest(**engine, *request, connection);

            if (!request->keepAlive) {
                break;
            }
        }
        connection.close();
    });

    ::SetConsoleCtrlHandler(consoleHandler, FALSE);
    g_listener = nullptr;

    if (!served && !g_stopping.load(std::memory_order_relaxed)) {
        std::fprintf(stderr, "error: %s\n", served.error().toString().c_str());
        return 1;
    }

    const server::Engine::Stats stats = (*engine)->stats();
    std::printf("\nStopped after %llu requests.\n",
                static_cast<unsigned long long>(stats.requests));
    if (stats.promptTokens > 0) {
        std::printf("Prompt cache served %.1f%% of %llu prompt tokens.\n",
                    100.0 * static_cast<double>(stats.cachedPromptTokens) /
                            static_cast<double>(stats.promptTokens),
                    static_cast<unsigned long long>(stats.promptTokens));
    }
    return 0;
}
