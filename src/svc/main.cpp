// tf-decode - the process that owns the model.
//
// The GUI talks to this over a named pipe rather than loading the model itself.
// A XAML crash, a hung UI thread or a shader-compile stall would otherwise take
// a CUDA context and 13 GiB of VRAM with it, and reloading costs about eight
// seconds - long enough that users would learn not to close the window.
//
// It also means the CLI, the server and the GUI all sit on the same runtime
// rather than on three copies of it.

#include <windows.h>

#include <atomic>
#include <memory>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <string_view>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <optional>
#include <thread>
#include <vector>

#include "Pipe.h"
#include "Protocol.h"
#include "tf/core/tokenizer/ChatTemplate.h"
#include "tf/core/tokenizer/Tokenizer.h"
#include "tf/gpu/Backend.h"
#include "tf/gpu/Preflight.h"
#include "tf/runtime/Generator.h"
#include "tf/runtime/Model.h"

using namespace tf;

namespace {

struct Arguments {
    std::filesystem::path modelDir;
    std::string pipeName;
    u64 contextLength = 4096;
    u32 prefillChunk = 128;
    u64 vramBudget = 0;
    u32 expertSlots = 0;  // zero leaves the planner's default
    /// Exit when the last front end disconnects. The GUI starts the service on
    /// demand and wants it gone with the window; a service started by hand
    /// usually should not vanish.
    bool exitWhenIdle = false;
    bool verbose = false;
    bool showHelp = false;
};

void printUsage() {
    std::puts(R"(tf-decode - decode service for the TurboFieldfare GUI

Usage:
  tf-decode --model <dir.gturbo> [options]

Holds the model and answers generate requests over a named pipe, so a front end
never owns the CUDA context.

Options:
  --model <dir>        Install to load, or $TF_GTURBO_DIR
  --pipe <name>        Pipe to listen on. Defaults to a per-user name so two
                       sessions do not fight over one model.
  --context <n>        KV context length, default 4096
  --vram-budget <GiB>  Constrain residency below the hardware
  --expert-slots <n>   Slots per streamed layer, default 16
  --prefill-chunk <n>  Prompt tokens batched per pass, default 128
  --exit-when-idle     Stop once the last front end disconnects
  -v, --verbose        Log every message
  -h, --help           This message)");
}

Result<Arguments> parseArguments(int argc, char** argv) {
    Arguments args;
    args.pipeName = svc::defaultPipeName();
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
        if (flag == "--exit-when-idle") {
            args.exitWhenIdle = true;
            continue;
        }

        if (i + 1 >= argc) {
            return makeError(ErrorCode::InvalidArgument, "{} needs a value", flag);
        }
        const std::string value = argv[++i];

        if (flag == "--model") {
            args.modelDir = value;
        } else if (flag == "--pipe") {
            args.pipeName = value;
        } else if (flag == "--context") {
            args.contextLength = std::strtoull(value.c_str(), nullptr, 10);
        } else if (flag == "--prefill-chunk") {
            args.prefillChunk = static_cast<u32>(std::strtoul(value.c_str(), nullptr, 10));
        } else if (flag == "--expert-slots") {
            args.expertSlots = static_cast<u32>(std::strtoul(value.c_str(), nullptr, 10));
        } else if (flag == "--vram-budget") {
            args.vramBudget = static_cast<u64>(std::strtod(value.c_str(), nullptr) * 1024.0 *
                                               1024.0 * 1024.0);
        } else {
            return makeError(ErrorCode::InvalidArgument, "unknown argument '{}'", flag);
        }
    }

    if (!args.showHelp && args.modelDir.empty()) {
        return makeError(ErrorCode::InvalidArgument, "--model is required, or set TF_GTURBO_DIR");
    }
    return args;
}

svc::PipeServer* g_server = nullptr;
std::atomic<bool> g_stopping{false};

BOOL WINAPI consoleHandler(DWORD signal) {
    if (signal == CTRL_C_EVENT || signal == CTRL_CLOSE_EVENT || signal == CTRL_BREAK_EVENT) {
        g_stopping.store(true, std::memory_order_relaxed);
        if (g_server != nullptr) {
            g_server->stop();
        }
        return TRUE;
    }
    return FALSE;
}

/// Everything the service owns for the life of the process.
///
/// Held indirectly, and that is load-bearing: Generator::create stores pointers
/// to the model and the tokenizer it is given, so moving either afterwards
/// leaves the forward runner reading a moved-from object. Constructing them in
/// place and never moving them is what keeps those pointers valid.
struct Service {
    gpu::BackendPtr backend;
    std::unique_ptr<runtime::Model> model;
    std::unique_ptr<Tokenizer> tokenizer;
    std::unique_ptr<ChatTemplate> chatTemplate;
    std::unique_ptr<runtime::Generator> generator;

    svc::StatusInfo status;
};

/// Queue between the reader thread and the request loop.
class MessageQueue {
public:
    void push(svc::Message message) {
        {
            const std::lock_guard lock{mutex_};
            queue_.push_back(std::move(message));
        }
        ready_.notify_one();
    }

    /// Signals that no more messages will arrive.
    void finish() {
        {
            const std::lock_guard lock{mutex_};
            finished_ = true;
        }
        ready_.notify_all();
    }

    /// Blocks for the next message. Empty when the peer is gone.
    [[nodiscard]] std::optional<svc::Message> pop() {
        std::unique_lock lock{mutex_};
        ready_.wait(lock, [this] { return !queue_.empty() || finished_; });
        if (queue_.empty()) {
            return std::nullopt;
        }
        svc::Message message = std::move(queue_.front());
        queue_.pop_front();
        return message;
    }

private:
    std::mutex mutex_;
    std::condition_variable ready_;
    std::deque<svc::Message> queue_;
    bool finished_ = false;
};

/// Serves one connected front end until it disconnects.
///
/// Requests are handled one at a time, but reading runs on its own thread. That
/// is what makes Stop work: a cancel arriving while generate() is running sets a
/// flag the token callback checks, so the stop takes effect within one token
/// rather than at the end of the answer. With a single thread the cancel would
/// not even be read until the generation it was meant to interrupt had
/// finished.
///
/// Only the request loop writes, so the write side stays single-threaded.
void serveConnection(Service& service, svc::PipeChannel& channel, const Arguments& args) {
    // Written by the reader thread, read by the decode callback.
    std::atomic<u64> cancelRequested{0};

    const auto send = [&](const svc::Message& message) {
        const auto status = channel.write(message);
        if (!status && args.verbose) {
            std::printf("  write failed: %s\n", status.error().message().c_str());
        }
        return status.has_value();
    };

    if (!send(svc::makeReady(svc::ReadyInfo{
                .model = service.status.model,
                .contextLength = service.status.contextLength,
                .device = service.backend->info().name,
                .fullyResident = service.model->residency().isFullyResident()}))) {
        return;
    }

    MessageQueue queue;
    std::thread reader{[&] {
        for (;;) {
            auto incoming = channel.read();
            if (!incoming) {
                // A clean disconnect is how a front end exits.
                if (args.verbose && incoming.error().code() != ErrorCode::Cancelled) {
                    std::printf("  read failed: %s\n", incoming.error().message().c_str());
                }
                queue.finish();
                return;
            }
            // Handled here rather than queued: a cancel behind a running
            // generation in the queue would arrive only after that generation
            // ended, which is exactly when it is useless.
            if (incoming->kind == svc::MessageKind::Cancel) {
                cancelRequested.store(incoming->id, std::memory_order_relaxed);
                continue;
            }
            queue.push(std::move(*incoming));
        }
    }};

    // The reader owns the pipe until it sees the peer go; closing here is what
    // releases it if the loop below decides to stop first.
    struct ReaderJoin {
        std::thread& thread;
        svc::PipeChannel& channel;
        ~ReaderJoin() {
            channel.close();
            if (thread.joinable()) {
                thread.join();
            }
        }
    } join{reader, channel};

    for (;;) {
        auto incoming = queue.pop();
        if (!incoming) {
            return;
        }
        const svc::Message& message = *incoming;

        if (args.verbose) {
            std::printf("  <- %s\n", std::string{svc::toString(message.kind)}.c_str());
            std::fflush(stdout);
        }

        if (message.version != svc::kProtocolVersion) {
            static_cast<void>(send(svc::makeError(
                    message.id,
                    std::format("protocol version {} is not the {} this service speaks",
                                message.version, svc::kProtocolVersion))));
            return;
        }

        switch (message.kind) {
            case svc::MessageKind::Hello:
                static_cast<void>(send(svc::makeReady(svc::ReadyInfo{
                        .model = service.status.model,
                        .contextLength = service.status.contextLength,
                        .device = service.backend->info().name,
                        .fullyResident =
                                service.model->residency().isFullyResident()})));
                break;

            case svc::MessageKind::Reset:
                service.generator->reset();
                service.status.position = 0;
                static_cast<void>(send(svc::makeStatusReport(service.status)));
                break;

            case svc::MessageKind::Status:
                service.status.position = service.generator->position();
                static_cast<void>(send(svc::makeStatusReport(service.status)));
                break;

            case svc::MessageKind::Shutdown:
                g_stopping.store(true, std::memory_order_relaxed);
                if (g_server != nullptr) {
                    g_server->stop();
                }
                return;


            case svc::MessageKind::Generate: {
                const svc::GenerateRequest& request = message.generate;

                std::vector<u32> promptIds;
                std::vector<u32> stopTokens;
                if (request.chat) {
                    auto rendered = service.chatTemplate->render(request.messages);
                    if (!rendered) {
                        static_cast<void>(
                                send(svc::makeError(request.id, rendered.error().message())));
                        break;
                    }
                    promptIds = std::move(*rendered);
                    stopTokens = service.chatTemplate->stopTokens();
                } else {
                    promptIds = service.tokenizer->encode(request.prompt);
                    promptIds.insert(promptIds.begin(), service.model->arch().bosTokenId);
                    stopTokens.assign(service.model->arch().eosTokenIds.begin(),
                                      service.model->arch().eosTokenIds.end());
                }

                runtime::GenerationOptions options;
                options.sampling = request.sampling;
                options.maxTokens = request.maxTokens;
                options.stopStrings = request.stopStrings;
                options.stopTokens = std::move(stopTokens);
                options.prefillChunk = args.prefillChunk;

                if (request.includeThinking) {
                    service.generator->setThinkingCallback(
                            [&](std::string_view piece) {
                                static_cast<void>(send(
                                        svc::makeThinking(request.id, std::string{piece})));
                            });
                } else {
                    service.generator->setThinkingCallback({});
                }

                service.status.busy = true;
                auto stats = service.generator->generate(
                        promptIds, options, [&](std::string_view piece) {
                            if (cancelRequested.load(std::memory_order_relaxed) ==
                                request.id) {
                                return false;
                            }
                            return send(svc::makeToken(request.id, std::string{piece}));
                        });
                service.status.busy = false;

                if (!stats) {
                    static_cast<void>(
                            send(svc::makeError(request.id, stats.error().message())));
                    break;
                }

                service.status.requests += 1;
                service.status.promptTokens += stats->promptTokens;
                service.status.cachedPromptTokens += stats->cachedPromptTokens;
                service.status.generatedTokens += stats->generatedTokens;
                service.status.position = service.generator->position();
                if (!service.model->residency().isFullyResident()) {
                    const auto& streamer = service.model->streamer().stats();
                    service.status.expertHitRate = streamer.hitRate();
                    service.status.expertBytesRead = streamer.bytesRead;
                }

                static_cast<void>(send(svc::makeDone(svc::DoneInfo{
                        .id = request.id,
                        .reason = std::string{runtime::describe(stats->reason)},
                        .promptTokens = stats->promptTokens,
                        .cachedPromptTokens = stats->cachedPromptTokens,
                        .generatedTokens = stats->generatedTokens,
                        .prefillSeconds = stats->prefillSeconds,
                        .decodeSeconds = stats->decodeSeconds})));
                break;
            }

            default:
                static_cast<void>(send(svc::makeError(
                        message.id, std::format("a service does not receive '{}'",
                                                svc::toString(message.kind)))));
                break;
        }
    }
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

    // The pipe is claimed before the model loads, so a second launch fails in a
    // second rather than after eight seconds of loading.
    auto server = svc::PipeServer::create(args.pipeName);
    if (!server) {
        std::fprintf(stderr, "error: %s\n", server.error().toString().c_str());
        return 1;
    }

    std::printf("Loading %s\n", args.modelDir.string().c_str());

    const auto backends = gpu::compiledBackends();
    if (backends.empty()) {
        std::fprintf(stderr, "error: this build has no GPU backend\n");
        return 1;
    }
    auto backend = gpu::createBackend(backends.front());
    if (!backend) {
        std::fprintf(stderr, "error: %s\n", backend.error().toString().c_str());
        return 1;
    }

    runtime::Model::LoadOptions load;
    load.budget.deviceBytes = args.vramBudget;
    load.budget.contextLength = args.contextLength;
    if (args.expertSlots > 0) {
        load.budget.slotsPerStreamedLayer = args.expertSlots;
    }
    load.budget.maxPrefillTokens = args.prefillChunk;

    // Each piece is moved into its final home before the next one takes a
    // reference to it, so nothing the generator points at ever moves again.
    Service service;
    service.backend = std::move(*backend);

    auto model = runtime::Model::load(*service.backend, args.modelDir, load);
    if (!model) {
        std::fprintf(stderr, "error: %s\n", model.error().toString().c_str());
        return 1;
    }
    service.model = std::make_unique<runtime::Model>(std::move(*model));

    auto tokenizer = Tokenizer::loadFromFile(args.modelDir / "tokenizer" / "tokenizer.json");
    if (!tokenizer) {
        std::fprintf(stderr, "error: %s\n", tokenizer.error().toString().c_str());
        return 1;
    }
    service.tokenizer = std::make_unique<Tokenizer>(std::move(*tokenizer));

    auto chatTemplate = ChatTemplate::create(*service.tokenizer);
    if (!chatTemplate) {
        std::fprintf(stderr, "error: %s\n", chatTemplate.error().toString().c_str());
        return 1;
    }
    service.chatTemplate = std::make_unique<ChatTemplate>(std::move(*chatTemplate));

    auto generator = runtime::Generator::create(*service.backend, *service.model,
                                                *service.tokenizer, args.contextLength,
                                                args.prefillChunk);
    if (!generator) {
        std::fprintf(stderr, "error: %s\n", generator.error().toString().c_str());
        return 1;
    }
    service.generator = std::make_unique<runtime::Generator>(std::move(*generator));
    service.status.model = args.modelDir.stem().string();
    service.status.contextLength = args.contextLength;

    g_server = &*server;
    ::SetConsoleCtrlHandler(consoleHandler, TRUE);

    std::printf("\n%s ready on %s\n", service.status.model.c_str(), args.pipeName.c_str());
    std::printf("Context %llu tokens on %s. Ctrl-C to stop.\n\n",
                static_cast<unsigned long long>(args.contextLength),
                service.backend->info().name.c_str());
    std::fflush(stdout);

    while (!g_stopping.load(std::memory_order_relaxed)) {
        auto channel = server->accept();
        if (!channel) {
            break;
        }
        if (args.verbose) {
            std::puts("front end connected");
        }

        serveConnection(service, *channel, args);
        channel->close();

        if (args.verbose) {
            std::puts("front end disconnected");
        }
        if (args.exitWhenIdle) {
            break;
        }
    }

    ::SetConsoleCtrlHandler(consoleHandler, FALSE);
    g_server = nullptr;

    std::printf("\nStopped after %llu requests.\n",
                static_cast<unsigned long long>(service.status.requests));
    return 0;
}
