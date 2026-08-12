// tf-decode-client - talks to the decode service from a terminal.
//
// The GUI's job in miniature: connect, send a prompt, stream tokens back. It
// exists so the service can be exercised end to end without building a XAML
// app, and so a user can tell whether a problem is in the service or the UI.

#include <cstdio>
#include <cstdlib>
#include <string>
#include <string_view>
#include <vector>

#include "Pipe.h"
#include "Protocol.h"

using namespace tf;

namespace {

void printUsage() {
    std::puts(R"(tf-decode-client - send a prompt to a running tf-decode

Usage:
  tf-decode-client --prompt "..." [options]
  tf-decode-client --status

Options:
  --prompt <text>      What to ask
  --raw                Skip the instruction template
  --tokens <n>         Most tokens to generate, default 256
  --temperature <t>    0 is greedy, default 0.2
  --seed <n>           Make a run reproducible
  --thinking           Show the thinking channel too
  --status             Print the service's counters and exit
  --reset              Clear the conversation and exit
  --pipe <name>        Service pipe, default the per-user one
  -h, --help           This message)");
}

}  // namespace

int main(int argc, char** argv) {
    std::string pipeName = svc::defaultPipeName();
    std::string prompt;
    bool chat = true;
    bool statusOnly = false;
    bool resetOnly = false;
    bool thinking = false;
    u64 maxTokens = 256;
    float temperature = 0.2f;
    u64 seed = 0;

    for (int i = 1; i < argc; ++i) {
        const std::string_view flag = argv[i];
        if (flag == "-h" || flag == "--help") {
            printUsage();
            return 0;
        }
        if (flag == "--raw") {
            chat = false;
            continue;
        }
        if (flag == "--status") {
            statusOnly = true;
            continue;
        }
        if (flag == "--reset") {
            resetOnly = true;
            continue;
        }
        if (flag == "--thinking") {
            thinking = true;
            continue;
        }
        if (i + 1 >= argc) {
            std::fprintf(stderr, "error: %.*s needs a value\n",
                         static_cast<int>(flag.size()), flag.data());
            return 2;
        }
        const std::string value = argv[++i];
        if (flag == "--prompt") {
            prompt = value;
        } else if (flag == "--pipe") {
            pipeName = value;
        } else if (flag == "--tokens") {
            maxTokens = std::strtoull(value.c_str(), nullptr, 10);
        } else if (flag == "--temperature") {
            temperature = static_cast<float>(std::strtod(value.c_str(), nullptr));
        } else if (flag == "--seed") {
            seed = std::strtoull(value.c_str(), nullptr, 10);
        } else {
            std::fprintf(stderr, "error: unknown argument '%s'\n", value.c_str());
            return 2;
        }
    }

    if (prompt.empty() && !statusOnly && !resetOnly) {
        std::fprintf(stderr, "error: --prompt is required\n\n");
        printUsage();
        return 2;
    }

    auto channel = svc::PipeClient::connect(pipeName);
    if (!channel) {
        std::fprintf(stderr, "error: %s\n", channel.error().toString().c_str());
        std::fprintf(stderr, "Is tf-decode running?\n");
        return 1;
    }

    // The service greets on connect, before anything is asked of it.
    auto ready = channel->read();
    if (!ready || ready->kind != svc::MessageKind::Ready) {
        std::fprintf(stderr, "error: the service did not greet\n");
        return 1;
    }
    std::fprintf(stderr, "%s on %s, %llu token context%s\n", ready->ready.model.c_str(),
                 ready->ready.device.c_str(),
                 static_cast<unsigned long long>(ready->ready.contextLength),
                 ready->ready.fullyResident ? "" : ", streaming experts");

    const auto request = [&](svc::Message message) -> int {
        if (const auto sent = channel->write(message); !sent) {
            std::fprintf(stderr, "error: %s\n", sent.error().toString().c_str());
            return 1;
        }
        return 0;
    };

    if (resetOnly) {
        if (request(svc::Message{.kind = svc::MessageKind::Reset}) != 0) {
            return 1;
        }
        static_cast<void>(channel->read());
        std::fprintf(stderr, "conversation cleared\n");
        return 0;
    }

    if (statusOnly) {
        if (request(svc::Message{.kind = svc::MessageKind::Status}) != 0) {
            return 1;
        }
        auto reply = channel->read();
        if (!reply || reply->kind != svc::MessageKind::StatusReport) {
            std::fprintf(stderr, "error: no status came back\n");
            return 1;
        }
        const svc::StatusInfo& status = reply->status;
        std::printf("model        %s\n", status.model.c_str());
        std::printf("context      %llu of %llu used\n",
                    static_cast<unsigned long long>(status.position),
                    static_cast<unsigned long long>(status.contextLength));
        std::printf("requests     %llu\n", static_cast<unsigned long long>(status.requests));
        std::printf("prompt       %llu tokens, %llu served from cache\n",
                    static_cast<unsigned long long>(status.promptTokens),
                    static_cast<unsigned long long>(status.cachedPromptTokens));
        std::printf("generated    %llu tokens\n",
                    static_cast<unsigned long long>(status.generatedTokens));
        if (status.expertBytesRead > 0) {
            std::printf("expert cache %.1f%% hits, %.2f GiB read\n",
                        status.expertHitRate * 100.0,
                        static_cast<double>(status.expertBytesRead) /
                                (1024.0 * 1024.0 * 1024.0));
        }
        return 0;
    }

    svc::Message generate{.kind = svc::MessageKind::Generate};
    generate.generate.id = 1;
    generate.generate.chat = chat;
    if (chat) {
        generate.generate.messages.push_back(ChatMessage{ChatRole::User, prompt});
    } else {
        generate.generate.prompt = prompt;
    }
    generate.generate.maxTokens = maxTokens;
    generate.generate.sampling.temperature = temperature;
    generate.generate.sampling.seed = seed;
    generate.generate.includeThinking = thinking;

    if (request(generate) != 0) {
        return 1;
    }

    for (;;) {
        auto incoming = channel->read();
        if (!incoming) {
            std::fprintf(stderr, "\nerror: %s\n", incoming.error().toString().c_str());
            return 1;
        }

        switch (incoming->kind) {
            case svc::MessageKind::Token:
                // Flushed per token: the point of streaming is that the first
                // words appear while the rest is still being produced.
                std::fwrite(incoming->text.data(), 1, incoming->text.size(), stdout);
                std::fflush(stdout);
                break;

            case svc::MessageKind::Thinking:
                std::fwrite(incoming->text.data(), 1, incoming->text.size(), stderr);
                break;

            case svc::MessageKind::Error:
                std::fprintf(stderr, "\nerror: %s\n", incoming->text.c_str());
                return 1;

            case svc::MessageKind::Done: {
                const svc::DoneInfo& done = incoming->done;
                std::fputc('\n', stdout);
                std::fprintf(stderr,
                             "\n%llu prompt tokens (%llu cached) in %.2fs, %llu generated in "
                             "%.2fs (%.1f tok/s), stopped on %s\n",
                             static_cast<unsigned long long>(done.promptTokens),
                             static_cast<unsigned long long>(done.cachedPromptTokens),
                             done.prefillSeconds,
                             static_cast<unsigned long long>(done.generatedTokens),
                             done.decodeSeconds,
                             done.decodeSeconds > 0.0
                                     ? static_cast<double>(done.generatedTokens) /
                                               done.decodeSeconds
                                     : 0.0,
                             done.reason.c_str());
                return 0;
            }

            default:
                break;
        }
    }
}
