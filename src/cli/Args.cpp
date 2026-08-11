#include "Args.h"

#include <charconv>
#include <cstdlib>
#include <format>

namespace tf::cli {
namespace {

/// Parses an integer, rejecting anything with trailing characters. strtoull
/// would read "12abc" as 12 and "abc" as 0, both of which are worse than an
/// error message.
template <class T>
[[nodiscard]] Result<T> parseInteger(std::string_view text, std::string_view flag) {
    T value{};
    const auto* end = text.data() + text.size();
    const auto [stop, code] = std::from_chars(text.data(), end, value);
    if (code != std::errc{} || stop != end) {
        return makeError(ErrorCode::InvalidArgument, "{} expects a whole number, not '{}'", flag,
                         text);
    }
    return value;
}

[[nodiscard]] Result<double> parseDouble(std::string_view text, std::string_view flag) {
    // from_chars for floating point is not universally available with the same
    // behaviour, so this goes through strtod and checks the end pointer, which
    // gives the same rejection of trailing characters.
    const std::string owned{text};
    char* stop = nullptr;
    const double value = std::strtod(owned.c_str(), &stop);
    if (stop != owned.c_str() + owned.size() || owned.empty()) {
        return makeError(ErrorCode::InvalidArgument, "{} expects a number, not '{}'", flag, text);
    }
    return value;
}

/// Turns the two-character escapes a shell will not pass through into real
/// characters, so `--stop '\n\nUser:'` works from any shell.
[[nodiscard]] std::string unescape(std::string_view text) {
    std::string out;
    out.reserve(text.size());
    for (usize i = 0; i < text.size(); ++i) {
        if (text[i] != '\\' || i + 1 >= text.size()) {
            out += text[i];
            continue;
        }
        switch (text[i + 1]) {
            case 'n': out += '\n'; ++i; break;
            case 't': out += '\t'; ++i; break;
            case 'r': out += '\r'; ++i; break;
            case '\\': out += '\\'; ++i; break;
            default: out += text[i]; break;
        }
    }
    return out;
}

}  // namespace

std::string usage() {
    return R"(tf-cli - run Gemma 4 on Windows

Usage:
  tf-cli --model <dir.gturbo> --prompt "..."
  tf-cli --model <dir.gturbo> --stdin < prompt.txt

Model:
  --model <dir>        Install to load. Defaults to $TF_GTURBO_DIR.
  --context <n>        KV context length, default 4096.

Prompt:
  --prompt <text>      The prompt.
  --stdin              Read the prompt from standard input instead.
  --system <text>      A system message placed before the prompt.
  --raw                Do not apply the instruction template. The checkpoint is
                       instruction-tuned, so raw completion of a question is out
                       of distribution and usually gives a worse answer.

Sampling:
  --temperature <t>    0 is greedy, default 0.2.
  --top-k <n>          Keep the n highest-scoring tokens, 0 disables. Default 64.
  --top-p <p>          Nucleus threshold, 1 disables. Default 0.95.
  --repeat-penalty <r> Divides the logits of recently produced tokens. Default 1.
  --repeat-window <n>  How far the penalty looks back, default 64.
  --seed <n>           Fix the sampler's seed to make a run reproducible.
  --tokens <n>         Most tokens to generate, default 512.
  --stop <text>        End generation at this text. Repeatable. \n and \t work.
  --show-thinking      Also print the model's thinking channel, to stderr. It is
                       separate markup, not part of the answer.

Memory:
  --vram-budget <GiB>  Plan residency below what the card has, which is how the
                       streamed path is exercised on hardware that would
                       otherwise hold the whole model.
  --expert-slots <n>   Slots per streamed layer. More is markedly faster when
                       anything streams; the planner picks the largest that
                       fits and reports it under --verbose.
  --prefill-chunk <n>  Prompt tokens batched per pass, default 128. Larger is
                       faster and costs scratch and KV rows.
  --unbuffered         Read experts with FILE_FLAG_NO_BUFFERING, bypassing the
                       page cache. Only worth it when RAM cannot hold the
                       expert set; on this machine it is slower.
  --read-threads <n>   Threads issuing expert reads, default 4.
  --backend <name>     cuda or d3d12. Default is the first compiled, which is
                       cuda where it is available. d3d12 runs on AMD and Intel
                       as well, and is slower on NVIDIA.

Output:
  -v, --verbose        Residency plan and timings.
  -q, --quiet          Only the generated text.
  --dry-run            Print the resolved settings and exit.
  --version            Version and build information.
  -h, --help           This message.)";
}

Result<Args> Args::parse(std::span<const std::string_view> arguments) {
    Args args;
    if (const char* env = std::getenv("TF_GTURBO_DIR")) {
        args.modelDir = env;
    }

    for (usize i = 0; i < arguments.size(); ++i) {
        const std::string_view flag = arguments[i];

        // Flags that take no value.
        if (flag == "-h" || flag == "--help") {
            args.showHelp = true;
            return args;
        }
        if (flag == "--version") {
            args.showVersion = true;
            return args;
        }
        if (flag == "--raw") {
            args.chat = false;
            continue;
        }
        if (flag == "--stdin") {
            args.promptFromStdin = true;
            continue;
        }
        if (flag == "--show-thinking") {
            args.showThinking = true;
            continue;
        }
        if (flag == "-v" || flag == "--verbose") {
            args.verbose = true;
            continue;
        }
        if (flag == "-q" || flag == "--quiet") {
            args.quiet = true;
            continue;
        }
        if (flag == "--dry-run") {
            args.dryRun = true;
            continue;
        }
        if (flag == "--unbuffered") {
            args.unbufferedReads = true;
            continue;
        }

        // Everything below needs a value.
        if (i + 1 >= arguments.size()) {
            return makeError(ErrorCode::InvalidArgument, "{} needs a value", flag);
        }
        const std::string_view value = arguments[++i];

        if (flag == "--model") {
            args.modelDir = value;
        } else if (flag == "--prompt") {
            args.prompt = value;
        } else if (flag == "--system") {
            args.system = value;
        } else if (flag == "--stop") {
            args.stopStrings.push_back(unescape(value));
        } else if (flag == "--temperature") {
            TF_TRY(const double temperature, parseDouble(value, flag));
            args.sampling.temperature = static_cast<float>(temperature);
        } else if (flag == "--top-k") {
            TF_TRY(args.sampling.topK, parseInteger<u32>(value, flag));
        } else if (flag == "--top-p") {
            TF_TRY(const double topP, parseDouble(value, flag));
            args.sampling.topP = static_cast<float>(topP);
        } else if (flag == "--repeat-penalty") {
            TF_TRY(const double penalty, parseDouble(value, flag));
            args.sampling.repetitionPenalty = static_cast<float>(penalty);
        } else if (flag == "--repeat-window") {
            TF_TRY(args.sampling.repetitionWindow, parseInteger<u64>(value, flag));
        } else if (flag == "--seed") {
            TF_TRY(args.sampling.seed, parseInteger<u64>(value, flag));
        } else if (flag == "--tokens") {
            TF_TRY(args.maxTokens, parseInteger<u64>(value, flag));
        } else if (flag == "--context") {
            TF_TRY(args.contextLength, parseInteger<u64>(value, flag));
        } else if (flag == "--vram-budget") {
            TF_TRY(const double gib, parseDouble(value, flag));
            args.vramBudget = static_cast<u64>(gib * 1024.0 * 1024.0 * 1024.0);
        } else if (flag == "--expert-slots") {
            TF_TRY(args.expertSlots, parseInteger<u32>(value, flag));
        } else if (flag == "--prefill-chunk") {
            TF_TRY(args.prefillChunk, parseInteger<u32>(value, flag));
        } else if (flag == "--backend") {
            args.backend = value;
        } else if (flag == "--read-threads") {
            TF_TRY(args.readThreads, parseInteger<u32>(value, flag));
        } else {
            return makeError(ErrorCode::InvalidArgument, "unknown argument '{}'", flag);
        }
    }

    return args;
}

Status Args::validate() const {
    if (showHelp || showVersion) {
        return {};
    }
    if (modelDir.empty()) {
        return makeError(ErrorCode::InvalidArgument,
                         "--model is required, or set TF_GTURBO_DIR");
    }
    if (prompt.empty() && !promptFromStdin) {
        return makeError(ErrorCode::InvalidArgument, "--prompt or --stdin is required");
    }
    if (!prompt.empty() && promptFromStdin) {
        return makeError(ErrorCode::InvalidArgument,
                         "--prompt and --stdin both give a prompt; use one");
    }
    if (maxTokens == 0) {
        return makeError(ErrorCode::InvalidArgument, "--tokens must be at least 1");
    }
    if (contextLength == 0) {
        return makeError(ErrorCode::InvalidArgument, "--context must be at least 1");
    }
    if (prefillChunk == 0) {
        return makeError(ErrorCode::InvalidArgument, "--prefill-chunk must be at least 1");
    }
    if (readThreads == 0) {
        return makeError(ErrorCode::InvalidArgument, "--read-threads must be at least 1");
    }
    if (prefillChunk > contextLength) {
        return makeError(ErrorCode::InvalidArgument,
                         "--prefill-chunk {} exceeds the {} token context", prefillChunk,
                         contextLength);
    }
    if (verbose && quiet) {
        return makeError(ErrorCode::InvalidArgument, "--verbose and --quiet contradict");
    }
    if (!system.empty() && !chat) {
        return makeError(ErrorCode::InvalidArgument,
                         "--system needs the instruction template; drop --raw");
    }
    TF_CHECK(sampling.validate());
    return {};
}

}  // namespace tf::cli
