// tf-repack - builds a .gturbo install from a Gemma 4 checkpoint.

#include <windows.h>

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

#include "tf/repack/Installer.h"
#include "tf/repack/Repacker.h"

using namespace tf;

namespace {

void printUsage() {
    std::puts(R"(tf-repack - build a .gturbo install from a Gemma 4 checkpoint

Usage:
  tf-repack --install --output <dir.gturbo> [options]
  tf-repack --checkpoint <dir> --output <dir.gturbo> [options]
  tf-repack --verify-install <dir.gturbo>

Source:
  --install              Stream directly from Hugging Face. The source is never
                         written to disk, so this needs room for the 14.3 GiB
                         install rather than for that plus a 15 GB download.
  --checkpoint <dir>     Repack from an already downloaded checkpoint instead.
  --repo-id <id>         Repository, default mlx-community/gemma-4-26b-a4b-it-4bit
  --revision <sha>       Commit to install. A hash rather than a branch, so a
                         resumed install cannot mix two revisions.
  --hf-token <token>     For a gated repository. Also read from HF_TOKEN.
  --endpoint <url>       Mirror or proxy, default https://huggingface.co

Options:
  --output <dir>         Destination .gturbo directory
  --overwrite            Replace an existing install at --output
  --no-resume            Start a streaming install over rather than continuing
  --no-verify            Skip SHA-256 hashing after writing (faster, less safe)
  --verify-install <dir> Re-verify an existing install and exit
  --max-op-bytes <n>     Copy scratch bound, default 8388608
  -h, --help             This message

The install is built in <output>.partial and renamed on success, so an
interrupted run never leaves a directory that looks complete. A streaming
install checkpoints as it goes and resumes from where it stopped.)");
}

std::string formatBytes(u64 bytes) {
    constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;
    constexpr double kMiB = 1024.0 * 1024.0;
    if (bytes >= static_cast<u64>(kGiB)) {
        return std::format("{:.2f} GiB", static_cast<double>(bytes) / kGiB);
    }
    return std::format("{:.1f} MiB", static_cast<double>(bytes) / kMiB);
}

std::string formatDuration(double seconds) {
    if (seconds < 60.0) {
        return std::format("{:.1f}s", seconds);
    }
    const auto minutes = static_cast<int>(seconds) / 60;
    const auto remainder = seconds - minutes * 60;
    return std::format("{}m {:.0f}s", minutes, remainder);
}

/// Draws a single-line progress bar that rewrites itself in place. Falls back
/// to plain lines when output is redirected, so logs stay readable.
class ProgressPrinter {
public:
    ProgressPrinter() {
        interactive_ = ::GetFileType(::GetStdHandle(STD_OUTPUT_HANDLE)) == FILE_TYPE_CHAR;
    }

    bool operator()(const repack::Progress& progress) {
        const auto now = std::chrono::steady_clock::now();
        const bool phaseChanged = progress.phase != lastPhase_;
        const bool complete = progress.bytesTotal > 0 &&
                              progress.bytesDone == progress.bytesTotal;

        // Throttle to ~10 Hz; the copy loop reports far more often than a
        // human can read.
        if (!phaseChanged && !complete &&
            now - lastDraw_ < std::chrono::milliseconds(100)) {
            return true;
        }
        lastDraw_ = now;
        lastPhase_ = progress.phase;

        const std::string_view phase = repack::toString(progress.phase);
        std::string line;
        if (progress.bytesTotal > 0) {
            const double fraction =
                    static_cast<double>(progress.bytesDone) / static_cast<double>(progress.bytesTotal);
            line = std::format("  {:<17} {:5.1f}%  {:>10} / {:<10} {}", phase, fraction * 100.0,
                               formatBytes(progress.bytesDone),
                               formatBytes(progress.bytesTotal), progress.detail);
        } else {
            line = std::format("  {:<17}        {}", phase, progress.detail);
        }

        if (interactive_) {
            // Pad to erase whatever the previous, possibly longer, line left.
            const usize width = std::max(line.size(), lastWidth_);
            std::printf("\r%-*s", static_cast<int>(width), line.c_str());
            lastWidth_ = line.size();
            std::fflush(stdout);
        } else if (phaseChanged || complete) {
            std::printf("%s\n", line.c_str());
        }
        return true;
    }

    void finishLine() {
        if (interactive_ && lastWidth_ > 0) {
            std::printf("\n");
            lastWidth_ = 0;
        }
    }

private:
    bool interactive_ = false;
    usize lastWidth_ = 0;
    repack::Phase lastPhase_ = repack::Phase::Planning;
    std::chrono::steady_clock::time_point lastDraw_{};
};

struct Arguments {
    std::filesystem::path checkpoint;
    std::filesystem::path output;
    std::filesystem::path verifyInstall;
    bool install = false;
    bool overwrite = false;
    bool verify = true;
    bool resume = true;
    u64 maxOpBytes = 8ull * 1024 * 1024;
    std::string repoId;
    std::string revision;
    std::string token;
    std::string endpoint;
};

Result<Arguments> parseArguments(int argc, char** argv) {
    Arguments args;

    const auto requireValue = [&](int& i, std::string_view flag) -> Result<std::string> {
        if (i + 1 >= argc) {
            return makeError(ErrorCode::InvalidArgument, "{} needs a value", flag);
        }
        return std::string{argv[++i]};
    };

    for (int i = 1; i < argc; ++i) {
        const std::string_view flag = argv[i];

        if (flag == "-h" || flag == "--help") {
            printUsage();
            std::exit(0);
        } else if (flag == "--checkpoint") {
            TF_TRY(const std::string value, requireValue(i, flag));
            args.checkpoint = value;
        } else if (flag == "--output") {
            TF_TRY(const std::string value, requireValue(i, flag));
            args.output = value;
        } else if (flag == "--verify-install") {
            TF_TRY(const std::string value, requireValue(i, flag));
            args.verifyInstall = value;
        } else if (flag == "--install") {
            args.install = true;
        } else if (flag == "--overwrite") {
            args.overwrite = true;
        } else if (flag == "--no-verify") {
            args.verify = false;
        } else if (flag == "--no-resume") {
            args.resume = false;
        } else if (flag == "--hf-token") {
            TF_TRY(args.token, requireValue(i, flag));
        } else if (flag == "--endpoint") {
            TF_TRY(args.endpoint, requireValue(i, flag));
        } else if (flag == "--max-op-bytes") {
            TF_TRY(const std::string value, requireValue(i, flag));
            args.maxOpBytes = std::strtoull(value.c_str(), nullptr, 10);
            if (args.maxOpBytes == 0) {
                return makeError(ErrorCode::InvalidArgument,
                                 "--max-op-bytes must be a positive integer");
            }
        } else if (flag == "--repo-id") {
            TF_TRY(args.repoId, requireValue(i, flag));
        } else if (flag == "--revision") {
            TF_TRY(args.revision, requireValue(i, flag));
        } else {
            return makeError(ErrorCode::InvalidArgument, "unknown argument '{}'", flag);
        }
    }

    if (args.verifyInstall.empty()) {
        if (args.install && !args.checkpoint.empty()) {
            return makeError(ErrorCode::InvalidArgument,
                             "--install and --checkpoint name two different sources; use one");
        }
        if (!args.install && args.checkpoint.empty()) {
            return makeError(ErrorCode::InvalidArgument,
                             "either --install or --checkpoint is required");
        }
        if (args.output.empty()) {
            return makeError(ErrorCode::InvalidArgument, "--output is required");
        }
    }
    // A token in the environment is the usual way to pass a secret; an explicit
    // flag wins so a one-off can override it.
    if (args.token.empty()) {
        if (const char* env = std::getenv("HF_TOKEN")) {
            args.token = env;
        }
    }
    return args;
}

int runVerify(const Arguments& args) {
    std::printf("Verifying %s\n", args.verifyInstall.string().c_str());

    ProgressPrinter printer;
    const auto status = repack::verifyInstalled(
            args.verifyInstall, [&](const repack::Progress& p) { return printer(p); });
    printer.finishLine();

    if (!status) {
        std::fprintf(stderr, "\nverification failed: %s\n", status.error().toString().c_str());
        return 1;
    }
    std::printf("Install verified.\n");
    return 0;
}

int runRepack(const Arguments& args) {
    repack::RepackOptions options;
    options.overwrite = args.overwrite;
    options.verifyAfterWrite = args.verify;
    options.plan.maxOpBytes = args.maxOpBytes;
    if (!args.repoId.empty()) {
        options.source.repoId = args.repoId;
    }
    if (!args.revision.empty()) {
        options.source.revision = args.revision;
    }

    std::printf("Source : %s\n", args.checkpoint.string().c_str());
    std::printf("Output : %s\n\n", args.output.string().c_str());

    ProgressPrinter printer;
    const auto result = repack::repackFromCheckpoint(
            args.checkpoint, args.output, options,
            [&](const repack::Progress& p) { return printer(p); });
    printer.finishLine();

    if (!result) {
        std::fprintf(stderr, "\nrepack failed: %s\n", result.error().toString().c_str());
        return 1;
    }

    std::printf("\nInstalled to %s\n", result->installDir.string().c_str());
    std::printf("  resident : %s\n", formatBytes(result->residentBytes).c_str());
    std::printf("  experts  : %s\n", formatBytes(result->expertBytes).c_str());
    std::printf("  total    : %s in %s\n", formatBytes(result->bytesWritten).c_str(),
                formatDuration(result->elapsedSeconds).c_str());
    if (result->elapsedSeconds > 0.0) {
        const double throughput = static_cast<double>(result->bytesWritten) /
                                  result->elapsedSeconds / (1024.0 * 1024.0);
        std::printf("  average  : %.0f MiB/s\n", throughput);
    }
    return 0;
}

int runInstall(const Arguments& args) {
    repack::InstallOptions options;
    options.overwrite = args.overwrite;
    options.verifyAfterWrite = args.verify;
    options.resume = args.resume;
    options.plan.maxOpBytes = args.maxOpBytes;
    if (!args.repoId.empty()) {
        options.source.repoId = args.repoId;
    }
    if (!args.revision.empty()) {
        options.source.revision = args.revision;
    }
    if (!args.token.empty()) {
        options.source.token = args.token;
    }
    if (!args.endpoint.empty()) {
        options.source.endpoint = args.endpoint;
    }

    std::printf("Source : %s @ %s\n", options.source.repoId.c_str(),
                options.source.revision.c_str());
    std::printf("Output : %s\n", args.output.string().c_str());
    std::printf("Auth   : %s\n\n", options.source.token.empty() ? "anonymous" : "token");

    ProgressPrinter printer;
    const auto result = repack::installFromRemote(
            args.output, options, [&](const repack::Progress& p) { return printer(p); });
    printer.finishLine();

    if (!result) {
        std::fprintf(stderr, "\ninstall failed: %s\n", result.error().toString().c_str());
        if (result.error().code() == ErrorCode::Network) {
            std::fprintf(stderr,
                         "The partial install was kept. Run the same command again to resume "
                         "from where it stopped.\n");
        }
        return 1;
    }

    std::printf("\nInstalled to %s\n", result->installDir.string().c_str());
    std::printf("  written    : %s\n", formatBytes(result->bytesWritten).c_str());
    std::printf("  downloaded : %s\n", formatBytes(result->bytesDownloaded).c_str());
    if (result->bytesResumed > 0) {
        std::printf("  resumed    : %s was already on disk\n",
                    formatBytes(result->bytesResumed).c_str());
    }
    if (result->reconnects > 0) {
        std::printf("  reconnects : %u\n", result->reconnects);
    }
    std::printf("  elapsed    : %s\n", formatDuration(result->elapsedSeconds).c_str());
    if (result->elapsedSeconds > 0.0) {
        const double throughput = static_cast<double>(result->bytesDownloaded) /
                                  result->elapsedSeconds / (1024.0 * 1024.0);
        std::printf("  average    : %.1f MiB/s\n", throughput);
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc == 1) {
        printUsage();
        return 1;
    }

    const auto args = parseArguments(argc, argv);
    if (!args) {
        std::fprintf(stderr, "%s\n\n", args.error().toString().c_str());
        printUsage();
        return 2;
    }

    if (!args->verifyInstall.empty()) {
        return runVerify(*args);
    }
    if (args->install) {
        return runInstall(*args);
    }
    return runRepack(*args);
}
