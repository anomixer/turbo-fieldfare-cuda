#pragma once

#include <filesystem>
#include <span>
#include <string>
#include <vector>

#include "tf/core/base/Error.h"
#include "tf/core/base/Types.h"
#include "tf/runtime/Generator.h"

namespace tf::cli {

/// What the user asked for.
///
/// Parsed as a value with no side effects so it can be tested directly:
/// argument handling is where a CLI accumulates quiet bugs, and "--tokens" with
/// a missing value should produce a message rather than a crash or a zero.
struct Args {
    std::filesystem::path modelDir;

    /// The prompt, or empty when reading from stdin.
    std::string prompt;
    /// Read the prompt from stdin instead of the command line.
    bool promptFromStdin = false;

    /// Wrap the prompt in the instruction template. On by default: this is an
    /// instruction-tuned checkpoint, and raw completion of a bare question is
    /// out of distribution for it.
    bool chat = true;
    /// A system message, prepended to the conversation. Empty for none.
    std::string system;

    runtime::SamplingParams sampling;
    u64 maxTokens = 512;
    u64 contextLength = 4096;
    std::vector<std::string> stopStrings;

    /// Which GPU backend to use. Empty takes the first compiled one, which is
    /// CUDA where it is available.
    std::string backend;

    /// Residency and streaming.
    u64 vramBudget = 0;
    /// Zero leaves the residency planner's default, so the number lives in one
    /// place rather than being restated by every front end.
    u32 expertSlots = 0;
    u32 prefillChunk = 128;

    /// Bypass the page cache for expert reads. Faster on a machine whose RAM
    /// cannot hold the expert set, permanently slower on one that can - see
    /// docs/BENCHMARKS.md.
    bool unbufferedReads = false;
    /// Threads issuing expert reads.
    u32 readThreads = 4;

    /// Show the model's thinking channel as well as its answer. Off by
    /// default: it is markup the user did not ask for, and on this checkpoint
    /// it is usually empty.
    bool showThinking = false;

    /// Print timings and the residency plan.
    bool verbose = false;
    /// Suppress everything but the generated text, for piping.
    bool quiet = false;
    /// Print the resolved configuration and exit without loading the model.
    bool dryRun = false;

    bool showHelp = false;
    bool showVersion = false;

    [[nodiscard]] static Result<Args> parse(std::span<const std::string_view> arguments);

    /// Checks the combinations that individual flags cannot: a prompt that is
    /// both given and read from stdin, a chunk above the context, and so on.
    [[nodiscard]] Status validate() const;
};

[[nodiscard]] std::string usage();

}  // namespace tf::cli
