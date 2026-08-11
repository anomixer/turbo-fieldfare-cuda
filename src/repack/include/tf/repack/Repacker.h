#pragma once

#include <filesystem>
#include <functional>
#include <string>

#include "tf/core/base/Error.h"
#include "tf/core/base/Types.h"
#include "tf/repack/RepackPlan.h"

namespace tf::repack {

enum class Phase {
    Planning,
    CopyingWeights,
    CopyingTokenizer,
    Hashing,
    WritingManifest,
    Promoting,
};

[[nodiscard]] std::string_view toString(Phase phase) noexcept;

struct Progress {
    Phase phase = Phase::Planning;
    u64 bytesDone = 0;
    u64 bytesTotal = 0;
    /// Currently active file or step, for display.
    std::string detail;
};

/// Return false to cancel. Cancellation leaves the partial directory in place
/// so a later run can resume rather than starting over.
using ProgressCallback = std::function<bool(const Progress&)>;

/// Provenance recorded in the manifest. Defaults name the checkpoint upstream
/// pins, since that is what a local download will almost always be.
struct SourceInfoOverride {
    std::string repoId = "mlx-community/gemma-4-26b-a4b-it-4bit";
    std::string revision = "0d77464eeb233a2da68ebf9d7dc4edaac7db956d";
};

struct RepackOptions {
    /// Replace an existing install at the destination.
    bool overwrite = false;
    /// Verify every written file's SHA-256 before promoting. Costs a full
    /// re-read of 14.3 GB; on by default because a silently corrupt install
    /// surfaces much later as unexplainable output.
    bool verifyAfterWrite = true;
    PlanOptions plan;
    SourceInfoOverride source;
};

struct RepackResult {
    std::filesystem::path installDir;
    u64 bytesWritten = 0;
    u64 residentBytes = 0;
    u64 expertBytes = 0;
    double elapsedSeconds = 0.0;
};

/// Builds a .gturbo install from a locally downloaded checkpoint directory.
///
/// Writes into `<output>.partial` and renames on success, so an interrupted run
/// can never leave a directory that looks complete. The manifest is written
/// last for the same reason: its presence is the completion marker.
[[nodiscard]] Result<RepackResult> repackFromCheckpoint(
        const std::filesystem::path& checkpointDir, const std::filesystem::path& outputDir,
        const RepackOptions& options = {}, const ProgressCallback& progress = {});

/// Loads an installed manifest and re-verifies every file against it.
[[nodiscard]] Status verifyInstalled(const std::filesystem::path& installDir,
                                     const ProgressCallback& progress = {});

}  // namespace tf::repack
