#pragma once

#include <filesystem>
#include <string>

#include "tf/core/base/Error.h"
#include "tf/core/base/Types.h"
#include "tf/repack/Remote.h"
#include "tf/repack/Repacker.h"

namespace tf::repack {

struct InstallOptions {
    RemoteSource source;
    RetryPolicy retry;

    /// Replace an existing install at the destination.
    bool overwrite = false;

    /// Continue a previous attempt from where it stopped, if the partial
    /// directory holds state for the same plan. On by default: a 14 GB download
    /// that has to restart from zero after a dropped connection is not usable.
    bool resume = true;

    /// Verify every written file's SHA-256 before promoting.
    bool verifyAfterWrite = true;

    PlanOptions plan;
};

struct InstallResult {
    std::filesystem::path installDir;
    u64 bytesWritten = 0;
    u64 bytesDownloaded = 0;
    /// Bytes that were already on disk from an earlier attempt.
    u64 bytesResumed = 0;
    u32 reconnects = 0;
    double elapsedSeconds = 0.0;
};

/// Installs a checkpoint directly from Hugging Face, without ever holding the
/// source on disk.
///
/// The repacker's operations are sorted by (shard, source offset), so a shard
/// is consumed strictly forward and one range request covers all of it. The
/// 14.3 GiB of output is written as it arrives; the 15 GB of source is never
/// materialized. That is the whole point - a machine with 20 GB free can
/// install a model it could not otherwise download and convert.
///
/// Writes into `<output>.partial` and renames on success, so an interrupted run
/// can never leave a directory that looks complete. Progress is checkpointed as
/// it goes, so a later run resumes rather than starting over.
[[nodiscard]] Result<InstallResult> installFromRemote(const std::filesystem::path& outputDir,
                                                      const InstallOptions& options = {},
                                                      const ProgressCallback& progress = {});

/// Resume state written into the partial directory.
///
/// Exposed for testing: the interesting property is that a state file must
/// never let a run resume against a plan that would put different bytes in
/// different places.
struct InstallState {
    std::string repoId;
    std::string revision;
    /// Identifies the plan's shape. A change to the alignment, the operation
    /// split or the source revision produces a different value, and a resume
    /// against a mismatched fingerprint starts over instead of interleaving two
    /// layouts into one file.
    std::string planFingerprint;
    /// Operations known to be on disk. The output files persist in the partial
    /// directory, so this is the only thing needed to pick up.
    u64 opsCompleted = 0;
    u64 bytesWritten = 0;

    [[nodiscard]] std::string toJson() const;
    [[nodiscard]] static Result<InstallState> parse(std::string_view json);
};

/// A fingerprint of everything that determines where a byte lands.
[[nodiscard]] std::string planFingerprint(const RepackPlan& plan, const RemoteSource& source);

/// Prevents two installers from writing the same partial directory.
///
/// Opened with no sharing, so a second process fails immediately with a clear
/// message rather than the two interleaving writes and producing an install
/// that passes its own hash check but is internally inconsistent.
class InstallLock {
public:
    InstallLock() = default;
    ~InstallLock();

    InstallLock(const InstallLock&) = delete;
    InstallLock& operator=(const InstallLock&) = delete;
    InstallLock(InstallLock&&) noexcept;
    InstallLock& operator=(InstallLock&&) noexcept;

    [[nodiscard]] static Result<InstallLock> acquire(const std::filesystem::path& directory);

    void release();

private:
    void* handle_ = nullptr;  // HANDLE
    std::filesystem::path path_;
};

}  // namespace tf::repack
