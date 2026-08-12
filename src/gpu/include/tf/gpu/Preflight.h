#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "tf/core/base/Error.h"
#include "tf/core/base/Types.h"
#include "tf/gpu/Backend.h"

/// Environment checks run before the engine tries to start.
///
/// The point is that a missing dependency produces a sentence telling the user
/// what to install, not a loader failure or a bare CUDA error code deep in a
/// call stack. The CLI prints this as a table, and the GUI shows it as a
/// blocking panel; both consume the same report.
namespace tf::gpu {

enum class CheckStatus {
    /// Requirement met.
    Ok,
    /// Usable, but degraded, slower, or close to a limit.
    Warning,
    /// The engine cannot run until this is resolved.
    Failed,
};

[[nodiscard]] std::string_view toString(CheckStatus status) noexcept;

struct Check {
    std::string name;
    CheckStatus status = CheckStatus::Ok;
    /// What was actually found.
    std::string detail;
    /// What to do about it. Empty when the check passed.
    std::string remediation;
};

struct PreflightReport {
    std::vector<Check> checks;

    [[nodiscard]] bool canRun() const;
    [[nodiscard]] bool hasWarnings() const;

    /// Aligned plain-text table for console output.
    [[nodiscard]] std::string format() const;
};

struct PreflightOptions {
    /// A .gturbo install to validate. Skipped when empty.
    std::filesystem::path installDir;

    /// Device memory the configuration is expected to need. Defaults to the
    /// resident core plus a 4K KV cache plus a modest expert slot budget, which
    /// is roughly what an 8 GB card runs.
    u64 requiredDeviceBytes = 3ull * 1024 * 1024 * 1024;

    /// Minimum compute capability. sm_70 is where the warp intrinsics the
    /// ported kernels rely on became uniformly available.
    u32 minimumArchitectureMajor = 7;

    u32 deviceIndex = 0;
};

/// Runs every check. Never fails as a whole: a problem is reported as a failed
/// check with an explanation, which is the entire point.
[[nodiscard]] PreflightReport runPreflight(const PreflightOptions& options = {});

}  // namespace tf::gpu
