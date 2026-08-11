#include "tf/gpu/Preflight.h"

#include <algorithm>
#include <format>

#include "tf/core/format/GTurbo.h"
#include "tf/core/io/Volume.h"

namespace tf::gpu {
namespace {

std::string gibibytes(u64 bytes) {
    return std::format("{:.2f} GiB", static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0));
}

Check pass(std::string name, std::string detail) {
    return Check{.name = std::move(name),
                 .status = CheckStatus::Ok,
                 .detail = std::move(detail),
                 .remediation = {}};
}

Check warn(std::string name, std::string detail, std::string remediation) {
    return Check{.name = std::move(name),
                 .status = CheckStatus::Warning,
                 .detail = std::move(detail),
                 .remediation = std::move(remediation)};
}

Check fail(std::string name, std::string detail, std::string remediation) {
    return Check{.name = std::move(name),
                 .status = CheckStatus::Failed,
                 .detail = std::move(detail),
                 .remediation = std::move(remediation)};
}

/// Checks the build can construct a backend at all, before any device is asked
/// about. A CUDA-less build is a configuration mistake, not a driver problem,
/// and deserves a different message.
void checkBuild(PreflightReport& report) {
    const auto backends = compiledBackends();
    if (backends.empty()) {
        report.checks.push_back(fail(
                "GPU backend compiled", "none",
                "This build was configured without a GPU backend. Reconfigure with "
                "-DTF_ENABLE_CUDA=ON and a CUDA Toolkit installed."));
        return;
    }

    std::string names;
    for (const auto kind : backends) {
        if (!names.empty()) {
            names += ", ";
        }
        names += toString(kind);
    }
    report.checks.push_back(pass("GPU backend compiled", names));
}

/// Attempts to bring up the device. This is where a missing or outdated NVIDIA
/// driver surfaces, and the error is turned into an instruction.
BackendPtr checkDevice(PreflightReport& report, const PreflightOptions& options) {
    const auto backends = compiledBackends();
    if (backends.empty()) {
        return nullptr;
    }

    auto backend = createBackend(backends.front(), options.deviceIndex);
    if (!backend) {
        const Error& error = backend.error();
        std::string remediation =
                "Install or update the NVIDIA driver, and confirm a CUDA-capable GPU is "
                "present and not disabled in Device Manager.";
        if (error.code() == ErrorCode::Unsupported) {
            remediation =
                    "No usable CUDA device was found. Check that an NVIDIA GPU is present "
                    "and that its driver is installed and current.";
        }
        report.checks.push_back(fail("GPU device", error.message(), std::move(remediation)));
        return nullptr;
    }

    const DeviceInfo& info = (*backend)->info();
    report.checks.push_back(pass(
            "GPU device",
            std::format("{} (sm_{}{}, {} SMs, {})", info.name, info.architectureMajor,
                        info.architectureMinor, info.multiprocessorCount,
                        gibibytes(info.totalMemoryBytes))));

    if (info.architectureMajor < options.minimumArchitectureMajor) {
        report.checks.push_back(fail(
                "Compute capability",
                std::format("sm_{}{} is below the required sm_{}0", info.architectureMajor,
                            info.architectureMinor, options.minimumArchitectureMajor),
                std::format("This engine needs compute capability {}.0 or newer for the warp "
                            "intrinsics its kernels use.",
                            options.minimumArchitectureMajor)));
    } else {
        report.checks.push_back(pass(
                "Compute capability",
                std::format("sm_{}{}", info.architectureMajor, info.architectureMinor)));
    }

    if (info.warpSize != 32) {
        report.checks.push_back(fail(
                "Warp size", std::format("{}", info.warpSize),
                "The kernels are ported from Metal simdgroups and assume 32 lanes."));
    }

    return std::move(*backend);
}

void checkMemory(PreflightReport& report, IGpuBackend& backend,
                 const PreflightOptions& options) {
    const auto memory = backend.memoryInfo();
    if (!memory) {
        report.checks.push_back(warn("Device memory", memory.error().message(),
                                     "Could not query free VRAM; the residency planner will "
                                     "fall back to the reported total."));
        return;
    }

    const std::string detail =
            std::format("{} free of {}", gibibytes(memory->freeBytes),
                        gibibytes(memory->totalBytes));

    if (memory->freeBytes < options.requiredDeviceBytes) {
        report.checks.push_back(fail(
                "Device memory", detail,
                std::format("At least {} of free VRAM is needed. Close other GPU "
                            "applications, or lower --vram-budget to stream more layers.",
                            gibibytes(options.requiredDeviceBytes))));
        return;
    }

    report.checks.push_back(pass("Device memory", detail));

    // A desktop session typically holds around 1.1 GiB. Flag the case where the
    // budget only just fits, since a browser opening later would break it.
    const u64 headroom = memory->freeBytes - options.requiredDeviceBytes;
    if (headroom < 512ull * 1024 * 1024) {
        report.checks.push_back(warn(
                "Device memory headroom", gibibytes(headroom),
                "Little VRAM is left beyond the configured budget. Another GPU application "
                "starting later may cause allocation failures mid-run."));
    }
}

void checkAsyncEngines(PreflightReport& report, const DeviceInfo& info) {
    if (info.asyncEngineCount == 0) {
        report.checks.push_back(warn(
                "Copy engines", "0",
                "This device cannot overlap transfers with compute, so expert streaming "
                "will serialize against the kernels and throughput will suffer."));
    } else {
        report.checks.push_back(pass(
                "Copy engines",
                std::format("{} (transfers overlap compute)", info.asyncEngineCount)));
    }
}

void checkInstall(PreflightReport& report, const PreflightOptions& options) {
    if (options.installDir.empty()) {
        return;
    }

    if (!std::filesystem::exists(options.installDir)) {
        report.checks.push_back(fail(
                "Model install", std::format("{} does not exist", options.installDir.string()),
                "Run tf-repack --checkpoint <dir> --output <dir.gturbo> to build an "
                "install."));
        return;
    }

    const auto manifest = gturbo::readManifest(options.installDir);
    if (!manifest) {
        const bool partial = manifest.error().code() == ErrorCode::IncompleteInstall;
        report.checks.push_back(fail(
                "Model install", manifest.error().message(),
                partial ? "The install is missing its manifest, so it is incomplete. Re-run "
                          "tf-repack with --overwrite."
                        : "The manifest could not be read. Re-run tf-repack with "
                          "--overwrite, or verify with tf-repack --verify-install."));
        return;
    }

    report.checks.push_back(pass(
            "Model install",
            std::format("{} layers, {} experts, {} resident tensors",
                        manifest->arch.numLayers, manifest->arch.numExperts,
                        manifest->resident.tensors().size())));

    // Where the install lives decides whether streaming works at all.
    const auto volume = io::queryVolume(options.installDir);
    if (!volume) {
        return;
    }

    switch (volume->storage) {
        case io::StorageKind::Rotational:
            report.checks.push_back(warn(
                    "Install volume",
                    std::format("{} is a rotational disk", volume->rootPath),
                    "Expert streaming issues many scattered reads per token, which a "
                    "spinning disk serves at a fraction of SSD speed. Move the install to "
                    "an NVMe or SATA SSD."));
            break;
        case io::StorageKind::Unknown:
            report.checks.push_back(warn(
                    "Install volume",
                    std::format("{} storage type could not be determined", volume->rootPath),
                    "If this is a network share, expert streaming will be far slower than "
                    "on local NVMe."));
            break;
        case io::StorageKind::SolidState:
            report.checks.push_back(pass(
                    "Install volume",
                    std::format("{} solid-state, {} free", volume->rootPath,
                                gibibytes(volume->freeBytes))));
            break;
    }

    // Unbuffered reads need every expert blob to start on a sector boundary.
    if (volume->sectorSize > 0 && !manifest->experts.supportsUnbufferedReads(volume->sectorSize)) {
        report.checks.push_back(warn(
                "Unbuffered reads",
                std::format("expert stride {} is not aligned to the {}-byte sector",
                            manifest->experts.stride, volume->sectorSize),
                "Expert streaming will use buffered reads. This is correct but copies "
                "through the page cache."));
    }
}

}  // namespace

std::string_view toString(CheckStatus status) noexcept {
    switch (status) {
        case CheckStatus::Ok:      return "ok";
        case CheckStatus::Warning: return "warn";
        case CheckStatus::Failed:  return "FAIL";
    }
    return "?";
}

bool PreflightReport::canRun() const {
    return std::ranges::none_of(
            checks, [](const Check& check) { return check.status == CheckStatus::Failed; });
}

bool PreflightReport::hasWarnings() const {
    return std::ranges::any_of(
            checks, [](const Check& check) { return check.status == CheckStatus::Warning; });
}

std::string PreflightReport::format() const {
    usize nameWidth = 0;
    for (const auto& check : checks) {
        nameWidth = std::max(nameWidth, check.name.size());
    }

    std::string out;
    for (const auto& check : checks) {
        out += std::format("  [{:>4}] {:<{}}  {}\n", toString(check.status), check.name,
                           nameWidth, check.detail);
        if (!check.remediation.empty()) {
            // Indent continuation under the detail column so the advice reads
            // as belonging to the line above it.
            out += std::format("         {:<{}}  -> {}\n", "", nameWidth, check.remediation);
        }
    }
    return out;
}

PreflightReport runPreflight(const PreflightOptions& options) {
    PreflightReport report;

    checkBuild(report);

    BackendPtr backend = checkDevice(report, options);
    if (backend) {
        checkMemory(report, *backend, options);
        checkAsyncEngines(report, backend->info());
    }

    checkInstall(report, options);

    return report;
}

}  // namespace tf::gpu
