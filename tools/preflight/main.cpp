// tf-preflight - reports whether this machine can run the engine, which model
// variant suits it, and what to do about anything missing.

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <string>
#include <string_view>

#include "tf/gpu/Preflight.h"
#include "tf/runtime/ModelCatalog.h"

using namespace tf;

namespace {

void printUsage() {
    std::puts(R"(tf-preflight - check that this machine can run the engine

Usage:
  tf-preflight [--install <dir.gturbo>] [--require-vram <GiB>] [--device <n>]
  tf-preflight --list-models

Options:
  --install <dir>       Also validate a .gturbo install
  --require-vram <GiB>  Device memory the configuration needs, default 3
  --device <n>          GPU index, default 0
  --list-models         Print the model catalog and exit
  --no-recommend        Skip the model recommendation
  -h, --help            This message

Exit codes: 0 all checks passed, 1 a check failed, 2 bad arguments.)");
}

std::filesystem::path defaultInstallDir() {
    if (const char* override = std::getenv("TF_GTURBO_DIR")) {
        return std::filesystem::path{override};
    }
    return {};
}

std::string gib(u64 bytes) {
    return std::format("{:.2f} GiB", static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0));
}

void listModels() {
    std::puts("\nGemma 4 catalog (4-bit MLX checkpoints)\n");
    std::printf("  %-10s %-30s %7s  %-18s %10s  %s\n", "variant", "name", "params",
                "architecture", "download", "runtime");
    for (const auto& variant : runtime::modelCatalog()) {
        std::printf("  %-10s %-30s %6.1fB  %-18s %10s  %s\n", variant.id.c_str(),
                    variant.displayName.c_str(), variant.totalParamsB,
                    std::string{runtime::toString(variant.family)}.c_str(),
                    gib(variant.downloadBytes).c_str(),
                    std::string{runtime::toString(variant.support)}.c_str());
    }
    std::puts(
            "\nOnly the mixture-of-experts variant streams. The dense models must be held\n"
            "entirely in VRAM, which is why a 26B MoE runs on hardware a 31B dense cannot.");
}

/// Fills in the GPU half of the machine profile. detectMachine covers disk and
/// RAM without needing a GPU, so this is the only part that can fail benignly.
runtime::MachineProfile buildProfile(const std::filesystem::path& installPath) {
    auto detected = runtime::detectMachine(installPath);
    runtime::MachineProfile profile =
            detected.has_value() ? *detected : runtime::MachineProfile{};

    const auto backends = gpu::compiledBackends();
    if (!backends.empty()) {
        if (auto backend = gpu::createBackend(backends.front()); backend.has_value()) {
            profile.hasGpu = true;
            if (const auto memory = (*backend)->memoryInfo(); memory.has_value()) {
                profile.deviceBytes = memory->freeBytes;
            } else {
                profile.deviceBytes = (*backend)->info().totalMemoryBytes;
            }
        }
    }
    return profile;
}

}  // namespace

int main(int argc, char** argv) {
    gpu::PreflightOptions options;
    options.installDir = defaultInstallDir();
    bool recommend = true;

    for (int i = 1; i < argc; ++i) {
        const std::string_view flag = argv[i];

        const auto value = [&](std::string_view name) -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "%.*s needs a value\n", static_cast<int>(name.size()),
                             name.data());
                std::exit(2);
            }
            return argv[++i];
        };

        if (flag == "-h" || flag == "--help") {
            printUsage();
            return 0;
        }
        if (flag == "--list-models") {
            listModels();
            return 0;
        }
        if (flag == "--no-recommend") {
            recommend = false;
        } else if (flag == "--install") {
            options.installDir = value(flag);
        } else if (flag == "--require-vram") {
            options.requiredDeviceBytes = static_cast<u64>(
                    std::strtod(value(flag), nullptr) * 1024.0 * 1024.0 * 1024.0);
        } else if (flag == "--device") {
            options.deviceIndex = static_cast<u32>(std::strtoul(value(flag), nullptr, 10));
        } else {
            std::fprintf(stderr, "unknown argument '%.*s'\n\n",
                         static_cast<int>(flag.size()), flag.data());
            printUsage();
            return 2;
        }
    }

    const gpu::PreflightReport report = gpu::runPreflight(options);
    std::printf("\nturbofieldfare preflight\n\n%s", report.format().c_str());

    if (recommend) {
        const std::filesystem::path probe = options.installDir.empty()
                                                    ? std::filesystem::current_path()
                                                    : options.installDir;
        const runtime::MachineProfile machine = buildProfile(probe);

        std::printf("\nMachine: %s VRAM free, %s RAM, %s free on the install volume%s\n\n",
                    gib(machine.deviceBytes).c_str(), gib(machine.systemRamBytes).c_str(),
                    gib(machine.diskFreeBytes).c_str(),
                    machine.diskIsSolidState ? "" : " (spinning disk)");

        const runtime::Recommendation recommendation = runtime::recommendModel(machine);
        std::printf("%s", recommendation.format().c_str());
    }

    if (!report.canRun()) {
        std::printf("\nNot ready: resolve the failures above.\n");
        return 1;
    }
    if (report.hasWarnings()) {
        std::printf("\nReady, with warnings.\n");
        return 0;
    }
    std::printf("\nReady.\n");
    return 0;
}
