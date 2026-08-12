#include "tf/repack/Repacker.h"

#include <algorithm>
#include <chrono>
#include <format>
#include <limits>
#include <system_error>
#include <vector>

#include "tf/core/io/File.h"
#include "tf/core/io/Sha256.h"

namespace tf::repack {
namespace {

/// Sidecars copied verbatim into tokenizer/. chat_template.jinja is optional:
/// the runtime has the Gemma template built in and reads the file only to
/// detect a checkpoint that disagrees.
constexpr std::string_view kRequiredTokenizerFiles[] = {
        "tokenizer.json",
        "tokenizer_config.json",
};
constexpr std::string_view kOptionalTokenizerFiles[] = {
        "chat_template.jinja",
        "generation_config.json",
        "special_tokens_map.json",
};

bool report(const ProgressCallback& callback, Phase phase, u64 done, u64 total,
            std::string detail) {
    if (!callback) {
        return true;
    }
    return callback(Progress{
            .phase = phase, .bytesDone = done, .bytesTotal = total, .detail = std::move(detail)});
}

/// Output files stay open for the whole run: 31 handles against tens of
/// thousands of scattered writes.
class OutputFiles {
public:
    [[nodiscard]] Status open(const std::filesystem::path& root, const RepackPlan& plan) {
        TF_TRY(io::File resident,
               io::File::create(root / gturbo::kResidentFile, /*overwrite=*/true));
        TF_CHECK(resident.preallocate(plan.residentBytes()));
        resident_ = std::move(resident);

        const std::filesystem::path expertsDir = root / gturbo::kExpertsDir;
        std::error_code ec;
        std::filesystem::create_directories(expertsDir, ec);
        if (ec) {
            return makeError(ErrorCode::Io, "creating {}: {}", expertsDir.string(),
                             ec.message());
        }

        layers_.reserve(plan.experts.numLayers);
        for (u64 layer = 0; layer < plan.experts.numLayers; ++layer) {
            TF_TRY(io::File file,
                   io::File::create(expertsDir / plan.experts.layerFiles[layer],
                                    /*overwrite=*/true));
            // Preallocating also zero-fills the inter-blob stride padding, so
            // the pad never contains recycled disk contents.
            TF_CHECK(file.preallocate(plan.experts.layerFileBytes()));
            layers_.push_back(std::move(file));
        }
        return {};
    }

    [[nodiscard]] Result<io::File*> forOp(const CopyOp& op) {
        if (op.destKind == DestKind::Resident) {
            return &resident_;
        }
        if (op.destLayer >= layers_.size()) {
            return makeError(ErrorCode::InvalidArgument, "no output file for layer {}",
                             op.destLayer);
        }
        return &layers_[op.destLayer];
    }

    [[nodiscard]] Status flushAll() {
        TF_CHECK(resident_.flush());
        for (auto& layer : layers_) {
            TF_CHECK(layer.flush());
        }
        return {};
    }

    void closeAll() {
        resident_.close();
        for (auto& layer : layers_) {
            layer.close();
        }
    }

private:
    io::File resident_;
    std::vector<io::File> layers_;
};

Status copyTokenizerFiles(const std::filesystem::path& checkpointDir,
                          const std::filesystem::path& root, std::vector<std::string>& copied,
                          const ProgressCallback& progress) {
    const std::filesystem::path tokenizerDir = root / gturbo::kTokenizerDir;
    std::error_code ec;
    std::filesystem::create_directories(tokenizerDir, ec);
    if (ec) {
        return makeError(ErrorCode::Io, "creating {}: {}", tokenizerDir.string(), ec.message());
    }

    const auto copyOne = [&](std::string_view name, bool required) -> Status {
        const std::filesystem::path source = checkpointDir / name;
        if (!std::filesystem::exists(source)) {
            if (required) {
                return makeError(ErrorCode::NotFound,
                                 "{} is missing from the checkpoint and is required", name);
            }
            return {};
        }

        if (!report(progress, Phase::CopyingTokenizer, 0, 0, std::string{name})) {
            return makeError(ErrorCode::Cancelled, "cancelled while copying {}", name);
        }

        // tokenizer.json is 30 MB, comfortably a whole-file read.
        TF_TRY(const std::vector<u8> contents, io::readBinaryFile(source));
        TF_CHECK(io::writeFileAtomic(tokenizerDir / name, contents));
        copied.push_back(std::format("{}/{}", gturbo::kTokenizerDir, name));
        return {};
    };

    for (const auto& name : kRequiredTokenizerFiles) {
        TF_CHECK(copyOne(name, /*required=*/true));
    }
    for (const auto& name : kOptionalTokenizerFiles) {
        TF_CHECK(copyOne(name, /*required=*/false));
    }
    return {};
}

Status removeAll(const std::filesystem::path& path) {
    std::error_code ec;
    std::filesystem::remove_all(path, ec);
    if (ec) {
        return makeError(ErrorCode::Io, "removing {}: {}", path.string(), ec.message());
    }
    return {};
}

}  // namespace

std::string_view toString(Phase phase) noexcept {
    switch (phase) {
        case Phase::Planning:         return "planning";
        case Phase::CopyingWeights:   return "copying weights";
        case Phase::CopyingTokenizer: return "copying tokenizer";
        case Phase::Hashing:          return "hashing";
        case Phase::WritingManifest:  return "writing manifest";
        case Phase::Promoting:        return "promoting";
    }
    return "?";
}

Result<RepackResult> repackFromCheckpoint(const std::filesystem::path& checkpointDir,
                                          const std::filesystem::path& outputDir,
                                          const RepackOptions& options,
                                          const ProgressCallback& progress) {
    const auto started = std::chrono::steady_clock::now();

    if (!std::filesystem::exists(checkpointDir)) {
        return makeError(ErrorCode::NotFound, "checkpoint directory {} does not exist",
                         checkpointDir.string());
    }
    if (std::filesystem::exists(outputDir)) {
        if (!options.overwrite) {
            return makeError(ErrorCode::InvalidArgument,
                             "{} already exists; pass --overwrite to replace it",
                             outputDir.string());
        }
    }

    // ---- Plan ------------------------------------------------------------
    if (!report(progress, Phase::Planning, 0, 0, "reading checkpoint metadata")) {
        return makeError(ErrorCode::Cancelled, "cancelled during planning");
    }

    TF_TRY(const ArchInfo arch, ArchInfo::readFromCheckpoint(checkpointDir));
    TF_TRY(const SafetensorsIndex index, readSafetensorsIndex(checkpointDir));

    std::vector<SafetensorsHeader> headers;
    headers.reserve(index.shardFiles().size());
    for (const auto& shard : index.shardFiles()) {
        TF_TRY(SafetensorsHeader header, readSafetensorsHeader(checkpointDir / shard));
        headers.push_back(std::move(header));
    }

    TF_TRY(const RepackPlan plan, buildPlan(arch, index, headers, options.plan));

    // ---- Space check -----------------------------------------------------
    const std::filesystem::path partialDir = outputDir.string() + ".partial";
    TF_TRY(const u64 available, io::availableSpace(partialDir));
    const u64 needed = plan.totalOutputBytes() + 64ull * 1024 * 1024;  // headroom for sidecars
    if (available < needed) {
        return makeError(ErrorCode::OutOfDiskSpace,
                         "{} needs {:.1f} GiB but only {:.1f} GiB is free",
                         partialDir.string(),
                         static_cast<double>(needed) / (1024.0 * 1024 * 1024),
                         static_cast<double>(available) / (1024.0 * 1024 * 1024));
    }

    // A leftover partial from an earlier run is discarded rather than reused:
    // resumable installs are M1b's job, and a stale directory here would be
    // indistinguishable from a good one.
    TF_CHECK(removeAll(partialDir));
    std::error_code ec;
    std::filesystem::create_directories(partialDir, ec);
    if (ec) {
        return makeError(ErrorCode::Io, "creating {}: {}", partialDir.string(), ec.message());
    }

    // ---- Copy weights ----------------------------------------------------
    {
        OutputFiles outputs;
        TF_CHECK(outputs.open(partialDir, plan));

        std::vector<io::File> shards;
        shards.reserve(plan.shards.size());
        for (const auto& shard : plan.shards) {
            TF_TRY(io::File file,
                   io::File::openRead(checkpointDir / shard, io::ReadMode::Sequential));
            shards.push_back(std::move(file));
        }

        // One fixed scratch buffer: operations were pre-split to maxOpBytes so
        // this never needs to grow, even for the 369 MB embedding table.
        std::vector<u8> scratch(static_cast<usize>(options.plan.maxOpBytes));

        const u64 totalBytes = plan.totalCopiedBytes();
        u64 doneBytes = 0;
        u32 currentShard = std::numeric_limits<u32>::max();

        for (const auto& op : plan.ops) {
            if (op.shardIndex != currentShard) {
                currentShard = op.shardIndex;
                if (!report(progress, Phase::CopyingWeights, doneBytes, totalBytes,
                            plan.shards[currentShard])) {
                    return makeError(ErrorCode::Cancelled, "cancelled while copying weights");
                }
            }

            const MutableByteSpan buffer{scratch.data(), static_cast<usize>(op.source.length)};
            TF_CHECK(shards[op.shardIndex].readExactAt(op.source.offset, buffer));

            TF_TRY(io::File* destination, outputs.forOp(op));
            TF_CHECK(destination->writeAt(op.destOffset, buffer));

            doneBytes += op.source.length;

            // Reporting every operation would be ~35k callbacks; every 64 MiB
            // is frequent enough for a progress bar and cheap enough to ignore.
            if ((doneBytes & (64ull * 1024 * 1024 - 1)) < op.source.length) {
                if (!report(progress, Phase::CopyingWeights, doneBytes, totalBytes,
                            plan.shards[op.shardIndex])) {
                    return makeError(ErrorCode::Cancelled, "cancelled while copying weights");
                }
            }
        }

        TF_CHECK(outputs.flushAll());
        outputs.closeAll();
        report(progress, Phase::CopyingWeights, totalBytes, totalBytes, "done");
    }

    // ---- Sidecars --------------------------------------------------------
    std::vector<std::string> tokenizerFiles;
    TF_CHECK(copyTokenizerFiles(checkpointDir, partialDir, tokenizerFiles, progress));

    const std::string layoutJson = plan.experts.toJson();
    TF_CHECK(io::writeFileAtomic(
            partialDir / gturbo::kExpertsDir / gturbo::kExpertLayoutFile,
            ByteSpan{reinterpret_cast<const u8*>(layoutJson.data()), layoutJson.size()}));

    // ---- Hash every file -------------------------------------------------
    gturbo::Manifest manifest;
    manifest.arch = plan.arch;
    manifest.resident = plan.resident;
    manifest.experts = plan.experts;
    manifest.source = gturbo::SourceInfo{.repoId = options.source.repoId,
                                         .revision = options.source.revision,
                                         .toolVersion = "tf-repack 0.1.0"};

    std::vector<std::string> hashTargets;
    hashTargets.emplace_back(gturbo::kResidentFile);
    for (u64 layer = 0; layer < plan.experts.numLayers; ++layer) {
        hashTargets.push_back(
                std::format("{}/{}", gturbo::kExpertsDir, plan.experts.layerFiles[layer]));
    }
    hashTargets.push_back(std::format("{}/{}", gturbo::kExpertsDir, gturbo::kExpertLayoutFile));
    for (const auto& tokenizer : tokenizerFiles) {
        hashTargets.push_back(tokenizer);
    }

    if (options.verifyAfterWrite) {
        const u64 totalToHash = plan.totalOutputBytes();
        u64 hashed = 0;
        for (const auto& relative : hashTargets) {
            const std::filesystem::path path = partialDir / relative;
            if (!report(progress, Phase::Hashing, hashed, totalToHash, relative)) {
                return makeError(ErrorCode::Cancelled, "cancelled while hashing");
            }

            TF_TRY(const std::string digest, io::Sha256::hashFile(path));
            const auto size = std::filesystem::file_size(path, ec);
            if (ec) {
                return makeError(ErrorCode::Io, "{}: {}", relative, ec.message());
            }

            manifest.files.push_back(gturbo::FileDigest{
                    .path = relative, .size = size, .sha256 = digest});
            hashed += size;
        }
        report(progress, Phase::Hashing, totalToHash, totalToHash, "done");
    } else {
        // Sizes are still recorded so a truncated file is caught even when the
        // caller opted out of hashing.
        for (const auto& relative : hashTargets) {
            const auto size = std::filesystem::file_size(partialDir / relative, ec);
            if (ec) {
                return makeError(ErrorCode::Io, "{}: {}", relative, ec.message());
            }
            manifest.files.push_back(
                    gturbo::FileDigest{.path = relative, .size = size, .sha256 = {}});
        }
    }

    TF_CHECK(manifest.validate());

    // ---- Manifest, written last -----------------------------------------
    report(progress, Phase::WritingManifest, 0, 0, std::string{gturbo::kManifestFile});
    const std::string manifestJson = manifest.toJson();
    TF_CHECK(io::writeFileAtomic(
            partialDir / gturbo::kManifestFile,
            ByteSpan{reinterpret_cast<const u8*>(manifestJson.data()), manifestJson.size()}));

    // ---- Promote ---------------------------------------------------------
    report(progress, Phase::Promoting, 0, 0, outputDir.string());
    if (std::filesystem::exists(outputDir)) {
        TF_CHECK(removeAll(outputDir));
    }
    std::filesystem::rename(partialDir, outputDir, ec);
    if (ec) {
        return makeError(ErrorCode::Io, "renaming {} to {}: {}", partialDir.string(),
                         outputDir.string(), ec.message());
    }

    const auto elapsed = std::chrono::steady_clock::now() - started;
    return RepackResult{
            .installDir = outputDir,
            .bytesWritten = plan.totalOutputBytes(),
            .residentBytes = plan.residentBytes(),
            .expertBytes = plan.expertBytes(),
            .elapsedSeconds = std::chrono::duration<double>(elapsed).count()};
}

Status verifyInstalled(const std::filesystem::path& installDir,
                       const ProgressCallback& progress) {
    TF_TRY(const gturbo::Manifest manifest, gturbo::readManifest(installDir));

    u64 total = 0;
    for (const auto& file : manifest.files) {
        total += file.size;
    }

    // Hashed here rather than delegating to gturbo::verifyInstall so progress
    // advances with the work instead of jumping at the end.
    u64 done = 0;
    for (const auto& file : manifest.files) {
        if (!report(progress, Phase::Hashing, done, total, file.path)) {
            return makeError(ErrorCode::Cancelled, "verification cancelled");
        }

        const std::filesystem::path path = installDir / file.path;
        std::error_code ec;
        if (!std::filesystem::exists(path, ec)) {
            return makeError(ErrorCode::VerificationFailed, "{} is missing", file.path);
        }

        const auto actualSize = std::filesystem::file_size(path, ec);
        if (ec) {
            return makeError(ErrorCode::Io, "{}: {}", file.path, ec.message());
        }
        if (actualSize != file.size) {
            return makeError(ErrorCode::VerificationFailed, "{} is {} bytes, expected {}",
                             file.path, actualSize, file.size);
        }

        // An empty digest means the install was written with hashing disabled,
        // so size is all there is to check.
        if (!file.sha256.empty()) {
            TF_TRY(const std::string digest, io::Sha256::hashFile(path));
            if (digest != file.sha256) {
                return makeError(ErrorCode::VerificationFailed,
                                 "{} hashes to {} but the manifest records {}", file.path,
                                 digest, file.sha256);
            }
        }

        done += file.size;
    }

    report(progress, Phase::Hashing, total, total, "done");
    return {};
}

}  // namespace tf::repack
