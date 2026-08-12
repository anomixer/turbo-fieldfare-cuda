#include "tf/repack/Installer.h"

#include <windows.h>

#include <algorithm>
#include <chrono>
#include <format>
#include <system_error>
#include <vector>

#include "tf/core/io/File.h"
#include "tf/core/io/Sha256.h"
#include "tf/core/io/Volume.h"
#include "tf/core/json/Json.h"

namespace tf::repack {
namespace {

constexpr std::string_view kStateFile = "install-state.json";
constexpr std::string_view kLockFile = "install.lock";

constexpr std::string_view kConfigFile = "config.json";
constexpr std::string_view kIndexFile = "model.safetensors.index.json";

/// Sidecars pulled into tokenizer/. Mirrors the local repacker's list.
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

Status removeAll(const std::filesystem::path& path) {
    std::error_code ec;
    std::filesystem::remove_all(path, ec);
    if (ec) {
        return makeError(ErrorCode::Io, "removing {}: {}", path.string(), ec.message());
    }
    return {};
}

/// Reads a shard's safetensors header without downloading the shard.
///
/// Two ranged reads: eight bytes for the length, then the JSON. The headers in
/// this checkpoint are around 90 KB against shards of several gigabytes, so
/// planning costs a fraction of a percent of the transfer.
[[nodiscard]] Result<SafetensorsHeader> fetchShardHeader(net::HttpClient& client,
                                                         const RemoteSource& source,
                                                         std::string_view shard, u64 shardSize,
                                                         const RetryPolicy& retry) {
    TF_TRY(RemoteFileStream stream,
           RemoteFileStream::open(client, source, shard, shardSize, retry));

    std::array<u8, 8> prefix{};
    TF_CHECK(stream.readExactAt(0, MutableByteSpan{prefix.data(), prefix.size()}));
    TF_TRY(const u64 headerLength, SafetensorsHeader::readHeaderLength(prefix));

    if (headerLength > SafetensorsHeader::kMaxHeaderBytes || headerLength + 8 > shardSize) {
        return makeError(ErrorCode::MalformedData,
                         "{} declares a {} byte header, which is not credible for a {} byte "
                         "file",
                         shard, headerLength, shardSize);
    }

    std::vector<u8> headerJson(static_cast<usize>(headerLength));
    TF_CHECK(stream.readExactAt(8, headerJson));
    return SafetensorsHeader::parse(
            std::string_view{reinterpret_cast<const char*>(headerJson.data()),
                             headerJson.size()},
            headerLength);
}

/// Output files stay open for the whole run: 31 handles against tens of
/// thousands of scattered writes.
class OutputFiles {
public:
    /// `fresh` false keeps whatever an earlier attempt wrote, which is what
    /// makes resuming possible - the bytes are already in these files.
    [[nodiscard]] Status open(const std::filesystem::path& root, const RepackPlan& plan,
                              bool fresh) {
        TF_TRY(io::File resident, openOne(root / gturbo::kResidentFile, plan.residentBytes(),
                                          fresh));
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
            TF_TRY(io::File file, openOne(expertsDir / plan.experts.layerFiles[layer],
                                          plan.experts.layerFileBytes(), fresh));
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
    [[nodiscard]] static Result<io::File> openOne(const std::filesystem::path& path, u64 bytes,
                                                  bool fresh) {
        if (!fresh && std::filesystem::exists(path)) {
            // Reopened without truncating: this file already holds most of the
            // install, which is the whole point of resuming. Preallocation
            // happened on the first attempt.
            TF_TRY(io::File existing, io::File::openWrite(path));
            return existing;
        }
        TF_TRY(io::File file, io::File::create(path, /*overwrite=*/true));
        // Preallocating also zero-fills the inter-blob stride padding, so the
        // pad never contains recycled disk contents.
        TF_CHECK(file.preallocate(bytes));
        return file;
    }

    io::File resident_;
    std::vector<io::File> layers_;
};

}  // namespace

// ---------------------------------------------------------------------------
// Resume state
// ---------------------------------------------------------------------------

std::string planFingerprint(const RepackPlan& plan, const RemoteSource& source) {
    // Everything that decides where a byte lands. Left as readable text rather
    // than hashed: it is only ever compared for equality, it is short, and when
    // a resume is refused the state file says why.
    std::string shards;
    for (const auto& shard : plan.shards) {
        if (!shards.empty()) {
            shards += ',';
        }
        shards += shard;
    }
    return std::format("{}@{} shards=[{}] ops={} resident={} experts={} layers={} stride={}",
                       source.repoId, source.revision, shards, plan.ops.size(),
                       plan.residentBytes(), plan.expertBytes(), plan.experts.numLayers,
                       plan.experts.stride);
}

std::string InstallState::toJson() const {
    return std::format(
            "{{\n  \"repoId\": \"{}\",\n  \"revision\": \"{}\",\n  \"planFingerprint\": "
            "\"{}\",\n  \"opsCompleted\": {},\n  \"bytesWritten\": {}\n}}\n",
            repoId, revision, planFingerprint, opsCompleted, bytesWritten);
}

Result<InstallState> InstallState::parse(std::string_view json) {
    TF_TRY(const json::Value root, json::parse(json));

    InstallState state;
    const auto readString = [&](std::string_view key, std::string& target) -> Status {
        TF_TRY(const json::Value* value, root.at(key));
        TF_TRY(const std::string_view text, value->asString());
        target = std::string{text};
        return {};
    };
    const auto readUInt = [&](std::string_view key, u64& target) -> Status {
        TF_TRY(const json::Value* value, root.at(key));
        TF_TRY(target, value->asUInt());
        return {};
    };

    TF_CHECK(readString("repoId", state.repoId));
    TF_CHECK(readString("revision", state.revision));
    TF_CHECK(readString("planFingerprint", state.planFingerprint));
    TF_CHECK(readUInt("opsCompleted", state.opsCompleted));
    TF_CHECK(readUInt("bytesWritten", state.bytesWritten));
    return state;
}

// ---------------------------------------------------------------------------
// Lock
// ---------------------------------------------------------------------------

InstallLock::~InstallLock() { release(); }

InstallLock::InstallLock(InstallLock&& other) noexcept
    : handle_(std::exchange(other.handle_, nullptr)), path_(std::move(other.path_)) {}

InstallLock& InstallLock::operator=(InstallLock&& other) noexcept {
    if (this != &other) {
        release();
        handle_ = std::exchange(other.handle_, nullptr);
        path_ = std::move(other.path_);
    }
    return *this;
}

Result<InstallLock> InstallLock::acquire(const std::filesystem::path& directory) {
    std::error_code ec;
    std::filesystem::create_directories(directory, ec);
    if (ec) {
        return makeError(ErrorCode::Io, "creating {}: {}", directory.string(), ec.message());
    }

    const std::filesystem::path path = directory / kLockFile;
    // No share flags at all: a second opener gets ERROR_SHARING_VIOLATION.
    // FILE_FLAG_DELETE_ON_CLOSE means a crashed installer does not leave a lock
    // behind that a later run cannot clear.
    const HANDLE handle = ::CreateFileW(path.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS,
                                        FILE_ATTRIBUTE_TEMPORARY | FILE_FLAG_DELETE_ON_CLOSE,
                                        nullptr);
    if (handle == INVALID_HANDLE_VALUE) {
        const DWORD code = ::GetLastError();
        if (code == ERROR_SHARING_VIOLATION || code == ERROR_ACCESS_DENIED) {
            return makeError(ErrorCode::Io,
                             "another installer is already writing {}; wait for it or remove "
                             "the directory",
                             directory.string());
        }
        return makeError(ErrorCode::Io, "creating the lock file {}: error {}", path.string(),
                         code);
    }

    InstallLock lock;
    lock.handle_ = handle;
    lock.path_ = path;
    return lock;
}

void InstallLock::release() {
    if (handle_ != nullptr) {
        ::CloseHandle(static_cast<HANDLE>(handle_));
        handle_ = nullptr;
    }
}

// ---------------------------------------------------------------------------
// Install
// ---------------------------------------------------------------------------

Result<InstallResult> installFromRemote(const std::filesystem::path& outputDir,
                                        const InstallOptions& options,
                                        const ProgressCallback& progress) {
    const auto started = std::chrono::steady_clock::now();
    TF_CHECK(options.source.validate());

    if (std::filesystem::exists(outputDir) && !options.overwrite) {
        return makeError(ErrorCode::InvalidArgument,
                         "{} already exists; pass --overwrite to replace it",
                         outputDir.string());
    }

    TF_TRY(net::HttpClient client, net::HttpClient::create());

    // ---- Metadata --------------------------------------------------------
    if (!report(progress, Phase::Planning, 0, 0, "fetching checkpoint metadata")) {
        return makeError(ErrorCode::Cancelled, "cancelled during planning");
    }

    TF_TRY(const std::vector<RemoteFile> remoteFiles,
           listRemoteFiles(client, options.source));

    const auto findRemote = [&](std::string_view name) -> const RemoteFile* {
        const auto found = std::ranges::find(remoteFiles, name, &RemoteFile::name);
        return found == remoteFiles.end() ? nullptr : &*found;
    };

    TF_TRY(const std::vector<u8> configBytes,
           fetchSmallFile(client, options.source, kConfigFile, options.retry));
    TF_TRY(const ArchInfo arch,
           ArchInfo::parseConfigJson(std::string_view{
                   reinterpret_cast<const char*>(configBytes.data()), configBytes.size()}));

    TF_TRY(const std::vector<u8> indexBytes,
           fetchSmallFile(client, options.source, kIndexFile, options.retry));
    TF_TRY(const SafetensorsIndex index,
           SafetensorsIndex::parse(std::string_view{
                   reinterpret_cast<const char*>(indexBytes.data()), indexBytes.size()}));

    // ---- Plan ------------------------------------------------------------
    std::vector<SafetensorsHeader> headers;
    headers.reserve(index.shardFiles().size());
    for (const auto& shard : index.shardFiles()) {
        const RemoteFile* remote = findRemote(shard);
        if (remote == nullptr) {
            return makeError(ErrorCode::NotFound,
                             "the index names {} but the repository does not list it", shard);
        }
        if (!report(progress, Phase::Planning, 0, 0, std::format("header of {}", shard))) {
            return makeError(ErrorCode::Cancelled, "cancelled during planning");
        }
        TF_TRY(SafetensorsHeader header,
               fetchShardHeader(client, options.source, shard, remote->size, options.retry));
        headers.push_back(std::move(header));
    }

    TF_TRY(const RepackPlan plan, buildPlan(arch, index, headers, options.plan));

    // ---- Space -----------------------------------------------------------
    const std::filesystem::path partialDir = outputDir.string() + ".partial";
    TF_TRY(const u64 available, io::availableSpace(partialDir));
    const u64 needed = plan.totalOutputBytes() + 64ull * 1024 * 1024;
    if (available < needed) {
        return makeError(ErrorCode::OutOfDiskSpace,
                         "{} needs {:.1f} GiB but only {:.1f} GiB is free. The source is never "
                         "written to disk, so this is the install size, not twice it",
                         partialDir.string(),
                         static_cast<double>(needed) / (1024.0 * 1024 * 1024),
                         static_cast<double>(available) / (1024.0 * 1024 * 1024));
    }

    // ---- Resume ----------------------------------------------------------
    //
    // The lock is taken before anything is read or cleared, so two installers
    // cannot both decide to start fresh and then interleave their writes. It
    // creates the directory as a side effect, which is what the first run
    // needs.
    TF_TRY(InstallLock lock, InstallLock::acquire(partialDir));

    const std::string fingerprint = planFingerprint(plan, options.source);
    u64 firstOp = 0;
    u64 resumedBytes = 0;

    if (options.resume && std::filesystem::exists(partialDir / kStateFile)) {
        const auto stateText = io::readTextFile(partialDir / kStateFile);
        if (stateText) {
            const auto state = InstallState::parse(*stateText);
            // A mismatched fingerprint means the earlier attempt was laying
            // bytes out differently. Resuming across that would interleave two
            // layouts in one file and produce an install that passes its own
            // hash check while being internally wrong.
            if (state && state->planFingerprint == fingerprint &&
                state->opsCompleted <= plan.ops.size()) {
                firstOp = state->opsCompleted;
                resumedBytes = state->bytesWritten;
            }
        }
    }

    // The state describes bytes sitting in specific files at specific sizes. If
    // one of them was deleted or truncated since - a half-finished cleanup, a
    // disk that filled - resuming would reopen it, zero-fill it, and skip the
    // operations that were meant to fill it. Starting over is the only safe
    // answer, and it is cheap next to a silently hollow install.
    if (firstOp > 0) {
        const auto intact = [&](const std::filesystem::path& path, u64 expected) {
            std::error_code sizeError;
            const auto actual = std::filesystem::file_size(path, sizeError);
            return !sizeError && actual == expected;
        };

        bool allPresent = intact(partialDir / gturbo::kResidentFile, plan.residentBytes());
        for (u64 layer = 0; allPresent && layer < plan.experts.numLayers; ++layer) {
            allPresent = intact(partialDir / gturbo::kExpertsDir / plan.experts.layerFiles[layer],
                                plan.experts.layerFileBytes());
        }
        if (!allPresent) {
            firstOp = 0;
            resumedBytes = 0;
        }
    }

    std::error_code ec;
    if (firstOp == 0) {
        // Starting over. The output files are reopened with overwrite, which
        // truncates and re-zeroes them, so only the stale state has to go -
        // and the lock file must not, since it is held open right here.
        std::filesystem::remove(partialDir / kStateFile, ec);
    }

    // ---- Stream the weights ---------------------------------------------
    u64 downloadedBytes = 0;
    u32 reconnects = 0;
    {
        OutputFiles outputs;
        TF_CHECK(outputs.open(partialDir, plan, /*fresh=*/firstOp == 0));

        std::vector<u8> scratch(static_cast<usize>(options.plan.maxOpBytes));

        const u64 totalBytes = plan.totalCopiedBytes();
        u64 doneBytes = resumedBytes;
        u32 currentShard = std::numeric_limits<u32>::max();
        RemoteFileStream stream;

        // Checkpointed often enough that a dropped connection costs seconds of
        // re-download, and rarely enough that the state file is not rewritten
        // thousands of times.
        constexpr u64 kCheckpointBytes = 256ull * 1024 * 1024;
        u64 nextCheckpoint = doneBytes + kCheckpointBytes;

        // Flushes before recording, so the state file never claims bytes are on
        // disk that are still in a buffer. A resume trusts this figure
        // completely: anything it over-reports is a hole in the install.
        const auto writeState = [&](u64 opsCompleted, u64 bytes) -> Status {
            const InstallState state{.repoId = options.source.repoId,
                                     .revision = options.source.revision,
                                     .planFingerprint = fingerprint,
                                     .opsCompleted = opsCompleted,
                                     .bytesWritten = bytes};
            const std::string text = state.toJson();
            TF_CHECK(outputs.flushAll());
            return io::writeFileAtomic(
                    partialDir / kStateFile,
                    ByteSpan{reinterpret_cast<const u8*>(text.data()), text.size()});
        };

        for (u64 opIndex = firstOp; opIndex < plan.ops.size(); ++opIndex) {
            const CopyOp& op = plan.ops[static_cast<usize>(opIndex)];

            if (op.shardIndex != currentShard) {
                currentShard = op.shardIndex;
                const std::string& shard = plan.shards[currentShard];
                const RemoteFile* remote = findRemote(shard);
                if (remote == nullptr) {
                    return makeError(ErrorCode::NotFound, "the repository does not list {}",
                                     shard);
                }
                // One connection at a time: closing the previous stream frees
                // its socket before the next is opened.
                reconnects += stream.reconnects();
                TF_TRY(stream, RemoteFileStream::open(client, options.source, shard,
                                                      remote->size, options.retry));

                if (!report(progress, Phase::CopyingWeights, doneBytes, totalBytes, shard)) {
                    TF_CHECK(writeState(opIndex, doneBytes));
                    return makeError(ErrorCode::Cancelled, "cancelled while downloading");
                }
            }

            const MutableByteSpan buffer{scratch.data(), static_cast<usize>(op.source.length)};
            TF_CHECK(stream.readExactAt(op.source.offset, buffer));

            TF_TRY(io::File* destination, outputs.forOp(op));
            TF_CHECK(destination->writeAt(op.destOffset, buffer));

            doneBytes += op.source.length;
            downloadedBytes += op.source.length;

            if (doneBytes >= nextCheckpoint) {
                TF_CHECK(writeState(opIndex + 1, doneBytes));
                nextCheckpoint = doneBytes + kCheckpointBytes;
                if (!report(progress, Phase::CopyingWeights, doneBytes, totalBytes,
                            plan.shards[op.shardIndex])) {
                    return makeError(ErrorCode::Cancelled, "cancelled while downloading");
                }
            } else if ((doneBytes & (64ull * 1024 * 1024 - 1)) < op.source.length) {
                if (!report(progress, Phase::CopyingWeights, doneBytes, totalBytes,
                            plan.shards[op.shardIndex])) {
                    TF_CHECK(writeState(opIndex + 1, doneBytes));
                    return makeError(ErrorCode::Cancelled, "cancelled while downloading");
                }
            }
        }

        reconnects += stream.reconnects();
        // The final checkpoint has to come before the files are closed, since
        // writing it flushes them.
        TF_CHECK(writeState(plan.ops.size(), doneBytes));
        outputs.closeAll();
        report(progress, Phase::CopyingWeights, totalBytes, totalBytes, "done");
    }

    // ---- Sidecars --------------------------------------------------------
    std::vector<std::string> tokenizerFiles;
    {
        const std::filesystem::path tokenizerDir = partialDir / gturbo::kTokenizerDir;
        std::filesystem::create_directories(tokenizerDir, ec);
        if (ec) {
            return makeError(ErrorCode::Io, "creating {}: {}", tokenizerDir.string(),
                             ec.message());
        }

        const auto fetchOne = [&](std::string_view name, bool required) -> Status {
            if (findRemote(name) == nullptr) {
                if (required) {
                    return makeError(ErrorCode::NotFound,
                                     "{} does not publish {}, which is required",
                                     options.source.repoId, name);
                }
                return {};
            }
            if (!report(progress, Phase::CopyingTokenizer, 0, 0, std::string{name})) {
                return makeError(ErrorCode::Cancelled, "cancelled while fetching {}", name);
            }
            TF_TRY(const std::vector<u8> contents,
                   fetchSmallFile(client, options.source, name, options.retry));
            TF_CHECK(io::writeFileAtomic(tokenizerDir / name, contents));
            tokenizerFiles.push_back(std::format("{}/{}", gturbo::kTokenizerDir, name));
            downloadedBytes += contents.size();
            return {};
        };

        for (const auto& name : kRequiredTokenizerFiles) {
            TF_CHECK(fetchOne(name, /*required=*/true));
        }
        for (const auto& name : kOptionalTokenizerFiles) {
            TF_CHECK(fetchOne(name, /*required=*/false));
        }
    }

    const std::string layoutJson = plan.experts.toJson();
    TF_CHECK(io::writeFileAtomic(
            partialDir / gturbo::kExpertsDir / gturbo::kExpertLayoutFile,
            ByteSpan{reinterpret_cast<const u8*>(layoutJson.data()), layoutJson.size()}));

    // ---- Hash and manifest ----------------------------------------------
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

    const u64 totalToHash = plan.totalOutputBytes();
    u64 hashed = 0;
    for (const auto& relative : hashTargets) {
        const std::filesystem::path path = partialDir / relative;
        const auto size = std::filesystem::file_size(path, ec);
        if (ec) {
            return makeError(ErrorCode::Io, "{}: {}", relative, ec.message());
        }

        std::string digest;
        if (options.verifyAfterWrite) {
            if (!report(progress, Phase::Hashing, hashed, totalToHash, relative)) {
                return makeError(ErrorCode::Cancelled, "cancelled while hashing");
            }
            TF_TRY(digest, io::Sha256::hashFile(path));
        }
        manifest.files.push_back(
                gturbo::FileDigest{.path = relative, .size = size, .sha256 = digest});
        hashed += size;
    }
    if (options.verifyAfterWrite) {
        report(progress, Phase::Hashing, totalToHash, totalToHash, "done");
    }

    TF_CHECK(manifest.validate());

    report(progress, Phase::WritingManifest, 0, 0, std::string{gturbo::kManifestFile});
    const std::string manifestJson = manifest.toJson();
    TF_CHECK(io::writeFileAtomic(
            partialDir / gturbo::kManifestFile,
            ByteSpan{reinterpret_cast<const u8*>(manifestJson.data()), manifestJson.size()}));

    // ---- Promote ---------------------------------------------------------
    report(progress, Phase::Promoting, 0, 0, outputDir.string());

    // The state file and the lock have no place in a finished install.
    std::filesystem::remove(partialDir / kStateFile, ec);
    lock.release();
    std::filesystem::remove(partialDir / kLockFile, ec);

    if (std::filesystem::exists(outputDir)) {
        TF_CHECK(removeAll(outputDir));
    }
    std::filesystem::rename(partialDir, outputDir, ec);
    if (ec) {
        return makeError(ErrorCode::Io, "renaming {} to {}: {}", partialDir.string(),
                         outputDir.string(), ec.message());
    }

    const auto elapsed = std::chrono::steady_clock::now() - started;
    return InstallResult{.installDir = outputDir,
                         .bytesWritten = plan.totalOutputBytes(),
                         .bytesDownloaded = downloadedBytes,
                         .bytesResumed = resumedBytes,
                         .reconnects = reconnects,
                         .elapsedSeconds = std::chrono::duration<double>(elapsed).count()};
}

}  // namespace tf::repack
