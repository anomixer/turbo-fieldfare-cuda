#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "tf/core/base/Error.h"
#include "tf/core/base/Types.h"
#include "tf/core/net/Http.h"

namespace tf::repack {

/// Where a checkpoint is fetched from.
struct RemoteSource {
    std::string repoId = "mlx-community/gemma-4-26b-a4b-it-4bit";
    /// A commit hash rather than a branch: a repo can be updated underneath a
    /// resumed install, and a moving target would silently mix two revisions.
    std::string revision = "0d77464eeb233a2da68ebf9d7dc4edaac7db956d";
    /// Sent as `Authorization: Bearer`, for gated repositories. Empty for the
    /// public case, which this checkpoint is.
    std::string token;
    /// Overridable for a mirror or a local proxy.
    std::string endpoint = "https://huggingface.co";

    /// Resolve-style URL for one file in the repo.
    [[nodiscard]] std::string fileUrl(std::string_view file) const;

    [[nodiscard]] net::HeaderList authHeaders() const;

    [[nodiscard]] Status validate() const;
};

/// One file in the repository, as the server describes it.
struct RemoteFile {
    std::string name;
    u64 size = 0;
    /// Whatever the server offered to identify the content, for detecting that
    /// a resumed download is looking at different bytes than it started with.
    std::string etag;
};

/// Retry policy for a transfer that can run for tens of minutes.
///
/// A 14 GB download over a home connection will meet a dropped connection or a
/// rate limit at some point; treating the first one as fatal would make the
/// installer unusable in practice.
struct RetryPolicy {
    u32 maxAttempts = 6;
    u32 initialDelayMillis = 500;
    /// Each attempt waits twice as long, capped here.
    u32 maxDelayMillis = 30000;

    /// Delay before attempt `attempt`, counted from 1.
    [[nodiscard]] u32 delayForAttempt(u32 attempt) const;
};

/// Fetches the repository's file list and sizes.
[[nodiscard]] Result<std::vector<RemoteFile>> listRemoteFiles(net::HttpClient& client,
                                                              const RemoteSource& source);

/// Largest sidecar fetched whole into memory.
///
/// Sized for tokenizer.json, which is 32 MB in this checkpoint - far larger than
/// "metadata" suggests, and the reason this is a named constant rather than an
/// assumption. Still small enough that a weight shard requested by mistake is
/// refused rather than buffered.
inline constexpr u64 kMaxSidecarBytes = 128ull * 1024 * 1024;

/// Downloads one whole file into memory. For config.json, the safetensors index
/// and the tokenizer - not for weights.
[[nodiscard]] Result<std::vector<u8>> fetchSmallFile(net::HttpClient& client,
                                                     const RemoteSource& source,
                                                     std::string_view name,
                                                     const RetryPolicy& retry = {},
                                                     u64 limitBytes = kMaxSidecarBytes);

/// Reads one remote file as a forward-only stream, reopening on failure.
///
/// The repacker's operations are sorted by (shard, source offset), so the
/// source is consumed strictly forward and a single range request covers a
/// whole shard. That is what makes a streaming install possible at all: the
/// alternative, a request per operation, would be 35000 round trips.
///
/// Small forward jumps are absorbed by reading and discarding, which is
/// cheaper than a new request once the gap is under a few hundred kilobytes.
/// A backward seek reopens, and should not happen with a sorted plan.
class RemoteFileStream {
public:
    RemoteFileStream() = default;

    [[nodiscard]] static Result<RemoteFileStream> open(net::HttpClient& client,
                                                       const RemoteSource& source,
                                                       std::string_view name, u64 fileSize,
                                                       const RetryPolicy& retry = {});

    /// Reads `destination.size()` bytes starting at `offset`, which must not be
    /// before the current position unless a reopen is acceptable.
    [[nodiscard]] Status readExactAt(u64 offset, MutableByteSpan destination);

    [[nodiscard]] u64 position() const noexcept { return position_; }
    [[nodiscard]] u64 size() const noexcept { return fileSize_; }

    /// How many times the connection had to be re-established. Reported at the
    /// end of an install, because a large number means something is wrong with
    /// the link even though the install succeeded.
    [[nodiscard]] u32 reconnects() const noexcept { return reconnects_; }

private:
    [[nodiscard]] Status reopenAt(u64 offset);

    net::HttpClient* client_ = nullptr;
    RemoteSource source_;
    std::string name_;
    RetryPolicy retry_;
    net::HttpStream stream_;
    u64 fileSize_ = 0;
    u64 position_ = 0;
    u32 reconnects_ = 0;

    /// Discard buffer for skipping a short gap.
    std::vector<u8> skipBuffer_;
};

/// Largest forward gap absorbed by reading and discarding rather than by
/// reopening the connection. A new HTTPS request to a CDN costs a round trip
/// plus a TLS handshake, which is worth more than 256 KiB of transfer.
inline constexpr u64 kMaxSkipBytes = 256ull * 1024;

}  // namespace tf::repack
