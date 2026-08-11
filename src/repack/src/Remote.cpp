#include "tf/repack/Remote.h"

#include <algorithm>
#include <chrono>
#include <format>
#include <thread>

#include "tf/core/json/Json.h"

namespace tf::repack {
namespace {

/// Percent-encodes the characters that appear in a repo id or a file name and
/// would otherwise change the meaning of the path. Slashes are left alone: a
/// repo id is `owner/name` and the slash is structural.
[[nodiscard]] std::string encodePath(std::string_view text) {
    std::string out;
    out.reserve(text.size());
    for (const char c : text) {
        const bool safe = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                          (c >= '0' && c <= '9') || c == '-' || c == '_' || c == '.' ||
                          c == '~' || c == '/';
        if (safe) {
            out += c;
        } else {
            out += std::format("%{:02X}", static_cast<unsigned char>(c));
        }
    }
    return out;
}

/// Sleeps, then reports whether another attempt is worth making.
[[nodiscard]] bool waitBeforeRetry(const RetryPolicy& retry, u32 attempt) {
    if (attempt >= retry.maxAttempts) {
        return false;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds{retry.delayForAttempt(attempt)});
    return true;
}

}  // namespace

std::string RemoteSource::fileUrl(std::string_view file) const {
    // The resolve endpoint serves the file itself and 302s to the CDN, which is
    // the redirect WinHTTP follows for us.
    return std::format("{}/{}/resolve/{}/{}", endpoint, encodePath(repoId),
                       encodePath(revision), encodePath(file));
}

net::HeaderList RemoteSource::authHeaders() const {
    if (token.empty()) {
        return {};
    }
    return {{"Authorization", std::format("Bearer {}", token)}};
}

Status RemoteSource::validate() const {
    if (repoId.empty()) {
        return makeError(ErrorCode::InvalidArgument, "no repository was given");
    }
    if (repoId.find('/') == std::string::npos) {
        return makeError(ErrorCode::InvalidArgument,
                         "'{}' is not an owner/name repository id", repoId);
    }
    if (revision.empty()) {
        return makeError(ErrorCode::InvalidArgument, "no revision was given");
    }
    if (endpoint.empty()) {
        return makeError(ErrorCode::InvalidArgument, "no endpoint was given");
    }
    return {};
}

u32 RetryPolicy::delayForAttempt(u32 attempt) const {
    if (attempt == 0) {
        return 0;
    }
    u64 delay = initialDelayMillis;
    for (u32 i = 1; i < attempt; ++i) {
        delay *= 2;
        if (delay >= maxDelayMillis) {
            return maxDelayMillis;
        }
    }
    return static_cast<u32>(std::min<u64>(delay, maxDelayMillis));
}

Result<std::vector<RemoteFile>> listRemoteFiles(net::HttpClient& client,
                                                const RemoteSource& source) {
    TF_CHECK(source.validate());

    // The tree endpoint lists names and sizes in one request. `recursive=1`
    // matters: the tokenizer lives in a subdirectory in some repos.
    const std::string url = std::format("{}/api/models/{}/tree/{}?recursive=1", source.endpoint,
                                        encodePath(source.repoId), encodePath(source.revision));

    TF_TRY(const net::HttpResponse response, client.get(url, source.authHeaders()));
    if (response.statusCode == 401 || response.statusCode == 403) {
        return makeError(ErrorCode::VerificationFailed,
                         "{} is gated or private; set a token with --hf-token or HF_TOKEN",
                         source.repoId);
    }
    if (response.statusCode == 404) {
        return makeError(ErrorCode::NotFound, "no repository {} at revision {}", source.repoId,
                         source.revision);
    }
    if (!response.ok()) {
        return makeError(ErrorCode::Network, "listing {} returned HTTP {}", source.repoId,
                         response.statusCode);
    }

    TF_TRY(const json::Value tree, json::parse(response.bodyText()));
    TF_TRY(const json::Array* entries, tree.asArray());

    std::vector<RemoteFile> files;
    for (const json::Value& entry : *entries) {
        // Directories are listed alongside files; only files have bytes.
        if (const json::Value* type = entry.find("type"); type != nullptr) {
            const auto kind = type->asString();
            if (kind && *kind != "file") {
                continue;
            }
        }
        const json::Value* path = entry.find("path");
        if (path == nullptr) {
            continue;
        }
        const auto name = path->asString();
        if (!name) {
            continue;
        }

        RemoteFile file;
        file.name = std::string{*name};
        if (const json::Value* size = entry.find("size"); size != nullptr) {
            if (const auto bytes = size->asUInt()) {
                file.size = *bytes;
            }
        }
        if (const json::Value* oid = entry.find("oid"); oid != nullptr) {
            if (const auto value = oid->asString()) {
                file.etag = std::string{*value};
            }
        }
        files.push_back(std::move(file));
    }

    if (files.empty()) {
        return makeError(ErrorCode::MalformedData, "{} listed no files", source.repoId);
    }
    return files;
}

Result<std::vector<u8>> fetchSmallFile(net::HttpClient& client, const RemoteSource& source,
                                       std::string_view name, const RetryPolicy& retry,
                                       u64 limitBytes) {
    TF_CHECK(source.validate());
    const std::string url = source.fileUrl(name);

    Error lastFailure{ErrorCode::Network, "no attempt was made"};
    for (u32 attempt = 1; attempt <= retry.maxAttempts; ++attempt) {
        auto response = client.get(url, source.authHeaders(), limitBytes);
        if (!response) {
            lastFailure = response.error();
            if (waitBeforeRetry(retry, attempt)) {
                continue;
            }
            break;
        }
        if (response->ok()) {
            return std::move(response->body);
        }
        if (response->statusCode == 404) {
            return makeError(ErrorCode::NotFound, "{} has no file {}", source.repoId, name);
        }
        if (response->statusCode == 401 || response->statusCode == 403) {
            return makeError(ErrorCode::VerificationFailed,
                             "access to {} in {} was refused; a token may be needed", name,
                             source.repoId);
        }

        lastFailure = Error{ErrorCode::Network,
                            std::format("fetching {} returned HTTP {}", name,
                                        response->statusCode)};
        if (!net::isRetryable(response->statusCode) || !waitBeforeRetry(retry, attempt)) {
            break;
        }
    }
    return std::unexpected(lastFailure);
}

Result<RemoteFileStream> RemoteFileStream::open(net::HttpClient& client,
                                                const RemoteSource& source,
                                                std::string_view name, u64 fileSize,
                                                const RetryPolicy& retry) {
    TF_CHECK(source.validate());

    RemoteFileStream file;
    file.client_ = &client;
    file.source_ = source;
    file.name_ = name;
    file.retry_ = retry;
    file.fileSize_ = fileSize;
    TF_CHECK(file.reopenAt(0));
    return file;
}

Status RemoteFileStream::reopenAt(u64 offset) {
    const std::string url = source_.fileUrl(name_);

    Error lastFailure{ErrorCode::Network, "no attempt was made"};
    for (u32 attempt = 1; attempt <= retry_.maxAttempts; ++attempt) {
        auto stream = client_->openRange(url, offset, 0, source_.authHeaders());
        if (!stream) {
            lastFailure = stream.error();
            if (waitBeforeRetry(retry_, attempt)) {
                continue;
            }
            break;
        }

        if (stream->statusCode() == 206) {
            stream_ = std::move(*stream);
            position_ = offset;
            return {};
        }
        if (stream->statusCode() == 200) {
            // The server ignored Range and is sending the whole file. Correct
            // only from the very beginning; anywhere else the bytes would be
            // silently misaligned, which is far worse than failing.
            if (offset == 0) {
                stream_ = std::move(*stream);
                position_ = 0;
                return {};
            }
            return makeError(ErrorCode::Network,
                             "the server ignored a range request for {} at offset {}; a "
                             "resumed install needs range support",
                             name_, offset);
        }
        if (stream->statusCode() == 404) {
            return makeError(ErrorCode::NotFound, "{} has no file {}", source_.repoId, name_);
        }
        if (stream->statusCode() == 401 || stream->statusCode() == 403) {
            return makeError(ErrorCode::VerificationFailed,
                             "access to {} was refused; a token may be needed", name_);
        }

        lastFailure = Error{ErrorCode::Network,
                            std::format("opening {} at offset {} returned HTTP {}", name_,
                                        offset, stream->statusCode())};
        if (!net::isRetryable(stream->statusCode()) || !waitBeforeRetry(retry_, attempt)) {
            break;
        }
    }
    return std::unexpected(lastFailure);
}

Status RemoteFileStream::readExactAt(u64 offset, MutableByteSpan destination) {
    if (destination.empty()) {
        return {};
    }
    if (offset + destination.size() > fileSize_) {
        return makeError(ErrorCode::InvalidArgument,
                         "reading {} bytes at offset {} runs past the {} byte file {}",
                         destination.size(), offset, fileSize_, name_);
    }

    // Absorb a short forward gap by discarding; reopen for anything else. A
    // backward seek means the plan was not sorted, which would make a streaming
    // install pathologically slow, so it is worth reopening loudly rather than
    // quietly.
    if (offset < position_ || offset - position_ > kMaxSkipBytes) {
        if (offset != position_) {
            ++reconnects_;
            TF_CHECK(reopenAt(offset));
        }
    } else if (offset > position_) {
        u64 remaining = offset - position_;
        if (skipBuffer_.size() < kMaxSkipBytes) {
            skipBuffer_.resize(static_cast<usize>(kMaxSkipBytes));
        }
        while (remaining > 0) {
            const usize chunk = static_cast<usize>(std::min<u64>(remaining, kMaxSkipBytes));
            const auto status = stream_.readExact(MutableByteSpan{skipBuffer_.data(), chunk});
            if (!status) {
                // The connection dropped mid-skip; a reopen puts us exactly
                // where we wanted to be anyway.
                ++reconnects_;
                TF_CHECK(reopenAt(offset));
                remaining = 0;
                break;
            }
            position_ += chunk;
            remaining -= chunk;
        }
    }

    // One retry around the read itself: a connection dropped after hours of
    // transfer should resume, not restart.
    const auto status = stream_.readExact(destination);
    if (!status) {
        ++reconnects_;
        TF_CHECK(reopenAt(offset));
        TF_CHECK(stream_.readExact(destination));
    }
    position_ = offset + destination.size();
    return {};
}

}  // namespace tf::repack
