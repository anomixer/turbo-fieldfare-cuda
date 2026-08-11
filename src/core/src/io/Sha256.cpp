#include "tf/core/io/Sha256.h"

#include <windows.h>
// bcrypt.h must follow windows.h
#include <bcrypt.h>

#include <algorithm>
#include <utility>

#include "tf/core/io/File.h"

#pragma comment(lib, "bcrypt.lib")

namespace tf::io {
namespace {

/// CNG reports NTSTATUS, not GetLastError, so failures are formatted directly.
Error cngError(NTSTATUS status, std::string_view what) {
    return Error{ErrorCode::Unknown,
                 std::format("{}: BCrypt failed with status 0x{:08X}", what,
                             static_cast<unsigned long>(status))};
}

/// Chunk size for streaming a file through the hash. Large enough to keep the
/// NVMe busy, small enough that memory stays flat on a 5 GB shard.
constexpr u64 kHashChunkBytes = 4ull * 1024 * 1024;

}  // namespace

Sha256::~Sha256() { reset(); }

Sha256::Sha256(Sha256&& other) noexcept
    : algorithm_(std::exchange(other.algorithm_, nullptr)),
      hash_(std::exchange(other.hash_, nullptr)),
      hashObject_(std::move(other.hashObject_)) {}

Sha256& Sha256::operator=(Sha256&& other) noexcept {
    if (this != &other) {
        reset();
        algorithm_ = std::exchange(other.algorithm_, nullptr);
        hash_ = std::exchange(other.hash_, nullptr);
        hashObject_ = std::move(other.hashObject_);
    }
    return *this;
}

void Sha256::reset() {
    if (hash_ != nullptr) {
        ::BCryptDestroyHash(static_cast<BCRYPT_HASH_HANDLE>(hash_));
        hash_ = nullptr;
    }
    if (algorithm_ != nullptr) {
        ::BCryptCloseAlgorithmProvider(static_cast<BCRYPT_ALG_HANDLE>(algorithm_), 0);
        algorithm_ = nullptr;
    }
    hashObject_.clear();
}

Result<Sha256> Sha256::create() {
    Sha256 sha;

    BCRYPT_ALG_HANDLE algorithm = nullptr;
    NTSTATUS status = ::BCryptOpenAlgorithmProvider(&algorithm, BCRYPT_SHA256_ALGORITHM,
                                                    nullptr, 0);
    if (!BCRYPT_SUCCESS(status)) {
        return std::unexpected(cngError(status, "opening SHA-256 provider"));
    }
    sha.algorithm_ = algorithm;

    // CNG wants the caller to supply the hash object's backing storage, whose
    // size it reports rather than fixing.
    DWORD objectSize = 0;
    DWORD copied = 0;
    status = ::BCryptGetProperty(algorithm, BCRYPT_OBJECT_LENGTH,
                                 reinterpret_cast<PUCHAR>(&objectSize), sizeof(objectSize),
                                 &copied, 0);
    if (!BCRYPT_SUCCESS(status)) {
        return std::unexpected(cngError(status, "querying SHA-256 object length"));
    }
    sha.hashObject_.resize(objectSize);

    BCRYPT_HASH_HANDLE hash = nullptr;
    status = ::BCryptCreateHash(algorithm, &hash, sha.hashObject_.data(), objectSize, nullptr,
                                0, 0);
    if (!BCRYPT_SUCCESS(status)) {
        return std::unexpected(cngError(status, "creating SHA-256 hash"));
    }
    sha.hash_ = hash;

    return sha;
}

Status Sha256::update(ByteSpan data) {
    if (hash_ == nullptr) {
        return makeError(ErrorCode::InvalidArgument, "update() on a finished or empty hash");
    }
    if (data.empty()) {
        return {};
    }

    // BCryptHashData takes a ULONG length, so very large spans are chunked.
    usize consumed = 0;
    while (consumed < data.size()) {
        const ULONG chunk =
                static_cast<ULONG>(std::min<usize>(data.size() - consumed, 1u << 30));
        const NTSTATUS status = ::BCryptHashData(
                static_cast<BCRYPT_HASH_HANDLE>(hash_),
                const_cast<PUCHAR>(data.data() + consumed), chunk, 0);
        if (!BCRYPT_SUCCESS(status)) {
            return std::unexpected(cngError(status, "hashing data"));
        }
        consumed += chunk;
    }
    return {};
}

Result<Sha256::Digest> Sha256::finish() {
    if (hash_ == nullptr) {
        return makeError(ErrorCode::InvalidArgument, "finish() on a finished or empty hash");
    }

    Digest digest{};
    const NTSTATUS status = ::BCryptFinishHash(static_cast<BCRYPT_HASH_HANDLE>(hash_),
                                               digest.data(),
                                               static_cast<ULONG>(digest.size()), 0);
    if (!BCRYPT_SUCCESS(status)) {
        return std::unexpected(cngError(status, "finalizing hash"));
    }

    // The hash handle is single-use once finished.
    ::BCryptDestroyHash(static_cast<BCRYPT_HASH_HANDLE>(hash_));
    hash_ = nullptr;

    return digest;
}

std::string Sha256::toHex(const Digest& digest) {
    static constexpr char kHex[] = "0123456789abcdef";
    std::string out;
    out.reserve(digest.size() * 2);
    for (const u8 byte : digest) {
        out.push_back(kHex[byte >> 4]);
        out.push_back(kHex[byte & 0x0F]);
    }
    return out;
}

Result<std::string> Sha256::hashBytes(ByteSpan data) {
    TF_TRY(Sha256 sha, create());
    TF_CHECK(sha.update(data));
    TF_TRY(const Digest digest, sha.finish());
    return toHex(digest);
}

Result<std::string> Sha256::hashFile(
        const std::filesystem::path& path,
        const std::function<bool(u64, u64)>& progress) {
    TF_TRY(File file, File::openRead(path, ReadMode::Sequential));
    TF_TRY(const u64 totalBytes, file.size());
    TF_TRY(Sha256 sha, create());

    std::vector<u8> buffer(static_cast<usize>(std::min(kHashChunkBytes, std::max(totalBytes, u64{1}))));

    u64 offset = 0;
    while (offset < totalBytes) {
        const u64 want = std::min<u64>(buffer.size(), totalBytes - offset);
        TF_TRY(const u64 read, file.readAt(offset, MutableByteSpan{buffer.data(),
                                                                   static_cast<usize>(want)}));
        if (read == 0) {
            return makeError(ErrorCode::Io, "{}: read stalled at offset {} of {}",
                             path.string(), offset, totalBytes);
        }
        TF_CHECK(sha.update(ByteSpan{buffer.data(), static_cast<usize>(read)}));
        offset += read;

        if (progress && !progress(offset, totalBytes)) {
            return makeError(ErrorCode::Cancelled, "hashing {} was cancelled", path.string());
        }
    }

    TF_TRY(const Digest digest, sha.finish());
    return toHex(digest);
}

}  // namespace tf::io
