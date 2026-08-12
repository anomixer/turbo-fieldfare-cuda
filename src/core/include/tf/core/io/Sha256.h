#pragma once

#include <array>
#include <filesystem>
#include <functional>
#include <string>
#include <vector>

#include "tf/core/base/Error.h"
#include "tf/core/base/Types.h"

namespace tf::io {

/// Incremental SHA-256 over Windows' in-box CNG provider, so the project needs
/// no crypto dependency. Move-only; not safe to share across threads.
class Sha256 {
public:
    static constexpr usize kDigestBytes = 32;
    using Digest = std::array<u8, kDigestBytes>;

    Sha256() = default;
    ~Sha256();

    Sha256(const Sha256&) = delete;
    Sha256& operator=(const Sha256&) = delete;
    Sha256(Sha256&& other) noexcept;
    Sha256& operator=(Sha256&& other) noexcept;

    [[nodiscard]] static Result<Sha256> create();

    [[nodiscard]] Status update(ByteSpan data);

    /// Finalizes and returns the digest. The object cannot be reused after this.
    [[nodiscard]] Result<Digest> finish();

    [[nodiscard]] static std::string toHex(const Digest& digest);

    /// One-shot digest of a buffer, as lowercase hex.
    [[nodiscard]] static Result<std::string> hashBytes(ByteSpan data);

    /// Streams a whole file through the hash in bounded chunks, so a 5 GB shard
    /// costs a fixed amount of memory. `progress` may be null; returning false
    /// from it cancels the hash.
    [[nodiscard]] static Result<std::string> hashFile(
            const std::filesystem::path& path,
            const std::function<bool(u64 bytesHashed, u64 totalBytes)>& progress = {});

private:
    void reset();

    void* algorithm_ = nullptr;  // BCRYPT_ALG_HANDLE
    void* hash_ = nullptr;       // BCRYPT_HASH_HANDLE
    std::vector<u8> hashObject_;
};

}  // namespace tf::io
