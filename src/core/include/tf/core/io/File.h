#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "tf/core/base/Error.h"
#include "tf/core/base/Types.h"

namespace tf::io {

/// Hints passed to CreateFile. The read strategy is a first-class choice here
/// because it is the central performance decision in the expert streamer.
enum class ReadMode {
    /// Buffered with FILE_FLAG_SEQUENTIAL_SCAN. For the repacker, which walks
    /// each shard front to back exactly once.
    Sequential,
    /// Buffered with FILE_FLAG_RANDOM_ACCESS. Repeat reads of the same expert
    /// are served from the Windows page cache at RAM speed, at the cost of one
    /// memcpy into the destination. The default for expert streaming on a
    /// machine with RAM to spare.
    RandomBuffered,
    /// FILE_FLAG_NO_BUFFERING: DMA straight into sector-aligned memory with no
    /// intermediate copy, bypassing the page cache. Offsets, byte counts and
    /// destination addresses must all be sector-aligned.
    Unbuffered,
};

/// RAII wrapper over a Win32 file HANDLE. Move-only.
class File {
public:
    File() = default;
    ~File();

    File(const File&) = delete;
    File& operator=(const File&) = delete;
    File(File&& other) noexcept;
    File& operator=(File&& other) noexcept;

    [[nodiscard]] static Result<File> openRead(const std::filesystem::path& path,
                                               ReadMode mode = ReadMode::Sequential);

    /// Creates or truncates a file for writing.
    [[nodiscard]] static Result<File> create(const std::filesystem::path& path,
                                             bool overwrite = true);

    /// Opens an existing file for writing, keeping its contents and its size.
    ///
    /// Distinct from create(): a resumed download reopens files that already
    /// hold most of the install, so truncating them would throw away exactly
    /// the work the resume exists to preserve.
    [[nodiscard]] static Result<File> openWrite(const std::filesystem::path& path);

    [[nodiscard]] bool isOpen() const noexcept;
    [[nodiscard]] Result<u64> size() const;

    /// Positional read, the equivalent of POSIX pread: the file pointer is not
    /// used, so concurrent reads on one handle from several threads are safe.
    /// Returns the number of bytes read, which is short only at end of file.
    [[nodiscard]] Result<u64> readAt(u64 offset, MutableByteSpan destination) const;

    /// Positional read that treats a short read as an error. This is what the
    /// repacker and expert streamer want: a truncated tensor is corruption, not
    /// a condition to handle.
    [[nodiscard]] Status readExactAt(u64 offset, MutableByteSpan destination) const;

    [[nodiscard]] Status writeAt(u64 offset, ByteSpan source);

    /// Appends at the current write position.
    [[nodiscard]] Status write(ByteSpan source);

    /// Reserves `bytes` on disk up front so a large write cannot fail partway
    /// through for lack of space, and so the file is less fragmented.
    [[nodiscard]] Status preallocate(u64 bytes);

    [[nodiscard]] Status flush();

    void close();

    /// Physical sector size of the volume holding `path`, needed to satisfy the
    /// alignment rules of ReadMode::Unbuffered.
    [[nodiscard]] static Result<u64> sectorSize(const std::filesystem::path& path);

private:
    void* handle_ = nullptr;  // HANDLE, INVALID_HANDLE_VALUE when closed
};

/// Read-only memory mapping. Used for the resident weight file, which the GPU
/// backend uploads from without the runtime ever copying it into its own heap.
class MappedFile {
public:
    MappedFile() = default;
    ~MappedFile();

    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;
    MappedFile(MappedFile&& other) noexcept;
    MappedFile& operator=(MappedFile&& other) noexcept;

    [[nodiscard]] static Result<MappedFile> open(const std::filesystem::path& path);

    [[nodiscard]] ByteSpan bytes() const noexcept { return {data_, size_}; }
    [[nodiscard]] const u8* data() const noexcept { return data_; }
    [[nodiscard]] u64 size() const noexcept { return size_; }
    [[nodiscard]] bool isOpen() const noexcept { return data_ != nullptr; }

    /// A sub-range of the mapping, bounds-checked.
    [[nodiscard]] Result<ByteSpan> range(ByteRange range) const;

    void close();

private:
    void* fileHandle_ = nullptr;
    void* mappingHandle_ = nullptr;
    const u8* data_ = nullptr;
    u64 size_ = 0;
};

// ---- Convenience helpers --------------------------------------------------

/// Reads an entire file. Intended for small sidecars (config.json, manifests);
/// refuses anything above `maxBytes` so a mistaken path cannot exhaust memory.
[[nodiscard]] Result<std::string> readTextFile(const std::filesystem::path& path,
                                               u64 maxBytes = 512ull * 1024 * 1024);

[[nodiscard]] Result<std::vector<u8>> readBinaryFile(const std::filesystem::path& path,
                                                     u64 maxBytes = 512ull * 1024 * 1024);

/// Writes via a sibling temporary then renames over the destination, so a
/// crash mid-write cannot leave a half-written manifest that later reads as
/// valid. Both paths must be on the same volume for the rename to be atomic.
[[nodiscard]] Status writeFileAtomic(const std::filesystem::path& path, ByteSpan contents);

/// Free bytes available on the volume containing `path`, for the installer's
/// pre-flight space check.
[[nodiscard]] Result<u64> availableSpace(const std::filesystem::path& path);

}  // namespace tf::io
