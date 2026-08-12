#include "tf/core/io/File.h"

#include <windows.h>

#include <algorithm>
#include <utility>

namespace tf::io {
namespace {

constexpr u64 kMaxSingleIo = 32ull * 1024 * 1024;

[[nodiscard]] HANDLE toHandle(void* raw) {
    return raw == nullptr ? INVALID_HANDLE_VALUE : static_cast<HANDLE>(raw);
}

[[nodiscard]] void* fromHandle(HANDLE handle) {
    return handle == INVALID_HANDLE_VALUE ? nullptr : handle;
}

/// Splits a u64 offset into the OVERLAPPED Offset/OffsetHigh pair, which is how
/// Win32 expresses a positional read.
void setOverlappedOffset(OVERLAPPED& overlapped, u64 offset) {
    overlapped = OVERLAPPED{};
    overlapped.Offset = static_cast<DWORD>(offset & 0xFFFFFFFFull);
    overlapped.OffsetHigh = static_cast<DWORD>(offset >> 32);
}

}  // namespace

// ---------------------------------------------------------------------------
// File
// ---------------------------------------------------------------------------

File::~File() { close(); }

File::File(File&& other) noexcept : handle_(std::exchange(other.handle_, nullptr)) {}

File& File::operator=(File&& other) noexcept {
    if (this != &other) {
        close();
        handle_ = std::exchange(other.handle_, nullptr);
    }
    return *this;
}

bool File::isOpen() const noexcept { return handle_ != nullptr; }

void File::close() {
    if (handle_ != nullptr) {
        ::CloseHandle(toHandle(handle_));
        handle_ = nullptr;
    }
}

Result<File> File::openRead(const std::filesystem::path& path, ReadMode mode) {
    DWORD flags = FILE_ATTRIBUTE_NORMAL;
    switch (mode) {
        case ReadMode::Sequential:
            flags |= FILE_FLAG_SEQUENTIAL_SCAN;
            break;
        case ReadMode::RandomBuffered:
            flags |= FILE_FLAG_RANDOM_ACCESS;
            break;
        case ReadMode::Unbuffered:
            flags |= FILE_FLAG_NO_BUFFERING;
            break;
    }

    // FILE_SHARE_READ lets several processes stream experts from one install,
    // which the decode service and CLI may do concurrently.
    const HANDLE handle = ::CreateFileW(path.c_str(), GENERIC_READ,
                                        FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr,
                                        OPEN_EXISTING, flags, nullptr);
    if (handle == INVALID_HANDLE_VALUE) {
        return std::unexpected(lastWin32Error(std::format("opening {}", path.string())));
    }

    File file;
    file.handle_ = fromHandle(handle);
    return file;
}

Result<File> File::create(const std::filesystem::path& path, bool overwrite) {
    const HANDLE handle =
            ::CreateFileW(path.c_str(), GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ, nullptr,
                          overwrite ? CREATE_ALWAYS : CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (handle == INVALID_HANDLE_VALUE) {
        return std::unexpected(lastWin32Error(std::format("creating {}", path.string())));
    }

    File file;
    file.handle_ = fromHandle(handle);
    return file;
}

Result<File> File::openWrite(const std::filesystem::path& path) {
    const HANDLE handle =
            ::CreateFileW(path.c_str(), GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ, nullptr,
                          OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (handle == INVALID_HANDLE_VALUE) {
        return std::unexpected(lastWin32Error(std::format("opening {} for writing",
                                                          path.string())));
    }

    File file;
    file.handle_ = fromHandle(handle);
    return file;
}

Result<u64> File::size() const {
    if (!isOpen()) {
        return makeError(ErrorCode::InvalidArgument, "size() on a closed file");
    }
    LARGE_INTEGER value{};
    if (::GetFileSizeEx(toHandle(handle_), &value) == 0) {
        return std::unexpected(lastWin32Error("querying file size"));
    }
    return static_cast<u64>(value.QuadPart);
}

Result<u64> File::readAt(u64 offset, MutableByteSpan destination) const {
    if (!isOpen()) {
        return makeError(ErrorCode::InvalidArgument, "readAt() on a closed file");
    }

    u64 totalRead = 0;
    while (totalRead < destination.size()) {
        // ReadFile takes a DWORD count, and very large single reads are also
        // worse for latency, so chunk them.
        const u64 remaining = destination.size() - totalRead;
        const DWORD toRead = static_cast<DWORD>(std::min(remaining, kMaxSingleIo));

        OVERLAPPED overlapped;
        setOverlappedOffset(overlapped, offset + totalRead);

        DWORD bytesRead = 0;
        if (::ReadFile(toHandle(handle_), destination.data() + totalRead, toRead, &bytesRead,
                       &overlapped) == 0) {
            const DWORD error = ::GetLastError();
            // A synchronous handle still reports EOF this way for a read that
            // starts at or past the end of the file.
            if (error == ERROR_HANDLE_EOF) {
                break;
            }
            return std::unexpected(win32Error(error, "reading file"));
        }
        if (bytesRead == 0) {
            break;  // end of file
        }
        totalRead += bytesRead;
    }
    return totalRead;
}

Status File::readExactAt(u64 offset, MutableByteSpan destination) const {
    TF_TRY(const u64 read, readAt(offset, destination));
    if (read != destination.size()) {
        return makeError(ErrorCode::MalformedData,
                         "short read at offset {}: wanted {} bytes, got {}", offset,
                         destination.size(), read);
    }
    return {};
}

Status File::writeAt(u64 offset, ByteSpan source) {
    if (!isOpen()) {
        return makeError(ErrorCode::InvalidArgument, "writeAt() on a closed file");
    }

    u64 totalWritten = 0;
    while (totalWritten < source.size()) {
        const u64 remaining = source.size() - totalWritten;
        const DWORD toWrite = static_cast<DWORD>(std::min(remaining, kMaxSingleIo));

        OVERLAPPED overlapped;
        setOverlappedOffset(overlapped, offset + totalWritten);

        DWORD written = 0;
        if (::WriteFile(toHandle(handle_), source.data() + totalWritten, toWrite, &written,
                        &overlapped) == 0) {
            return std::unexpected(lastWin32Error("writing file"));
        }
        if (written == 0) {
            return makeError(ErrorCode::Io, "write made no progress at offset {}",
                             offset + totalWritten);
        }
        totalWritten += written;
    }
    return {};
}

Status File::write(ByteSpan source) {
    if (!isOpen()) {
        return makeError(ErrorCode::InvalidArgument, "write() on a closed file");
    }

    u64 totalWritten = 0;
    while (totalWritten < source.size()) {
        const u64 remaining = source.size() - totalWritten;
        const DWORD toWrite = static_cast<DWORD>(std::min(remaining, kMaxSingleIo));

        DWORD written = 0;
        if (::WriteFile(toHandle(handle_), source.data() + totalWritten, toWrite, &written,
                        nullptr) == 0) {
            return std::unexpected(lastWin32Error("writing file"));
        }
        if (written == 0) {
            return makeError(ErrorCode::Io, "write made no progress");
        }
        totalWritten += written;
    }
    return {};
}

Status File::preallocate(u64 bytes) {
    if (!isOpen()) {
        return makeError(ErrorCode::InvalidArgument, "preallocate() on a closed file");
    }

    LARGE_INTEGER position{};
    position.QuadPart = static_cast<LONGLONG>(bytes);
    if (::SetFilePointerEx(toHandle(handle_), position, nullptr, FILE_BEGIN) == 0) {
        return std::unexpected(lastWin32Error("seeking to preallocate"));
    }
    if (::SetEndOfFile(toHandle(handle_)) == 0) {
        return std::unexpected(lastWin32Error("setting end of file"));
    }

    // Deliberately not calling SetFileValidData: it would skip zero-filling and
    // so expose whatever the disk previously held, and it needs
    // SE_MANAGE_VOLUME_NAME. Reserving the extent is enough.
    LARGE_INTEGER rewind{};
    if (::SetFilePointerEx(toHandle(handle_), rewind, nullptr, FILE_BEGIN) == 0) {
        return std::unexpected(lastWin32Error("rewinding after preallocate"));
    }
    return {};
}

Status File::flush() {
    if (!isOpen()) {
        return makeError(ErrorCode::InvalidArgument, "flush() on a closed file");
    }
    if (::FlushFileBuffers(toHandle(handle_)) == 0) {
        return std::unexpected(lastWin32Error("flushing file"));
    }
    return {};
}

Result<u64> File::sectorSize(const std::filesystem::path& path) {
    const std::filesystem::path absolute = std::filesystem::absolute(path);
    const std::wstring root = absolute.root_path().wstring();

    DWORD sectorsPerCluster = 0;
    DWORD bytesPerSector = 0;
    DWORD freeClusters = 0;
    DWORD totalClusters = 0;
    if (::GetDiskFreeSpaceW(root.c_str(), &sectorsPerCluster, &bytesPerSector, &freeClusters,
                            &totalClusters) == 0) {
        return std::unexpected(lastWin32Error(std::format("querying sector size of {}",
                                                          absolute.root_path().string())));
    }
    return static_cast<u64>(bytesPerSector);
}

// ---------------------------------------------------------------------------
// MappedFile
// ---------------------------------------------------------------------------

MappedFile::~MappedFile() { close(); }

MappedFile::MappedFile(MappedFile&& other) noexcept
    : fileHandle_(std::exchange(other.fileHandle_, nullptr)),
      mappingHandle_(std::exchange(other.mappingHandle_, nullptr)),
      data_(std::exchange(other.data_, nullptr)),
      size_(std::exchange(other.size_, 0)) {}

MappedFile& MappedFile::operator=(MappedFile&& other) noexcept {
    if (this != &other) {
        close();
        fileHandle_ = std::exchange(other.fileHandle_, nullptr);
        mappingHandle_ = std::exchange(other.mappingHandle_, nullptr);
        data_ = std::exchange(other.data_, nullptr);
        size_ = std::exchange(other.size_, 0);
    }
    return *this;
}

void MappedFile::close() {
    if (data_ != nullptr) {
        ::UnmapViewOfFile(data_);
        data_ = nullptr;
    }
    if (mappingHandle_ != nullptr) {
        ::CloseHandle(static_cast<HANDLE>(mappingHandle_));
        mappingHandle_ = nullptr;
    }
    if (fileHandle_ != nullptr) {
        ::CloseHandle(static_cast<HANDLE>(fileHandle_));
        fileHandle_ = nullptr;
    }
    size_ = 0;
}

Result<MappedFile> MappedFile::open(const std::filesystem::path& path) {
    const HANDLE file =
            ::CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr,
                          OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE) {
        return std::unexpected(lastWin32Error(std::format("opening {} for mapping",
                                                          path.string())));
    }

    LARGE_INTEGER fileSize{};
    if (::GetFileSizeEx(file, &fileSize) == 0) {
        const Error error = lastWin32Error("sizing mapped file");
        ::CloseHandle(file);
        return std::unexpected(error);
    }

    // CreateFileMapping rejects a zero-length file, so report it as data rather
    // than surfacing an opaque Win32 error.
    if (fileSize.QuadPart == 0) {
        ::CloseHandle(file);
        return makeError(ErrorCode::MalformedData, "{} is empty", path.string());
    }

    const HANDLE mapping = ::CreateFileMappingW(file, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (mapping == nullptr) {
        const Error error = lastWin32Error("creating file mapping");
        ::CloseHandle(file);
        return std::unexpected(error);
    }

    const void* view = ::MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0);
    if (view == nullptr) {
        const Error error = lastWin32Error("mapping view of file");
        ::CloseHandle(mapping);
        ::CloseHandle(file);
        return std::unexpected(error);
    }

    MappedFile mapped;
    mapped.fileHandle_ = file;
    mapped.mappingHandle_ = mapping;
    mapped.data_ = static_cast<const u8*>(view);
    mapped.size_ = static_cast<u64>(fileSize.QuadPart);
    return mapped;
}

Result<ByteSpan> MappedFile::range(ByteRange requested) const {
    if (data_ == nullptr) {
        return makeError(ErrorCode::InvalidArgument, "range() on a closed mapping");
    }
    if (requested.offset > size_ || requested.length > size_ - requested.offset) {
        return makeError(ErrorCode::MalformedData,
                         "range [{}, {}) is outside the {} byte mapping", requested.offset,
                         requested.end(), size_);
    }
    return ByteSpan{data_ + requested.offset, requested.length};
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

namespace {

Result<std::vector<u8>> readWhole(const std::filesystem::path& path, u64 maxBytes) {
    TF_TRY(File file, File::openRead(path, ReadMode::Sequential));
    TF_TRY(const u64 fileSize, file.size());

    if (fileSize > maxBytes) {
        return makeError(ErrorCode::InvalidArgument,
                         "{} is {} bytes, above the {} byte limit for a whole-file read",
                         path.string(), fileSize, maxBytes);
    }

    std::vector<u8> contents(static_cast<usize>(fileSize));
    if (fileSize > 0) {
        TF_CHECK(file.readExactAt(0, contents));
    }
    return contents;
}

}  // namespace

Result<std::string> readTextFile(const std::filesystem::path& path, u64 maxBytes) {
    TF_TRY(const std::vector<u8> bytes, readWhole(path, maxBytes));
    return std::string{reinterpret_cast<const char*>(bytes.data()), bytes.size()};
}

Result<std::vector<u8>> readBinaryFile(const std::filesystem::path& path, u64 maxBytes) {
    return readWhole(path, maxBytes);
}

Status writeFileAtomic(const std::filesystem::path& path, ByteSpan contents) {
    std::filesystem::path temporary = path;
    temporary += L".tmp";

    {
        TF_TRY(File file, File::create(temporary, /*overwrite=*/true));
        TF_CHECK(file.write(contents));
        TF_CHECK(file.flush());
    }

    // MOVEFILE_REPLACE_EXISTING makes this a same-volume rename over any prior
    // file; MOVEFILE_WRITE_THROUGH waits for the change to reach the disk.
    if (::MoveFileExW(temporary.c_str(), path.c_str(),
                      MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) == 0) {
        const Error error = lastWin32Error(std::format("renaming {} over {}",
                                                       temporary.string(), path.string()));
        std::error_code ignored;
        std::filesystem::remove(temporary, ignored);
        return std::unexpected(error);
    }
    return {};
}

Result<u64> availableSpace(const std::filesystem::path& path) {
    // Walk up to the nearest existing ancestor: the install directory usually
    // does not exist yet when the space check runs.
    std::filesystem::path probe = std::filesystem::absolute(path);
    std::error_code ec;
    while (!probe.empty() && !std::filesystem::exists(probe, ec)) {
        const std::filesystem::path parent = probe.parent_path();
        if (parent == probe) {
            break;
        }
        probe = parent;
    }

    ULARGE_INTEGER available{};
    ULARGE_INTEGER total{};
    ULARGE_INTEGER free{};
    if (::GetDiskFreeSpaceExW(probe.c_str(), &available, &total, &free) == 0) {
        return std::unexpected(
                lastWin32Error(std::format("querying free space on {}", probe.string())));
    }
    return static_cast<u64>(available.QuadPart);
}

}  // namespace tf::io
