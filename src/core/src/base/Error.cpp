#include "tf/core/base/Error.h"

#include <windows.h>

namespace tf {

std::string_view toString(ErrorCode code) noexcept {
    switch (code) {
        case ErrorCode::Unknown:            return "unknown";
        case ErrorCode::InvalidArgument:    return "invalid-argument";
        case ErrorCode::NotFound:           return "not-found";
        case ErrorCode::MalformedData:      return "malformed-data";
        case ErrorCode::VerificationFailed: return "verification-failed";
        case ErrorCode::IncompleteInstall:  return "incomplete-install";
        case ErrorCode::Io:                 return "io";
        case ErrorCode::OutOfMemory:        return "out-of-memory";
        case ErrorCode::OutOfDiskSpace:     return "out-of-disk-space";
        case ErrorCode::Network:            return "network";
        case ErrorCode::Unsupported:        return "unsupported";
        case ErrorCode::Cancelled:          return "cancelled";
        case ErrorCode::GpuFailure:         return "gpu-failure";
    }
    return "unknown";
}

namespace {

/// Maps the Win32 status codes we actually branch on. Everything else becomes a
/// generic I/O error - the formatted message carries the detail.
ErrorCode classifyWin32(DWORD code) noexcept {
    switch (code) {
        case ERROR_FILE_NOT_FOUND:
        case ERROR_PATH_NOT_FOUND:
            return ErrorCode::NotFound;
        case ERROR_NOT_ENOUGH_MEMORY:
        case ERROR_OUTOFMEMORY:
            return ErrorCode::OutOfMemory;
        case ERROR_DISK_FULL:
        case ERROR_HANDLE_DISK_FULL:
            return ErrorCode::OutOfDiskSpace;
        case ERROR_OPERATION_ABORTED:
        case ERROR_CANCELLED:
            return ErrorCode::Cancelled;
        case ERROR_INVALID_PARAMETER:
            return ErrorCode::InvalidArgument;
        default:
            return ErrorCode::Io;
    }
}

std::string formatWin32Message(DWORD code) {
    LPWSTR buffer = nullptr;
    const DWORD length = ::FormatMessageW(
            FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
                    FORMAT_MESSAGE_IGNORE_INSERTS,
            nullptr, code, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
            reinterpret_cast<LPWSTR>(&buffer), 0, nullptr);

    if (length == 0 || buffer == nullptr) {
        if (buffer != nullptr) {
            ::LocalFree(buffer);
        }
        return std::format("win32 error {}", code);
    }

    // FormatMessage terminates most system messages with CRLF.
    std::wstring_view wide{buffer, length};
    while (!wide.empty() && (wide.back() == L'\r' || wide.back() == L'\n')) {
        wide.remove_suffix(1);
    }

    std::string narrow;
    const int needed = ::WideCharToMultiByte(CP_UTF8, 0, wide.data(),
                                             static_cast<int>(wide.size()),
                                             nullptr, 0, nullptr, nullptr);
    if (needed > 0) {
        narrow.resize(static_cast<usize>(needed));
        ::WideCharToMultiByte(CP_UTF8, 0, wide.data(), static_cast<int>(wide.size()),
                              narrow.data(), needed, nullptr, nullptr);
    }

    ::LocalFree(buffer);

    if (narrow.empty()) {
        return std::format("win32 error {}", code);
    }
    return std::format("{} (win32 {})", narrow, code);
}

}  // namespace

Error win32Error(unsigned long code, std::string_view what) {
    const auto dwCode = static_cast<DWORD>(code);
    return Error{classifyWin32(dwCode),
                 std::format("{}: {}", what, formatWin32Message(dwCode))};
}

Error lastWin32Error(std::string_view what) {
    return win32Error(::GetLastError(), what);
}

}  // namespace tf
