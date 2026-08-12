#pragma once

#include <expected>
#include <format>
#include <string>
#include <string_view>
#include <utility>

#include "tf/core/base/Types.h"

namespace tf {

enum class ErrorCode {
    Unknown,
    InvalidArgument,
    NotFound,
    /// A file exists but its contents do not match the expected schema/version.
    MalformedData,
    /// Manifest present but hashes or sizes disagree with what is on disk.
    VerificationFailed,
    /// A .gturbo directory is missing its manifest, so the install is partial.
    IncompleteInstall,
    Io,
    OutOfMemory,
    OutOfDiskSpace,
    Network,
    Unsupported,
    Cancelled,
    GpuFailure,
};

[[nodiscard]] std::string_view toString(ErrorCode code) noexcept;

class Error {
public:
    Error() = default;
    Error(ErrorCode code, std::string message)
        : code_(code), message_(std::move(message)) {}

    [[nodiscard]] ErrorCode code() const noexcept { return code_; }
    [[nodiscard]] const std::string& message() const noexcept { return message_; }

    /// Prefixes the message with additional context while preserving the code, so
    /// a low-level read failure can surface as "loading layer_07.bin: <detail>".
    [[nodiscard]] Error wrap(std::string_view context) const {
        return Error{code_, std::format("{}: {}", context, message_)};
    }

    [[nodiscard]] std::string toString() const {
        return std::format("[{}] {}", tf::toString(code_), message_);
    }

private:
    ErrorCode code_ = ErrorCode::Unknown;
    std::string message_;
};

template <class T>
using Result = std::expected<T, Error>;

using Status = std::expected<void, Error>;

template <class... Args>
[[nodiscard]] std::unexpected<Error> makeError(ErrorCode code,
                                               std::format_string<Args...> fmt,
                                               Args&&... args) {
    return std::unexpected(Error{code, std::format(fmt, std::forward<Args>(args)...)});
}

/// Formats the calling thread's GetLastError() value. Call immediately after the
/// failing Win32 call - any intervening API may clobber it.
[[nodiscard]] Error lastWin32Error(std::string_view what);

/// Formats an explicit Win32 error code or HRESULT.
[[nodiscard]] Error win32Error(unsigned long code, std::string_view what);

}  // namespace tf

// Two-level indirection so __LINE__ expands before pasting; without it every
// TF_TRY in a function would declare the same identifier.
#define TF_DETAIL_CONCAT_INNER(a, b) a##b
#define TF_DETAIL_CONCAT(a, b) TF_DETAIL_CONCAT_INNER(a, b)
#define TF_DETAIL_UNIQUE(prefix) TF_DETAIL_CONCAT(prefix, __LINE__)

/// Propagates an error from a Result-returning expression, otherwise binds the
/// value. Free of statement expressions so it works under MSVC.
///
///     TF_TRY(auto manifest, readManifest(path));
///
/// If `expr` contains a top-level comma (a multi-argument template), wrap it in
/// parentheses.
#define TF_TRY(decl, expr) TF_DETAIL_TRY(decl, expr, TF_DETAIL_UNIQUE(tf_try_))

#define TF_DETAIL_TRY(decl, expr, tmp)      \
    auto&& tmp = (expr);                    \
    if (!tmp) {                             \
        return std::unexpected((tmp).error()); \
    }                                       \
    decl = std::move(*(tmp))

/// Propagates an error from a Status-returning expression that yields no value.
#define TF_CHECK(expr)                          \
    do {                                        \
        auto&& tf_check_result = (expr);        \
        if (!tf_check_result) {                 \
            return std::unexpected(tf_check_result.error()); \
        }                                       \
    } while (false)
