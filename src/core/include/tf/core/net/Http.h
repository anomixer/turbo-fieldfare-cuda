#pragma once

#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "tf/core/base/Error.h"
#include "tf/core/base/Types.h"

/// HTTPS over WinHTTP.
///
/// WinHTTP rather than libcurl or WinINet: it is in-box on every supported
/// Windows, it works from a service (WinINet does not), and it follows the 302
/// that Hugging Face issues to its CDN without any redirect handling here.
///
/// The interface is deliberately pull-shaped. Installing a 14 GB checkpoint
/// means reading a response body far larger than memory, so a callback that
/// hands over the whole body is not an option, and a push callback would invert
/// control away from the repacker's copy loop.
namespace tf::net {

using HeaderList = std::vector<std::pair<std::string, std::string>>;

/// Status codes this code actually distinguishes.
enum class HttpStatus : u32 {
    Ok = 200,
    PartialContent = 206,
    Unauthorized = 401,
    Forbidden = 403,
    NotFound = 404,
    TooManyRequests = 429,
};

/// A response whose body is small enough to hold: metadata, not weights.
struct HttpResponse {
    u32 statusCode = 0;
    std::vector<u8> body;
    HeaderList headers;

    [[nodiscard]] bool ok() const noexcept { return statusCode >= 200 && statusCode < 300; }
    [[nodiscard]] std::string_view header(std::string_view name) const;
    [[nodiscard]] std::string bodyText() const {
        return std::string{reinterpret_cast<const char*>(body.data()), body.size()};
    }
};

/// An open response body, read incrementally.
///
/// Owns its WinHTTP request handle, so destroying it closes the connection -
/// which is how an abandoned range request is cancelled.
class HttpStream {
public:
    HttpStream();
    ~HttpStream();

    HttpStream(const HttpStream&) = delete;
    HttpStream& operator=(const HttpStream&) = delete;
    HttpStream(HttpStream&&) noexcept;
    HttpStream& operator=(HttpStream&&) noexcept;

    /// Reads up to `destination.size()` bytes. Returns 0 at the end of the
    /// body; a short read is normal and not an error.
    [[nodiscard]] Result<usize> read(MutableByteSpan destination);

    /// Fills `destination` completely, or fails. The repacker wants this: a
    /// short read mid-tensor is a truncated download, not a smaller tensor.
    [[nodiscard]] Status readExact(MutableByteSpan destination);

    [[nodiscard]] u32 statusCode() const noexcept { return statusCode_; }

    /// Bytes the server said the body holds. Zero when it declined to say.
    [[nodiscard]] u64 contentLength() const noexcept { return contentLength_; }

    [[nodiscard]] bool isOpen() const noexcept;
    void close();

private:
    friend class HttpClient;

    struct Impl;
    std::unique_ptr<Impl> impl_;
    u32 statusCode_ = 0;
    u64 contentLength_ = 0;
};

struct HttpOptions {
    /// Sent as User-Agent. Servers behave differently for unknown clients, and
    /// an identifiable one is the polite default.
    std::string userAgent = "turbo-fieldfare-win/0.1";

    /// Seconds. Generous because a cold CDN edge can take a while to answer,
    /// and a spurious timeout mid-install is expensive.
    u32 connectTimeoutSeconds = 30;
    u32 receiveTimeoutSeconds = 120;
};

class HttpClient {
public:
    HttpClient();
    ~HttpClient();

    HttpClient(const HttpClient&) = delete;
    HttpClient& operator=(const HttpClient&) = delete;
    HttpClient(HttpClient&&) noexcept;
    HttpClient& operator=(HttpClient&&) noexcept;

    [[nodiscard]] static Result<HttpClient> create(const HttpOptions& options = {});

    /// Fetches a whole resource into memory. For metadata only - `limitBytes`
    /// refuses anything larger rather than growing without bound.
    [[nodiscard]] Result<HttpResponse> get(std::string_view url,
                                           const HeaderList& headers = {},
                                           u64 limitBytes = 8ull * 1024 * 1024);

    /// Asks only for the headers, which is how a file's size is learned without
    /// downloading it.
    [[nodiscard]] Result<HttpResponse> head(std::string_view url,
                                            const HeaderList& headers = {});

    /// Opens a byte range for streaming. `length` of zero means "to the end".
    ///
    /// A server that ignores Range answers 200 with the whole body; the caller
    /// is told through statusCode() rather than being silently handed the wrong
    /// bytes.
    [[nodiscard]] Result<HttpStream> openRange(std::string_view url, u64 offset,
                                               u64 length = 0,
                                               const HeaderList& headers = {});

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/// Splits a URL into the pieces WinHTTP needs. Exposed because getting this
/// wrong produces confusing failures, and it is worth testing directly.
struct ParsedUrl {
    std::string host;
    std::string path;  ///< includes the query
    u16 port = 443;
    bool secure = true;

    [[nodiscard]] bool valid() const noexcept { return !host.empty(); }
};

[[nodiscard]] Result<ParsedUrl> parseUrl(std::string_view url);

/// Whether a failed attempt is worth repeating. A 404 will not become a 200,
/// but a 429, a 5xx or a dropped connection usually will.
[[nodiscard]] bool isRetryable(u32 statusCode) noexcept;

}  // namespace tf::net
