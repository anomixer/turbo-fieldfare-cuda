#include "tf/core/net/Http.h"

#include <windows.h>
// winhttp.h must follow windows.h.
#include <winhttp.h>

#include <algorithm>
#include <cctype>
#include <cwchar>
#include <format>
#include <utility>

namespace tf::net {
namespace {

/// WinHTTP is a wide-character API and everything here is UTF-8, so these two
/// conversions sit on every boundary.
[[nodiscard]] std::wstring widen(std::string_view text) {
    if (text.empty()) {
        return {};
    }
    const int needed = ::MultiByteToWideChar(CP_UTF8, 0, text.data(),
                                             static_cast<int>(text.size()), nullptr, 0);
    if (needed <= 0) {
        return {};
    }
    std::wstring wide(static_cast<usize>(needed), L'\0');
    ::MultiByteToWideChar(CP_UTF8, 0, text.data(), static_cast<int>(text.size()), wide.data(),
                          needed);
    return wide;
}

[[nodiscard]] std::string narrow(std::wstring_view text) {
    if (text.empty()) {
        return {};
    }
    const int needed = ::WideCharToMultiByte(CP_UTF8, 0, text.data(),
                                             static_cast<int>(text.size()), nullptr, 0, nullptr,
                                             nullptr);
    if (needed <= 0) {
        return {};
    }
    std::string narrowed(static_cast<usize>(needed), '\0');
    ::WideCharToMultiByte(CP_UTF8, 0, text.data(), static_cast<int>(text.size()),
                          narrowed.data(), needed, nullptr, nullptr);
    return narrowed;
}

[[nodiscard]] Error lastError(std::string_view what) {
    const DWORD code = ::GetLastError();
    // WinHTTP's errors are in its own range and FormatMessage needs to be
    // pointed at winhttp.dll to render them.
    char buffer[512] = {};
    const HMODULE module = ::GetModuleHandleW(L"winhttp.dll");
    const DWORD length = ::FormatMessageA(
            FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS |
                    (module != nullptr ? FORMAT_MESSAGE_FROM_HMODULE : 0u),
            module, code, 0, buffer, sizeof(buffer) - 1, nullptr);

    std::string detail{buffer, length};
    while (!detail.empty() && (detail.back() == '\n' || detail.back() == '\r')) {
        detail.pop_back();
    }
    if (detail.empty()) {
        detail = std::format("error {}", code);
    }

    ErrorCode kind = ErrorCode::Network;
    if (code == ERROR_WINHTTP_TIMEOUT || code == ERROR_WINHTTP_NAME_NOT_RESOLVED ||
        code == ERROR_WINHTTP_CANNOT_CONNECT || code == ERROR_WINHTTP_CONNECTION_ERROR) {
        kind = ErrorCode::Network;
    }
    return Error{kind, std::format("{}: {}", what, detail)};
}

/// Closes a WinHTTP handle exactly once.
class Handle {
public:
    Handle() = default;
    explicit Handle(HINTERNET handle) : handle_(handle) {}

    ~Handle() { reset(); }

    Handle(const Handle&) = delete;
    Handle& operator=(const Handle&) = delete;

    Handle(Handle&& other) noexcept : handle_(std::exchange(other.handle_, nullptr)) {}
    Handle& operator=(Handle&& other) noexcept {
        if (this != &other) {
            reset();
            handle_ = std::exchange(other.handle_, nullptr);
        }
        return *this;
    }

    void reset() {
        if (handle_ != nullptr) {
            ::WinHttpCloseHandle(handle_);
            handle_ = nullptr;
        }
    }

    [[nodiscard]] HINTERNET get() const noexcept { return handle_; }
    [[nodiscard]] explicit operator bool() const noexcept { return handle_ != nullptr; }

private:
    HINTERNET handle_ = nullptr;
};

[[nodiscard]] Result<u32> queryStatusCode(HINTERNET request) {
    DWORD status = 0;
    DWORD size = sizeof(status);
    if (!::WinHttpQueryHeaders(request,
                               WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER,
                               WINHTTP_HEADER_NAME_BY_INDEX, &status, &size,
                               WINHTTP_NO_HEADER_INDEX)) {
        return std::unexpected(lastError("reading the response status"));
    }
    return static_cast<u32>(status);
}

/// Content-Length, or zero when the server did not send one - which is normal
/// for a chunked response and is not an error.
[[nodiscard]] u64 queryContentLength(HINTERNET request) {
    wchar_t buffer[64] = {};
    DWORD size = sizeof(buffer);
    if (!::WinHttpQueryHeaders(request, WINHTTP_QUERY_CONTENT_LENGTH,
                               WINHTTP_HEADER_NAME_BY_INDEX, buffer, &size,
                               WINHTTP_NO_HEADER_INDEX)) {
        return 0;
    }
    return std::wcstoull(buffer, nullptr, 10);
}

[[nodiscard]] HeaderList queryAllHeaders(HINTERNET request) {
    DWORD size = 0;
    ::WinHttpQueryHeaders(request, WINHTTP_QUERY_RAW_HEADERS_CRLF,
                          WINHTTP_HEADER_NAME_BY_INDEX, nullptr, &size, WINHTTP_NO_HEADER_INDEX);
    if (size == 0) {
        return {};
    }

    std::wstring raw(size / sizeof(wchar_t), L'\0');
    if (!::WinHttpQueryHeaders(request, WINHTTP_QUERY_RAW_HEADERS_CRLF,
                               WINHTTP_HEADER_NAME_BY_INDEX, raw.data(), &size,
                               WINHTTP_NO_HEADER_INDEX)) {
        return {};
    }

    HeaderList headers;
    const std::string text = narrow(raw);
    usize start = 0;
    while (start < text.size()) {
        const usize end = text.find("\r\n", start);
        const std::string line = text.substr(start, end == std::string::npos ? end : end - start);
        const usize colon = line.find(':');
        if (colon != std::string::npos) {
            std::string name = line.substr(0, colon);
            usize valueStart = colon + 1;
            while (valueStart < line.size() && line[valueStart] == ' ') {
                ++valueStart;
            }
            headers.emplace_back(std::move(name), line.substr(valueStart));
        }
        if (end == std::string::npos) {
            break;
        }
        start = end + 2;
    }
    return headers;
}

}  // namespace

std::string_view HttpResponse::header(std::string_view name) const {
    const auto found = std::ranges::find_if(headers, [name](const auto& entry) {
        return entry.first.size() == name.size() &&
               std::equal(entry.first.begin(), entry.first.end(), name.begin(),
                          [](char a, char b) {
                              return std::tolower(static_cast<unsigned char>(a)) ==
                                     std::tolower(static_cast<unsigned char>(b));
                          });
    });
    return found == headers.end() ? std::string_view{} : std::string_view{found->second};
}

Result<ParsedUrl> parseUrl(std::string_view url) {
    const std::wstring wide = widen(url);

    URL_COMPONENTS components{};
    components.dwStructSize = sizeof(components);
    components.dwSchemeLength = static_cast<DWORD>(-1);
    components.dwHostNameLength = static_cast<DWORD>(-1);
    components.dwUrlPathLength = static_cast<DWORD>(-1);
    components.dwExtraInfoLength = static_cast<DWORD>(-1);

    if (!::WinHttpCrackUrl(wide.c_str(), static_cast<DWORD>(wide.size()), 0, &components)) {
        return makeError(ErrorCode::InvalidArgument, "'{}' is not a usable URL", url);
    }

    ParsedUrl parsed;
    parsed.host = narrow(std::wstring_view{components.lpszHostName, components.dwHostNameLength});
    parsed.port = components.nPort;
    parsed.secure = components.nScheme == INTERNET_SCHEME_HTTPS;
    parsed.path = narrow(std::wstring_view{components.lpszUrlPath, components.dwUrlPathLength});
    parsed.path += narrow(std::wstring_view{components.lpszExtraInfo,
                                            components.dwExtraInfoLength});
    if (parsed.path.empty()) {
        parsed.path = "/";
    }
    if (parsed.host.empty()) {
        return makeError(ErrorCode::InvalidArgument, "'{}' has no host", url);
    }
    return parsed;
}

bool isRetryable(u32 statusCode) noexcept {
    // 408 request timeout and 429 rate limit are explicitly "try again"; 5xx is
    // the server's problem and usually transient. Everything else - a 404, a
    // 403 on a gated repo - will not change on a retry.
    return statusCode == 408 || statusCode == 429 || (statusCode >= 500 && statusCode < 600);
}

// ---------------------------------------------------------------------------
// HttpStream
// ---------------------------------------------------------------------------

struct HttpStream::Impl {
    Handle connection;
    Handle request;
};

HttpStream::HttpStream() : impl_(std::make_unique<Impl>()) {}
HttpStream::~HttpStream() = default;
HttpStream::HttpStream(HttpStream&&) noexcept = default;
HttpStream& HttpStream::operator=(HttpStream&&) noexcept = default;

bool HttpStream::isOpen() const noexcept {
    return impl_ != nullptr && static_cast<bool>(impl_->request);
}

void HttpStream::close() {
    if (impl_) {
        impl_->request.reset();
        impl_->connection.reset();
    }
}

Result<usize> HttpStream::read(MutableByteSpan destination) {
    if (!isOpen()) {
        return makeError(ErrorCode::Io, "reading from a closed HTTP stream");
    }
    if (destination.empty()) {
        return usize{0};
    }

    DWORD read = 0;
    if (!::WinHttpReadData(impl_->request.get(), destination.data(),
                           static_cast<DWORD>(std::min<u64>(destination.size(), 0x7FFFFFFF)),
                           &read)) {
        return std::unexpected(lastError("reading the response body"));
    }
    return static_cast<usize>(read);
}

Status HttpStream::readExact(MutableByteSpan destination) {
    usize filled = 0;
    while (filled < destination.size()) {
        TF_TRY(const usize read, this->read(destination.subspan(filled)));
        if (read == 0) {
            return makeError(ErrorCode::Network,
                             "the connection ended {} bytes short of the {} expected",
                             destination.size() - filled, destination.size());
        }
        filled += read;
    }
    return {};
}

// ---------------------------------------------------------------------------
// HttpClient
// ---------------------------------------------------------------------------

struct HttpClient::Impl {
    Handle session;
    HttpOptions options;
};

HttpClient::HttpClient() : impl_(std::make_unique<Impl>()) {}
HttpClient::~HttpClient() = default;
HttpClient::HttpClient(HttpClient&&) noexcept = default;
HttpClient& HttpClient::operator=(HttpClient&&) noexcept = default;

Result<HttpClient> HttpClient::create(const HttpOptions& options) {
    HttpClient client;
    client.impl_->options = options;

    // WINHTTP_ACCESS_TYPE_AUTOMATIC_PROXY picks up the system proxy, which is
    // what a corporate machine needs and costs nothing elsewhere.
    const std::wstring agent = widen(options.userAgent);
    client.impl_->session = Handle{::WinHttpOpen(agent.c_str(),
                                                 WINHTTP_ACCESS_TYPE_AUTOMATIC_PROXY,
                                                 WINHTTP_NO_PROXY_NAME, WINHTTP_NO_PROXY_BYPASS,
                                                 0)};
    if (!client.impl_->session) {
        return std::unexpected(lastError("opening a WinHTTP session"));
    }

    const auto milliseconds = [](u32 seconds) { return static_cast<int>(seconds) * 1000; };
    ::WinHttpSetTimeouts(client.impl_->session.get(), milliseconds(options.connectTimeoutSeconds),
                         milliseconds(options.connectTimeoutSeconds),
                         milliseconds(options.receiveTimeoutSeconds),
                         milliseconds(options.receiveTimeoutSeconds));

    // Follow the 302 Hugging Face issues to its CDN. Cross-host redirects are
    // the norm here, so the default policy is what is wanted; only an
    // https-to-http downgrade stays refused.
    DWORD policy = WINHTTP_OPTION_REDIRECT_POLICY_DISALLOW_HTTPS_TO_HTTP;
    ::WinHttpSetOption(client.impl_->session.get(), WINHTTP_OPTION_REDIRECT_POLICY, &policy,
                       sizeof(policy));

    return client;
}

namespace {

/// Sends one request and receives the response headers, leaving the body
/// unread. Shared by get, head and openRange.
[[nodiscard]] Status sendRequest(HINTERNET session, const ParsedUrl& url,
                                 std::wstring_view method, const HeaderList& headers,
                                 std::string_view rangeHeader, Handle& connection,
                                 Handle& request) {
    connection = Handle{::WinHttpConnect(session, widen(url.host).c_str(), url.port, 0)};
    if (!connection) {
        return std::unexpected(lastError(std::format("connecting to {}", url.host)));
    }

    const DWORD flags = url.secure ? WINHTTP_FLAG_SECURE : 0u;
    request = Handle{::WinHttpOpenRequest(connection.get(), method.data(),
                                          widen(url.path).c_str(), nullptr,
                                          WINHTTP_NO_REFERER,
                                          WINHTTP_DEFAULT_ACCEPT_TYPES, flags)};
    if (!request) {
        return std::unexpected(lastError("opening the request"));
    }

    std::string combined;
    for (const auto& [name, value] : headers) {
        combined += std::format("{}: {}\r\n", name, value);
    }
    if (!rangeHeader.empty()) {
        combined += std::format("Range: {}\r\n", rangeHeader);
    }
    if (!combined.empty()) {
        const std::wstring wide = widen(combined);
        if (!::WinHttpAddRequestHeaders(request.get(), wide.c_str(),
                                        static_cast<DWORD>(wide.size()),
                                        WINHTTP_ADDREQ_FLAG_ADD)) {
            return std::unexpected(lastError("adding request headers"));
        }
    }

    if (!::WinHttpSendRequest(request.get(), WINHTTP_NO_ADDITIONAL_HEADERS, 0,
                              WINHTTP_NO_REQUEST_DATA, 0, 0, 0)) {
        return std::unexpected(lastError(std::format("sending the request to {}", url.host)));
    }
    if (!::WinHttpReceiveResponse(request.get(), nullptr)) {
        return std::unexpected(lastError("waiting for the response"));
    }
    return {};
}

}  // namespace

Result<HttpResponse> HttpClient::get(std::string_view url, const HeaderList& headers,
                                     u64 limitBytes) {
    TF_TRY(const ParsedUrl parsed, parseUrl(url));

    Handle connection;
    Handle request;
    TF_CHECK(sendRequest(impl_->session.get(), parsed, L"GET", headers, {}, connection,
                         request));

    HttpResponse response;
    TF_TRY(response.statusCode, queryStatusCode(request.get()));
    response.headers = queryAllHeaders(request.get());

    const u64 declared = queryContentLength(request.get());
    if (declared > limitBytes) {
        return makeError(ErrorCode::InvalidArgument,
                         "{} returned {} bytes, past the {} byte limit for a metadata fetch",
                         url, declared, limitBytes);
    }

    // Read until the end regardless of Content-Length: a chunked response
    // declares none.
    std::vector<u8> body;
    body.reserve(static_cast<usize>(std::min<u64>(declared, 1ull << 20)));
    std::vector<u8> chunk(64 * 1024);
    for (;;) {
        DWORD read = 0;
        if (!::WinHttpReadData(request.get(), chunk.data(),
                               static_cast<DWORD>(chunk.size()), &read)) {
            return std::unexpected(lastError("reading the response body"));
        }
        if (read == 0) {
            break;
        }
        if (body.size() + read > limitBytes) {
            return makeError(ErrorCode::InvalidArgument,
                             "{} returned more than the {} byte limit for a metadata fetch",
                             url, limitBytes);
        }
        body.insert(body.end(), chunk.begin(), chunk.begin() + read);
    }

    response.body = std::move(body);
    return response;
}

Result<HttpResponse> HttpClient::head(std::string_view url, const HeaderList& headers) {
    TF_TRY(const ParsedUrl parsed, parseUrl(url));

    Handle connection;
    Handle request;
    TF_CHECK(sendRequest(impl_->session.get(), parsed, L"HEAD", headers, {}, connection,
                         request));

    HttpResponse response;
    TF_TRY(response.statusCode, queryStatusCode(request.get()));
    response.headers = queryAllHeaders(request.get());
    return response;
}

Result<HttpStream> HttpClient::openRange(std::string_view url, u64 offset, u64 length,
                                         const HeaderList& headers) {
    TF_TRY(const ParsedUrl parsed, parseUrl(url));

    // An open-ended range is "everything from here", which is what a resumed
    // sequential download wants.
    const std::string range = length == 0
                                      ? std::format("bytes={}-", offset)
                                      : std::format("bytes={}-{}", offset, offset + length - 1);

    HttpStream stream;
    TF_CHECK(sendRequest(impl_->session.get(), parsed, L"GET", headers, range,
                         stream.impl_->connection, stream.impl_->request));

    TF_TRY(stream.statusCode_, queryStatusCode(stream.impl_->request.get()));
    stream.contentLength_ = queryContentLength(stream.impl_->request.get());
    return stream;
}

}  // namespace tf::net
