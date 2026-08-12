#include "Http.h"

#include <winsock2.h>
#include <ws2tcpip.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <charconv>
#include <format>
#include <thread>

namespace tf::server {
namespace {

/// Winsock needs process-wide setup. A refcounted static keeps that out of the
/// caller's hands without leaking it when the last listener goes away.
class WinsockScope {
public:
    [[nodiscard]] static Status ensure() {
        static const Status once = [] -> Status {
            WSADATA data{};
            const int result = ::WSAStartup(MAKEWORD(2, 2), &data);
            if (result != 0) {
                return makeError(ErrorCode::Network, "WSAStartup failed with {}", result);
            }
            return Status{};
        }();
        return once;
    }
};

[[nodiscard]] Error socketError(std::string_view what) {
    const int code = ::WSAGetLastError();
    char* buffer = nullptr;
    const DWORD length = ::FormatMessageA(
            FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
                    FORMAT_MESSAGE_IGNORE_INSERTS,
            nullptr, static_cast<DWORD>(code), 0, reinterpret_cast<char*>(&buffer), 0, nullptr);

    std::string detail{buffer != nullptr ? buffer : "", length};
    if (buffer != nullptr) {
        ::LocalFree(buffer);
    }
    while (!detail.empty() && (detail.back() == '\n' || detail.back() == '\r')) {
        detail.pop_back();
    }
    if (detail.empty()) {
        detail = std::format("winsock error {}", code);
    }
    return Error{ErrorCode::Network, std::format("{}: {}", what, detail)};
}

[[nodiscard]] bool equalsIgnoringCase(std::string_view a, std::string_view b) {
    return a.size() == b.size() &&
           std::equal(a.begin(), a.end(), b.begin(), [](char x, char y) {
               return std::tolower(static_cast<unsigned char>(x)) ==
                      std::tolower(static_cast<unsigned char>(y));
           });
}

[[nodiscard]] std::string_view trim(std::string_view text) {
    while (!text.empty() && (text.front() == ' ' || text.front() == '\t')) {
        text.remove_prefix(1);
    }
    while (!text.empty() && (text.back() == ' ' || text.back() == '\t' || text.back() == '\r')) {
        text.remove_suffix(1);
    }
    return text;
}

[[nodiscard]] std::string_view reasonPhrase(u32 status) {
    switch (status) {
        case 204: return "No Content";
        case 200: return "OK";
        case 400: return "Bad Request";
        case 404: return "Not Found";
        case 405: return "Method Not Allowed";
        case 408: return "Request Timeout";
        case 413: return "Payload Too Large";
        case 500: return "Internal Server Error";
        case 501: return "Not Implemented";
        case 503: return "Service Unavailable";
        default: return "Unknown";
    }
}

}  // namespace

// ---------------------------------------------------------------------------
// Request and Response
// ---------------------------------------------------------------------------

std::string_view Request::path() const {
    const usize query = target.find('?');
    return query == std::string::npos ? std::string_view{target}
                                      : std::string_view{target}.substr(0, query);
}

std::string_view Request::header(std::string_view name) const {
    const auto found = std::ranges::find_if(headers, [name](const auto& entry) {
        return equalsIgnoringCase(entry.first, name);
    });
    return found == headers.end() ? std::string_view{} : std::string_view{found->second};
}

Response Response::json(u32 status, std::string body) {
    return Response{.status = status, .contentType = "application/json",
                    .body = std::move(body)};
}

Response Response::error(u32 status, std::string_view message, std::string_view type) {
    // The shape OpenAI clients expect. A bare string body renders as an empty
    // message in every SDK, which turns a clear server-side error into a
    // mystery on the client.
    std::string escaped;
    escaped.reserve(message.size() + 16);
    for (const char c : message) {
        switch (c) {
            case '"': escaped += "\\\""; break;
            case '\\': escaped += "\\\\"; break;
            case '\n': escaped += "\\n"; break;
            case '\r': escaped += "\\r"; break;
            case '\t': escaped += "\\t"; break;
            default: escaped += c; break;
        }
    }
    return Response::json(
            status, std::format(R"({{"error":{{"message":"{}","type":"{}","code":null}}}})",
                                escaped, type));
}

// ---------------------------------------------------------------------------
// Connection
// ---------------------------------------------------------------------------

struct Connection::Impl {
    SOCKET socket = INVALID_SOCKET;
    /// Bytes read past the end of the last request, belonging to the next one.
    std::string pending;

    ~Impl() {
        if (socket != INVALID_SOCKET) {
            ::closesocket(socket);
        }
    }

    [[nodiscard]] Status writeAll(std::string_view data) {
        usize sent = 0;
        while (sent < data.size()) {
            const int chunk = ::send(socket, data.data() + sent,
                                     static_cast<int>(std::min<usize>(data.size() - sent,
                                                                      64 * 1024)),
                                     0);
            if (chunk == SOCKET_ERROR) {
                return std::unexpected(socketError("sending the response"));
            }
            sent += static_cast<usize>(chunk);
        }
        return {};
    }

    /// Reads until `pending` holds `terminator`, or the limit is reached.
    [[nodiscard]] Result<usize> readUntil(std::string_view terminator, usize limit) {
        usize searchFrom = 0;
        for (;;) {
            const usize found = pending.find(terminator, searchFrom);
            if (found != std::string::npos) {
                return found;
            }
            if (pending.size() >= limit) {
                return makeError(ErrorCode::InvalidArgument,
                                 "the request headers exceeded {} bytes", limit);
            }
            searchFrom = pending.size() >= terminator.size()
                                 ? pending.size() - terminator.size() + 1
                                 : 0;

            char buffer[8192];
            const int read = ::recv(socket, buffer, sizeof(buffer), 0);
            if (read == 0) {
                return makeError(ErrorCode::Cancelled, "the client closed the connection");
            }
            if (read == SOCKET_ERROR) {
                return std::unexpected(socketError("reading the request"));
            }
            pending.append(buffer, static_cast<usize>(read));
        }
    }

    [[nodiscard]] Status readAtLeast(usize bytes) {
        while (pending.size() < bytes) {
            char buffer[8192];
            const int read = ::recv(socket, buffer, sizeof(buffer), 0);
            if (read == 0) {
                return makeError(ErrorCode::Network,
                                 "the connection ended {} bytes short of the body",
                                 bytes - pending.size());
            }
            if (read == SOCKET_ERROR) {
                return std::unexpected(socketError("reading the request body"));
            }
            pending.append(buffer, static_cast<usize>(read));
        }
        return {};
    }
};

Connection::Connection() : impl_(std::make_unique<Impl>()) {}
Connection::~Connection() = default;
Connection::Connection(Connection&&) noexcept = default;
Connection& Connection::operator=(Connection&&) noexcept = default;

bool Connection::isOpen() const noexcept {
    return impl_ != nullptr && impl_->socket != INVALID_SOCKET;
}

void Connection::close() {
    if (impl_ && impl_->socket != INVALID_SOCKET) {
        ::shutdown(impl_->socket, SD_BOTH);
        ::closesocket(impl_->socket);
        impl_->socket = INVALID_SOCKET;
    }
}

Result<Request> Connection::readRequest() {
    if (!isOpen()) {
        return makeError(ErrorCode::Cancelled, "the connection is closed");
    }

    TF_TRY(const usize headerEnd, impl_->readUntil("\r\n\r\n", kMaxHeaderBytes));

    const std::string_view head{impl_->pending.data(), headerEnd};
    Request request;

    // Request line.
    const usize lineEnd = head.find("\r\n");
    const std::string_view line = head.substr(0, lineEnd);
    const usize firstSpace = line.find(' ');
    const usize secondSpace = firstSpace == std::string_view::npos
                                      ? std::string_view::npos
                                      : line.find(' ', firstSpace + 1);
    if (firstSpace == std::string_view::npos || secondSpace == std::string_view::npos) {
        return makeError(ErrorCode::InvalidArgument, "malformed request line '{}'", line);
    }
    request.method = std::string{line.substr(0, firstSpace)};
    request.target = std::string{line.substr(firstSpace + 1, secondSpace - firstSpace - 1)};
    const std::string_view version = line.substr(secondSpace + 1);

    // HTTP/1.0 defaults to closing; HTTP/1.1 defaults to keeping alive.
    request.keepAlive = version != "HTTP/1.0";

    // Headers.
    usize cursor = lineEnd + 2;
    while (cursor < head.size()) {
        const usize end = head.find("\r\n", cursor);
        const std::string_view entry =
                head.substr(cursor, end == std::string_view::npos ? end : end - cursor);
        const usize colon = entry.find(':');
        if (colon != std::string_view::npos) {
            request.headers.emplace_back(std::string{trim(entry.substr(0, colon))},
                                         std::string{trim(entry.substr(colon + 1))});
        }
        if (end == std::string_view::npos) {
            break;
        }
        cursor = end + 2;
    }

    if (const std::string_view connection = request.header("Connection");
        !connection.empty()) {
        request.keepAlive = !equalsIgnoringCase(connection, "close");
    }

    impl_->pending.erase(0, headerEnd + 4);

    // Body.
    const std::string_view lengthText = request.header("Content-Length");
    usize contentLength = 0;
    if (!lengthText.empty()) {
        const auto* end = lengthText.data() + lengthText.size();
        const auto [stop, code] = std::from_chars(lengthText.data(), end, contentLength);
        if (code != std::errc{} || stop != end) {
            return makeError(ErrorCode::InvalidArgument, "Content-Length '{}' is not a number",
                             lengthText);
        }
        if (contentLength > kMaxBodyBytes) {
            return makeError(ErrorCode::InvalidArgument,
                             "a {} byte body exceeds the {} byte limit", contentLength,
                             kMaxBodyBytes);
        }
    } else if (equalsIgnoringCase(request.header("Transfer-Encoding"), "chunked")) {
        // Nothing an OpenAI client sends is chunked, and guessing would be
        // worse than saying so.
        return makeError(ErrorCode::Unsupported, "chunked request bodies are not supported");
    }

    if (contentLength > 0) {
        TF_CHECK(impl_->readAtLeast(contentLength));
        request.body.assign(impl_->pending, 0, contentLength);
        impl_->pending.erase(0, contentLength);
    }

    return request;
}

Status Connection::sendResponse(const Response& response) {
    std::string head = std::format("HTTP/1.1 {} {}\r\n", response.status,
                                   reasonPhrase(response.status));
    // A browser checks the response to the actual request as well as its
    // preflight. Keep this at the HTTP boundary so errors, models, health,
    // completions, and future endpoints cannot accidentally omit it.
    head += "Access-Control-Allow-Origin: *\r\n";
    head += "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n";
    head += "Access-Control-Allow-Headers: Content-Type, Authorization\r\n";
    if (response.status != 204) {
        head += std::format("Content-Type: {}\r\n", response.contentType);
        head += std::format("Content-Length: {}\r\n", response.body.size());
    }
    for (const auto& [name, value] : response.extraHeaders) {
        head += std::format("{}: {}\r\n", name, value);
    }
    head += "\r\n";
    head += response.body;
    return impl_->writeAll(head);
}

Status Connection::beginEventStream() {
    // No Content-Length: the body ends when the connection does. Cache-Control
    // and the nginx hint keep an intermediary from buffering the whole stream
    // and defeating the point of streaming.
    const std::string head =
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: text/event-stream\r\n"
            "Cache-Control: no-cache\r\n"
            "Connection: close\r\n"
            "X-Accel-Buffering: no\r\n"
            "Access-Control-Allow-Origin: *\r\n"
            "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
            "Access-Control-Allow-Headers: Content-Type, Authorization\r\n"
            "\r\n";
    return impl_->writeAll(head);
}

Status Connection::sendEvent(std::string_view data) {
    // A blank line terminates an event, so a payload containing one would be
    // read as two frames. JSON never contains one, which is why the contract is
    // stated rather than enforced by escaping.
    std::string frame;
    frame.reserve(data.size() + 8);
    frame += "data: ";
    frame += data;
    frame += "\n\n";
    return impl_->writeAll(frame);
}

Status Connection::endEventStream() {
    const Status sent = impl_->writeAll("data: [DONE]\n\n");
    if (sent) {
        // beginEventStream advertises Connection: close. Close after the
        // sentinel so clients waiting for EOF can finish instead of leaving
        // the listener blocked in readRequest() for the next turn.
        close();
    }
    return sent;
}

// ---------------------------------------------------------------------------
// Listener
// ---------------------------------------------------------------------------

struct Listener::Impl {
    SOCKET socket = INVALID_SOCKET;
    u16 port = 0;
    std::atomic<bool> stopping{false};

    ~Impl() {
        if (socket != INVALID_SOCKET) {
            ::closesocket(socket);
        }
    }
};

Listener::Listener() : impl_(std::make_unique<Impl>()) {}
Listener::~Listener() = default;
Listener::Listener(Listener&&) noexcept = default;
Listener& Listener::operator=(Listener&&) noexcept = default;

u16 Listener::port() const noexcept { return impl_ == nullptr ? 0 : impl_->port; }

Result<Listener> Listener::bind(const ListenerOptions& options) {
    TF_CHECK(WinsockScope::ensure());

    addrinfo hints{};
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_protocol = IPPROTO_TCP;
    hints.ai_flags = AI_PASSIVE;

    const std::string service = std::to_string(options.port);
    addrinfo* resolved = nullptr;
    if (::getaddrinfo(options.address.c_str(), service.c_str(), &hints, &resolved) != 0) {
        return std::unexpected(
                socketError(std::format("resolving {}:{}", options.address, options.port)));
    }

    Listener listener;
    listener.impl_->socket =
            ::socket(resolved->ai_family, resolved->ai_socktype, resolved->ai_protocol);
    if (listener.impl_->socket == INVALID_SOCKET) {
        ::freeaddrinfo(resolved);
        return std::unexpected(socketError("creating the listening socket"));
    }

    // SO_EXCLUSIVEADDRUSE rather than SO_REUSEADDR: on Windows the latter lets
    // another process bind the same port and steal connections.
    const int exclusive = 1;
    ::setsockopt(listener.impl_->socket, SOL_SOCKET, SO_EXCLUSIVEADDRUSE,
                 reinterpret_cast<const char*>(&exclusive), sizeof(exclusive));

    if (::bind(listener.impl_->socket, resolved->ai_addr,
               static_cast<int>(resolved->ai_addrlen)) == SOCKET_ERROR) {
        const Error error = socketError(
                std::format("binding {}:{}", options.address, options.port));
        ::freeaddrinfo(resolved);
        return std::unexpected(error);
    }
    ::freeaddrinfo(resolved);

    if (::listen(listener.impl_->socket, options.backlog) == SOCKET_ERROR) {
        return std::unexpected(socketError("listening"));
    }

    // Read back the port, which matters when the caller asked for zero.
    sockaddr_in bound{};
    int boundSize = sizeof(bound);
    if (::getsockname(listener.impl_->socket, reinterpret_cast<sockaddr*>(&bound), &boundSize) ==
        0) {
        listener.impl_->port = ::ntohs(bound.sin_port);
    } else {
        listener.impl_->port = options.port;
    }

    return listener;
}

Status Listener::serve(const Handler& handler) {
    if (impl_->socket == INVALID_SOCKET) {
        return makeError(ErrorCode::InvalidArgument, "the listener is not bound");
    }

    while (!impl_->stopping.load(std::memory_order_relaxed)) {
        const SOCKET accepted = ::accept(impl_->socket, nullptr, nullptr);
        if (accepted == INVALID_SOCKET) {
            if (impl_->stopping.load(std::memory_order_relaxed)) {
                break;
            }
            // One bad accept should not take the server down; the next client
            // may well succeed.
            continue;
        }

        // Nagle would hold back a short SSE frame waiting for more to send,
        // which is exactly the latency streaming exists to avoid.
        const int noDelay = 1;
        ::setsockopt(accepted, IPPROTO_TCP, TCP_NODELAY,
                     reinterpret_cast<const char*>(&noDelay), sizeof(noDelay));

        Connection connection;
        connection.impl_->socket = accepted;

        // Thread per connection. This is a single-user loopback server whose
        // requests are serialized on one GPU anyway, so an IOCP reactor would
        // be complexity bought for nothing.
        std::thread{[&handler, moved = std::move(connection)]() mutable {
            handler(std::move(moved));
        }}.detach();
    }
    return {};
}

void Listener::stop() {
    if (impl_ == nullptr) {
        return;
    }
    impl_->stopping.store(true, std::memory_order_relaxed);
    // Closing the listening socket is what breaks accept() out of its block.
    if (impl_->socket != INVALID_SOCKET) {
        ::closesocket(impl_->socket);
        impl_->socket = INVALID_SOCKET;
    }
}

std::string urlDecode(std::string_view text) {
    const auto hex = [](char c) -> int {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        return -1;
    };

    std::string out;
    out.reserve(text.size());
    for (usize i = 0; i < text.size(); ++i) {
        if (text[i] == '+') {
            out += ' ';
        } else if (text[i] == '%' && i + 2 < text.size()) {
            const int high = hex(text[i + 1]);
            const int low = hex(text[i + 2]);
            if (high >= 0 && low >= 0) {
                out += static_cast<char>(high * 16 + low);
                i += 2;
            } else {
                out += text[i];
            }
        } else {
            out += text[i];
        }
    }
    return out;
}

}  // namespace tf::server
