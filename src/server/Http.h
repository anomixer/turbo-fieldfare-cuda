#pragma once

#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "tf/core/base/Error.h"
#include "tf/core/base/Types.h"

/// A minimal HTTP/1.1 server over Winsock.
///
/// Scoped to what an OpenAI-compatible endpoint on loopback needs: no TLS, no
/// chunked request bodies, no multipart. Writing that much is less work than
/// taking a dependency, and it keeps the zero-dependency line the rest of the
/// project holds.
namespace tf::server {

using HeaderList = std::vector<std::pair<std::string, std::string>>;

struct Request {
    std::string method;
    /// Path and query as sent, e.g. "/v1/models?limit=1".
    std::string target;
    std::string body;
    HeaderList headers;
    /// False when the client asked to close, or sent HTTP/1.0 without
    /// keep-alive.
    bool keepAlive = true;

    /// The target with any query string removed.
    [[nodiscard]] std::string_view path() const;

    /// Case-insensitive lookup; empty when absent.
    [[nodiscard]] std::string_view header(std::string_view name) const;
};

struct Response {
    u32 status = 200;
    std::string contentType = "application/json";
    std::string body;
    HeaderList extraHeaders;

    [[nodiscard]] static Response json(u32 status, std::string body);

    /// An OpenAI-shaped error envelope. Clients surface `error.message`, so a
    /// bare string body would show up as nothing useful.
    [[nodiscard]] static Response error(u32 status, std::string_view message,
                                        std::string_view type = "invalid_request_error");
};

/// Largest request this server will accept, to bound what a bad client can make
/// it allocate. A prompt has to fit comfortably; weights never travel this way.
inline constexpr usize kMaxHeaderBytes = 64 * 1024;
inline constexpr usize kMaxBodyBytes = 32 * 1024 * 1024;

/// One accepted connection.
///
/// Reads requests, writes responses, and can switch into server-sent events for
/// a streaming completion. Once streaming starts the connection is committed to
/// it: the status line is already on the wire, so a later failure can only be
/// reported inside the stream, never as an HTTP status.
class Connection {
public:
    Connection();
    ~Connection();

    Connection(const Connection&) = delete;
    Connection& operator=(const Connection&) = delete;
    Connection(Connection&&) noexcept;
    Connection& operator=(Connection&&) noexcept;

    /// Reads the next request. Returns a Cancelled error when the peer closed
    /// cleanly between requests, which is ordinary keep-alive behaviour rather
    /// than a fault.
    [[nodiscard]] Result<Request> readRequest();

    [[nodiscard]] Status sendResponse(const Response& response);

    /// Switches to text/event-stream and writes the headers.
    [[nodiscard]] Status beginEventStream();

    /// Writes one `data:` frame. The payload is sent verbatim, so it must not
    /// contain a blank line.
    [[nodiscard]] Status sendEvent(std::string_view data);

    /// Writes the `[DONE]` sentinel every OpenAI client waits for.
    [[nodiscard]] Status endEventStream();

    [[nodiscard]] bool isOpen() const noexcept;
    void close();

private:
    friend class Listener;

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

struct ListenerOptions {
    /// Loopback by default. This process holds a 13 GiB model and answers
    /// unauthenticated requests, so it has no business on a public interface
    /// unless the operator says so deliberately.
    std::string address = "127.0.0.1";
    u16 port = 8080;
    /// Pending connections the kernel will hold before refusing.
    int backlog = 16;
};

/// Accepts connections and hands each to a handler on its own thread.
class Listener {
public:
    Listener();
    ~Listener();

    Listener(const Listener&) = delete;
    Listener& operator=(const Listener&) = delete;
    Listener(Listener&&) noexcept;
    Listener& operator=(Listener&&) noexcept;

    [[nodiscard]] static Result<Listener> bind(const ListenerOptions& options);

    /// The port actually bound, which differs from the requested one when the
    /// caller asked for zero.
    [[nodiscard]] u16 port() const noexcept;

    /// Blocks, accepting until stop() is called. Each connection runs the
    /// handler on a detached thread.
    using Handler = std::function<void(Connection)>;
    [[nodiscard]] Status serve(const Handler& handler);

    /// Wakes serve() out of accept(). Safe from another thread or a console
    /// control handler.
    void stop();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/// Percent-decodes a query or path segment.
[[nodiscard]] std::string urlDecode(std::string_view text);

}  // namespace tf::server
