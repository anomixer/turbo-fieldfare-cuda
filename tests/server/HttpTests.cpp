// HTTP parsing and the server socket.
//
// Request parsing runs against a real loopback connection rather than a string,
// because the interesting failures are about how bytes arrive: a request split
// across packets, two pipelined into one read, a body that has not landed yet.
// Feeding a complete buffer to a parser tests none of that.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <atomic>
#include <string>
#include <thread>

#include <winsock2.h>
#include <ws2tcpip.h>

#include "Http.h"

using namespace tf;
using namespace tf::server;

namespace {

/// A client socket to the listener under test.
class Client {
public:
    explicit Client(u16 port) {
        socket_ = ::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
        REQUIRE(socket_ != INVALID_SOCKET);

        sockaddr_in address{};
        address.sin_family = AF_INET;
        address.sin_port = ::htons(port);
        ::inet_pton(AF_INET, "127.0.0.1", &address.sin_addr);
        REQUIRE(::connect(socket_, reinterpret_cast<sockaddr*>(&address), sizeof(address)) ==
                0);
    }

    ~Client() {
        if (socket_ != INVALID_SOCKET) {
            ::closesocket(socket_);
        }
    }

    Client(const Client&) = delete;
    Client& operator=(const Client&) = delete;

    void send(std::string_view data) const {
        usize sent = 0;
        while (sent < data.size()) {
            const int chunk = ::send(socket_, data.data() + sent,
                                     static_cast<int>(data.size() - sent), 0);
            REQUIRE(chunk > 0);
            sent += static_cast<usize>(chunk);
        }
    }

    /// Reads until the peer closes, which is how the tests get a whole
    /// response without parsing framing.
    [[nodiscard]] std::string readAll() const {
        std::string out;
        char buffer[4096];
        for (;;) {
            const int read = ::recv(socket_, buffer, sizeof(buffer), 0);
            if (read <= 0) {
                break;
            }
            out.append(buffer, static_cast<usize>(read));
        }
        return out;
    }

    void halfClose() const { ::shutdown(socket_, SD_SEND); }

private:
    SOCKET socket_ = INVALID_SOCKET;
};

/// Runs a listener on an ephemeral port for the duration of a test, handing
/// every request to `respond`.
class TestServer {
public:
    using Responder = std::function<void(const Request&, Connection&)>;

    explicit TestServer(Responder responder) : responder_(std::move(responder)) {
        auto bound = Listener::bind(ListenerOptions{.address = "127.0.0.1", .port = 0});
        REQUIRE(bound.has_value());
        listener_ = std::make_unique<Listener>(std::move(*bound));
        port_ = listener_->port();
        REQUIRE(port_ != 0);

        thread_ = std::thread{[this] {
            static_cast<void>(listener_->serve([this](Connection connection) {
                while (connection.isOpen()) {
                    auto request = connection.readRequest();
                    if (!request) {
                        break;
                    }
                    lastError_.clear();
                    responder_(*request, connection);
                    if (!request->keepAlive) {
                        break;
                    }
                }
                connection.close();
            }));
        }};
    }

    ~TestServer() {
        listener_->stop();
        if (thread_.joinable()) {
            thread_.join();
        }
    }

    [[nodiscard]] u16 port() const noexcept { return port_; }

private:
    Responder responder_;
    std::unique_ptr<Listener> listener_;
    std::thread thread_;
    std::string lastError_;
    u16 port_ = 0;
};

/// Echoes what the server parsed, so a test can assert on it.
void echoRequest(const Request& request, Connection& connection) {
    std::string body = request.method;
    body += ' ';
    body += request.path();
    body += '|';
    body += request.header("X-Test");
    body += '|';
    body += request.body;
    static_cast<void>(connection.sendResponse(
            Response{.status = 200, .contentType = "text/plain", .body = std::move(body)}));
}

}  // namespace

TEST_CASE("a simple request is parsed and answered", "[http]") {
    TestServer server{echoRequest};
    Client client{server.port()};

    client.send("GET /v1/models HTTP/1.1\r\nHost: x\r\nX-Test: hello\r\n\r\n");
    client.halfClose();

    const std::string response = client.readAll();
    CHECK_THAT(response, Catch::Matchers::ContainsSubstring("HTTP/1.1 200 OK"));
    CHECK_THAT(response, Catch::Matchers::ContainsSubstring("GET /v1/models|hello|"));
}

TEST_CASE("a query string is not part of the path", "[http]") {
    TestServer server{echoRequest};
    Client client{server.port()};

    // Routing compares the path, so a query would make every route miss.
    client.send("GET /v1/models?limit=2&x=y HTTP/1.1\r\nHost: x\r\n\r\n");
    client.halfClose();

    CHECK_THAT(client.readAll(), Catch::Matchers::ContainsSubstring("GET /v1/models|"));
}

TEST_CASE("headers are matched without regard to case", "[http]") {
    TestServer server{echoRequest};
    Client client{server.port()};

    // Clients capitalize headers however they like.
    client.send("GET / HTTP/1.1\r\nHost: x\r\nx-TEST: found\r\n\r\n");
    client.halfClose();

    CHECK_THAT(client.readAll(), Catch::Matchers::ContainsSubstring("|found|"));
}

TEST_CASE("a body is read according to Content-Length", "[http]") {
    TestServer server{echoRequest};
    Client client{server.port()};

    client.send(
            "POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Length: 13\r\n\r\n"
            "{\"a\":\"hello\"}");
    client.halfClose();

    CHECK_THAT(client.readAll(), Catch::Matchers::ContainsSubstring("|{\"a\":\"hello\"}"));
}

TEST_CASE("a request split across packets is reassembled", "[http]") {
    TestServer server{echoRequest};
    Client client{server.port()};

    // TCP does not preserve write boundaries, so the parser must not assume a
    // request arrives whole - and with a large prompt it will not.
    client.send("POST /v1/completions HTTP/1.1\r\nHost: x\r\nCont");
    std::this_thread::sleep_for(std::chrono::milliseconds{20});
    client.send("ent-Length: 9\r\n\r\n{\"p\":");
    std::this_thread::sleep_for(std::chrono::milliseconds{20});
    client.send("\"a\"}");
    client.halfClose();

    // The echo is method, path, the X-Test header (absent here) and the body.
    CHECK_THAT(client.readAll(),
               Catch::Matchers::ContainsSubstring("POST /v1/completions||{\"p\":\"a\"}"));
}

TEST_CASE("two pipelined requests are answered in order", "[http]") {
    TestServer server{echoRequest};
    Client client{server.port()};

    // Arriving in one read: the parser has to keep the second request's bytes
    // rather than discard whatever followed the first.
    client.send(
            "GET /one HTTP/1.1\r\nHost: x\r\n\r\n"
            "GET /two HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n");

    const std::string response = client.readAll();
    const usize one = response.find("GET /one|");
    const usize two = response.find("GET /two|");
    REQUIRE(one != std::string::npos);
    REQUIRE(two != std::string::npos);
    CHECK(one < two);
}

TEST_CASE("keep-alive serves several requests on one connection", "[http]") {
    std::atomic<int> served{0};
    TestServer server{[&](const Request& request, Connection& connection) {
        served.fetch_add(1);
        echoRequest(request, connection);
    }};
    Client client{server.port()};

    client.send("GET /a HTTP/1.1\r\nHost: x\r\n\r\n");
    std::this_thread::sleep_for(std::chrono::milliseconds{30});
    client.send("GET /b HTTP/1.1\r\nHost: x\r\n\r\n");
    std::this_thread::sleep_for(std::chrono::milliseconds{30});
    client.send("GET /c HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n");

    const std::string response = client.readAll();
    CHECK(served.load() == 3);
    CHECK_THAT(response, Catch::Matchers::ContainsSubstring("GET /c|"));
}

TEST_CASE("HTTP/1.0 closes unless it asks otherwise", "[http]") {
    TestServer server{echoRequest};

    {
        Client client{server.port()};
        client.send("GET /a HTTP/1.0\r\n\r\n");
        // No half-close: the server must close on its own, or readAll blocks
        // forever.
        CHECK_THAT(client.readAll(), Catch::Matchers::ContainsSubstring("GET /a|"));
    }
}

TEST_CASE("an oversized header block is refused rather than buffered", "[http]") {
    TestServer server{echoRequest};
    Client client{server.port()};

    // A client that never sends the blank line must not be able to make the
    // server allocate without bound.
    std::string flood = "GET / HTTP/1.1\r\nHost: x\r\nX-Big: ";
    flood.append(kMaxHeaderBytes + 1024, 'a');
    client.send(flood);

    // The server gives up and closes; the test only needs it not to hang or
    // grow forever.
    const std::string response = client.readAll();
    CHECK(response.empty());
}

TEST_CASE("event stream frames are shaped for SSE", "[http]") {
    TestServer server{[](const Request&, Connection& connection) {
        REQUIRE(connection.beginEventStream().has_value());
        REQUIRE(connection.sendEvent(R"({"a":1})").has_value());
        REQUIRE(connection.sendEvent(R"({"a":2})").has_value());
        REQUIRE(connection.endEventStream().has_value());
        connection.close();
    }};
    Client client{server.port()};

    client.send("POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\n\r\n");
    const std::string response = client.readAll();

    CHECK_THAT(response, Catch::Matchers::ContainsSubstring("Content-Type: text/event-stream"));
    // No Content-Length: the body ends with the connection.
    CHECK(response.find("Content-Length") == std::string::npos);
    CHECK_THAT(response, Catch::Matchers::ContainsSubstring("data: {\"a\":1}\n\n"));
    // The sentinel every OpenAI client waits for before finishing.
    CHECK_THAT(response, Catch::Matchers::ContainsSubstring("data: [DONE]\n\n"));
}

TEST_CASE("responses carry the length of what they send", "[http]") {
    TestServer server{[](const Request&, Connection& connection) {
        static_cast<void>(connection.sendResponse(
                Response{.status = 404, .contentType = "application/json",
                         .body = R"({"error":"nope"})"}));
    }};
    Client client{server.port()};

    client.send("GET /missing HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n");
    const std::string response = client.readAll();

    CHECK_THAT(response, Catch::Matchers::ContainsSubstring("HTTP/1.1 404 Not Found"));
    CHECK_THAT(response, Catch::Matchers::ContainsSubstring("Content-Length: 16"));
}

TEST_CASE("responses include CORS headers", "[http]") {
    TestServer server{[](const Request&, Connection& connection) {
        static_cast<void>(connection.sendResponse(Response::json(200, "{}")));
    }};
    Client client{server.port()};

    client.send("GET /v1/models HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n");
    const std::string response = client.readAll();

    CHECK_THAT(response, Catch::Matchers::ContainsSubstring("Access-Control-Allow-Origin: *"));
    CHECK_THAT(response, Catch::Matchers::ContainsSubstring(
                                 "Access-Control-Allow-Methods: GET, POST, OPTIONS"));
    CHECK_THAT(response, Catch::Matchers::ContainsSubstring(
                                 "Access-Control-Allow-Headers: Content-Type, Authorization"));
}

TEST_CASE("a no-content response has no body framing", "[http]") {
    TestServer server{[](const Request&, Connection& connection) {
        static_cast<void>(connection.sendResponse(Response{.status = 204}));
    }};
    Client client{server.port()};

    client.send("OPTIONS /v1/chat/completions HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n");
    const std::string response = client.readAll();

    CHECK_THAT(response, Catch::Matchers::StartsWith("HTTP/1.1 204 No Content\r\n"));
    CHECK(response.find("\r\nContent-Length:") == std::string::npos);
    CHECK(response.find("\r\nContent-Type:") == std::string::npos);
}

TEST_CASE("binding an ephemeral port reports which one", "[http]") {
    auto listener = Listener::bind(ListenerOptions{.address = "127.0.0.1", .port = 0});
    REQUIRE(listener.has_value());
    // Zero means "pick one", and the caller has to be told which.
    CHECK(listener->port() != 0);
}

TEST_CASE("a port already in use is reported, not silently shared", "[http]") {
    auto first = Listener::bind(ListenerOptions{.address = "127.0.0.1", .port = 0});
    REQUIRE(first.has_value());

    // SO_EXCLUSIVEADDRUSE rather than SO_REUSEADDR: on Windows the latter would
    // let this succeed and then steal connections from the first listener.
    auto second = Listener::bind(ListenerOptions{.address = "127.0.0.1",
                                                 .port = first->port()});
    CHECK_FALSE(second.has_value());
}

TEST_CASE("percent-encoded characters decode", "[http]") {
    CHECK(urlDecode("hello%20world") == "hello world");
    CHECK(urlDecode("a+b") == "a b");
    CHECK(urlDecode("100%25") == "100%");
    // A stray percent is left alone rather than eating the next characters.
    CHECK(urlDecode("50%") == "50%");
    CHECK(urlDecode("%zz") == "%zz");
}
