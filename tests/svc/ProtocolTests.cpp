// The decode service protocol and its transport.
//
// The protocol is the seam between the GUI and the model, and both sides are
// separately compiled binaries - so a field that round-trips wrong shows up as
// a setting the GUI appears to apply and the service silently ignores. Every
// message therefore round-trips here, and the pipe is exercised over a real
// named pipe rather than a mock: the interesting failures are about framing and
// disconnects, which a mock cannot have.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <windows.h>

#include <atomic>
#include <format>
#include <string>
#include <thread>
#include <vector>

#include "Pipe.h"
#include "Protocol.h"

using namespace tf;
using namespace tf::svc;

namespace {

/// A pipe name unique to this test run, so a stray service or a parallel run
/// cannot collide.
[[nodiscard]] std::string testPipeName() {
    static std::atomic<int> counter{0};
    return std::format(R"(\\.\pipe\tf-test-{}-{})", ::GetCurrentProcessId(),
                       counter.fetch_add(1));
}

/// Round-trips a message and returns what came back.
[[nodiscard]] Message roundTrip(const Message& message) {
    auto decoded = Message::decode(message.encode());
    REQUIRE(decoded.has_value());
    return std::move(*decoded);
}

}  // namespace

// ---------------------------------------------------------------------------
// Encoding
// ---------------------------------------------------------------------------

TEST_CASE("every message kind has a name that round-trips", "[protocol]") {
    for (const MessageKind kind :
         {MessageKind::Hello, MessageKind::Generate, MessageKind::Cancel, MessageKind::Reset,
          MessageKind::Status, MessageKind::Shutdown, MessageKind::Ready, MessageKind::Token,
          MessageKind::Thinking, MessageKind::Done, MessageKind::Error,
          MessageKind::StatusReport}) {
        const std::string_view name = toString(kind);
        INFO("kind " << name);
        CHECK(name != "?");
        auto parsed = parseKind(name);
        REQUIRE(parsed.has_value());
        CHECK(*parsed == kind);
    }
}

TEST_CASE("an unknown message type is refused", "[protocol]") {
    CHECK_FALSE(parseKind("teleport").has_value());
    CHECK_FALSE(Message::decode(R"({"type":"teleport"})").has_value());
    CHECK_FALSE(Message::decode("not json").has_value());
    // A message with no type at all cannot be dispatched.
    CHECK_FALSE(Message::decode(R"({"id":1})").has_value());
}

TEST_CASE("a chat generate request round-trips every field", "[protocol]") {
    Message message{.kind = MessageKind::Generate};
    message.generate.id = 42;
    message.generate.chat = true;
    message.generate.messages = {ChatMessage{ChatRole::System, "be brief"},
                                 ChatMessage{ChatRole::User, "hello"},
                                 ChatMessage{ChatRole::Model, "hi"}};
    message.generate.sampling.temperature = 0.35f;
    message.generate.sampling.topK = 32;
    message.generate.sampling.topP = 0.88f;
    message.generate.sampling.repetitionPenalty = 1.15f;
    message.generate.sampling.repetitionWindow = 96;
    message.generate.sampling.seed = 987654321;
    message.generate.maxTokens = 777;
    message.generate.stopStrings = {"\n\nUser:", "END"};
    message.generate.includeThinking = true;

    const Message back = roundTrip(message);
    REQUIRE(back.kind == MessageKind::Generate);
    CHECK(back.generate.id == 42);
    CHECK(back.generate.chat);
    REQUIRE(back.generate.messages.size() == 3);
    CHECK(back.generate.messages[0].role == ChatRole::System);
    CHECK(back.generate.messages[1].content == "hello");
    CHECK(back.generate.messages[2].role == ChatRole::Model);
    CHECK(back.generate.sampling.temperature == 0.35f);
    CHECK(back.generate.sampling.topK == 32);
    CHECK(back.generate.sampling.topP == 0.88f);
    CHECK(back.generate.sampling.repetitionPenalty == 1.15f);
    CHECK(back.generate.sampling.repetitionWindow == 96);
    CHECK(back.generate.sampling.seed == 987654321);
    CHECK(back.generate.maxTokens == 777);
    REQUIRE(back.generate.stopStrings.size() == 2);
    CHECK(back.generate.stopStrings[0] == "\n\nUser:");
    CHECK(back.generate.includeThinking);
}

TEST_CASE("a raw generate request carries a prompt", "[protocol]") {
    Message message{.kind = MessageKind::Generate};
    message.generate.id = 1;
    message.generate.chat = false;
    message.generate.prompt = "once upon a";

    const Message back = roundTrip(message);
    CHECK_FALSE(back.generate.chat);
    CHECK(back.generate.prompt == "once upon a");
}

TEST_CASE("a generate request with nothing to generate from is refused",
          "[protocol]") {
    CHECK_FALSE(Message::decode(R"({"type":"generate","chat":true,"messages":[]})")
                        .has_value());
    CHECK_FALSE(Message::decode(R"({"type":"generate","chat":false,"prompt":""})")
                        .has_value());
    // Invalid sampling must not reach the generator.
    CHECK_FALSE(Message::decode(
                        R"({"type":"generate","chat":false,"prompt":"x","temperature":-1})")
                        .has_value());
}

TEST_CASE("omitted generate fields keep their defaults", "[protocol]") {
    // A front end sends what it cares about; everything else has to inherit
    // rather than become zero.
    auto decoded = Message::decode(R"({"type":"generate","chat":false,"prompt":"x"})");
    REQUIRE(decoded.has_value());
    CHECK(decoded->generate.sampling.temperature ==
          runtime::SamplingParams{}.temperature);
    CHECK(decoded->generate.sampling.topK == runtime::SamplingParams{}.topK);
    CHECK(decoded->generate.maxTokens == GenerateRequest{}.maxTokens);
}

TEST_CASE("text carrying quotes and newlines survives the round trip",
          "[protocol]") {
    // Tokens are arbitrary model output: they will contain both.
    const Message back = roundTrip(makeToken(7, "line one\n\"quoted\"\tand\\a backslash"));
    CHECK(back.id == 7);
    CHECK(back.text == "line one\n\"quoted\"\tand\\a backslash");
}

TEST_CASE("multi-byte text survives the round trip", "[protocol]") {
    // A token can split a UTF-8 sequence, so whatever arrives must survive
    // being encoded as JSON and decoded again unchanged.
    const std::string text = "\xE6\x97\xA5\xE6\x9C\xAC\xE8\xAA\x9E and \xF0\x9F\x8E\x89";
    CHECK(roundTrip(makeToken(1, text)).text == text);
}

TEST_CASE("ready, done and status round-trip their numbers", "[protocol]") {
    const Message ready = roundTrip(makeReady(ReadyInfo{.model = "gemma4",
                                                        .contextLength = 4096,
                                                        .device = "RTX 5060 Ti",
                                                        .fullyResident = false}));
    CHECK(ready.ready.model == "gemma4");
    CHECK(ready.ready.contextLength == 4096);
    CHECK(ready.ready.device == "RTX 5060 Ti");
    CHECK_FALSE(ready.ready.fullyResident);

    const Message done = roundTrip(makeDone(DoneInfo{.id = 3,
                                                     .reason = "stop token",
                                                     .promptTokens = 61,
                                                     .cachedPromptTokens = 43,
                                                     .generatedTokens = 9,
                                                     .prefillSeconds = 0.25,
                                                     .decodeSeconds = 1.5}));
    CHECK(done.done.id == 3);
    CHECK(done.done.reason == "stop token");
    CHECK(done.done.cachedPromptTokens == 43);
    CHECK(done.done.prefillSeconds == 0.25);

    const Message status = roundTrip(makeStatusReport(StatusInfo{.model = "gemma4",
                                                                 .contextLength = 4096,
                                                                 .position = 120,
                                                                 .requests = 5,
                                                                 .expertHitRate = 0.499,
                                                                 .busy = true}));
    CHECK(status.status.position == 120);
    CHECK(status.status.expertHitRate == 0.499);
    CHECK(status.status.busy);
}

TEST_CASE("framing prefixes a little-endian length", "[protocol]") {
    const std::string framed = frame("hello");
    REQUIRE(framed.size() == 9);
    CHECK(static_cast<u8>(framed[0]) == 5);
    CHECK(static_cast<u8>(framed[1]) == 0);
    CHECK(static_cast<u8>(framed[2]) == 0);
    CHECK(static_cast<u8>(framed[3]) == 0);
    CHECK(framed.substr(4) == "hello");
}

TEST_CASE("the default pipe name is per-user", "[protocol]") {
    const std::string name = defaultPipeName();
    CHECK(name.starts_with(R"(\\.\pipe\turbofieldfare-decode)"));
    // Two sessions on one machine must not fight over one model.
    CHECK(name.size() > std::string_view{R"(\\.\pipe\turbofieldfare-decode)"}.size());
}

// ---------------------------------------------------------------------------
// Pipe transport
// ---------------------------------------------------------------------------

TEST_CASE("messages cross a real pipe intact", "[pipe]") {
    const std::string name = testPipeName();

    auto server = PipeServer::create(name);
    REQUIRE(server.has_value());

    std::thread client{[&] {
        auto channel = PipeClient::connect(name);
        REQUIRE(channel.has_value());
        REQUIRE(channel->write(makeHello()).has_value());

        auto reply = channel->read();
        REQUIRE(reply.has_value());
        CHECK(reply->kind == MessageKind::Ready);
        CHECK(reply->ready.model == "gemma4");
    }};

    auto accepted = server->accept();
    REQUIRE(accepted.has_value());

    auto hello = accepted->read();
    REQUIRE(hello.has_value());
    CHECK(hello->kind == MessageKind::Hello);
    REQUIRE(accepted->write(makeReady(ReadyInfo{.model = "gemma4"})).has_value());

    client.join();
}

TEST_CASE("a message larger than the pipe buffer is reassembled", "[pipe]") {
    // The pipe buffer is 64 KiB; a long prompt is bigger, and a reader that
    // assumed one read per message would truncate it.
    const std::string name = testPipeName();
    auto server = PipeServer::create(name);
    REQUIRE(server.has_value());

    const std::string big(400 * 1024, 'x');

    std::thread client{[&] {
        auto channel = PipeClient::connect(name);
        REQUIRE(channel.has_value());

        Message message{.kind = MessageKind::Generate};
        message.generate.id = 1;
        message.generate.chat = false;
        message.generate.prompt = big;
        REQUIRE(channel->write(message).has_value());
    }};

    auto accepted = server->accept();
    REQUIRE(accepted.has_value());
    auto received = accepted->read();
    REQUIRE(received.has_value());
    CHECK(received->generate.prompt.size() == big.size());
    CHECK(received->generate.prompt == big);

    client.join();
}

TEST_CASE("several messages in flight arrive in order", "[pipe]") {
    const std::string name = testPipeName();
    auto server = PipeServer::create(name);
    REQUIRE(server.has_value());

    constexpr int kCount = 200;

    std::thread client{[&] {
        auto channel = PipeClient::connect(name);
        REQUIRE(channel.has_value());
        for (int i = 0; i < kCount; ++i) {
            REQUIRE(channel->write(makeToken(static_cast<u64>(i), std::to_string(i)))
                            .has_value());
        }
    }};

    auto accepted = server->accept();
    REQUIRE(accepted.has_value());
    for (int i = 0; i < kCount; ++i) {
        auto received = accepted->read();
        REQUIRE(received.has_value());
        // Streaming tokens arriving out of order would scramble the answer.
        CHECK(received->id == static_cast<u64>(i));
        CHECK(received->text == std::to_string(i));
    }

    client.join();
}

TEST_CASE("a peer closing reads as cancelled rather than as a fault", "[pipe]") {
    const std::string name = testPipeName();
    auto server = PipeServer::create(name);
    REQUIRE(server.has_value());

    std::thread client{[&] {
        auto channel = PipeClient::connect(name);
        REQUIRE(channel.has_value());
        REQUIRE(channel->write(makeHello()).has_value());
        channel->close();
    }};

    auto accepted = server->accept();
    REQUIRE(accepted.has_value());
    REQUIRE(accepted->read().has_value());

    // A front end exiting is ordinary. The service distinguishes this from a
    // real failure to decide whether to log it.
    auto ended = accepted->read();
    REQUIRE_FALSE(ended.has_value());
    CHECK(ended.error().code() == ErrorCode::Cancelled);

    client.join();
}

TEST_CASE("a second service cannot claim the same pipe", "[pipe]") {
    const std::string name = testPipeName();
    auto first = PipeServer::create(name);
    REQUIRE(first.has_value());

    // FILE_FLAG_FIRST_PIPE_INSTANCE: without it a second launch would create
    // another instance and some clients would silently reach the wrong one.
    auto second = PipeServer::create(name);
    REQUIRE_FALSE(second.has_value());
    CHECK_THAT(second.error().message(),
               Catch::Matchers::ContainsSubstring("already listening"));
}

TEST_CASE("connecting to nothing times out rather than hanging", "[pipe]") {
    auto channel = PipeClient::connect(testPipeName(), /*timeoutMillis=*/200);
    REQUIRE_FALSE(channel.has_value());
    CHECK(channel.error().code() == ErrorCode::Cancelled);
}

TEST_CASE("isRunning reports whether a service is listening", "[pipe]") {
    const std::string name = testPipeName();
    CHECK_FALSE(PipeClient::isRunning(name));

    auto server = PipeServer::create(name);
    REQUIRE(server.has_value());
    // This is how a front end decides whether to start a service.
    CHECK(PipeClient::isRunning(name));
}

TEST_CASE("stop wakes a blocked accept", "[pipe]") {
    const std::string name = testPipeName();
    auto server = PipeServer::create(name);
    REQUIRE(server.has_value());

    std::thread stopper{[&] {
        std::this_thread::sleep_for(std::chrono::milliseconds{100});
        server->stop();
    }};

    // Without an interruptible accept, Ctrl-C could only be honoured by
    // connecting to ourselves.
    auto accepted = server->accept();
    REQUIRE_FALSE(accepted.has_value());
    CHECK(accepted.error().code() == ErrorCode::Cancelled);

    stopper.join();
}
