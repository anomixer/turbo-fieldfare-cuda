// The OpenAI wire format.
//
// This is the compatibility surface: a field spelled wrong here does not fail,
// it makes a client silently ignore a setting or render an empty message. So
// the shapes are pinned against what the SDKs actually read, and parsing is
// checked against what they actually send - including the variants, since a
// client that sends `stop` as a string and one that sends an array are both
// correct.
//
// No socket and no GPU: Api.cpp is deliberately free of both.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <algorithm>
#include <string>

#include "Api.h"
#include "Http.h"
#include "tf/core/json/Json.h"

using namespace tf;
using namespace tf::server;

namespace {

[[nodiscard]] json::Value parsed(std::string_view text) {
    auto value = json::parse(text);
    REQUIRE(value.has_value());
    return std::move(*value);
}

/// Resolves a dotted path, treating an all-digit segment as an array index.
///
/// json::Value::path walks object keys only, and every interesting field here
/// sits inside `choices[0]`. Returns nullptr rather than failing so the absence
/// of a field can itself be asserted.
[[nodiscard]] const json::Value* at(const json::Value& root, std::string_view path) {
    const json::Value* current = &root;
    usize start = 0;
    while (start <= path.size() && current != nullptr) {
        const usize dot = path.find('.', start);
        const std::string_view segment =
                path.substr(start, dot == std::string_view::npos ? dot : dot - start);

        const bool numeric = !segment.empty() &&
                             std::ranges::all_of(segment, [](char c) {
                                 return c >= '0' && c <= '9';
                             });
        if (numeric) {
            const auto array = current->asArray();
            if (!array) {
                return nullptr;
            }
            const usize index = static_cast<usize>(std::stoul(std::string{segment}));
            if (index >= (*array)->size()) {
                return nullptr;
            }
            current = &(**array)[index];
        } else {
            current = current->find(segment);
        }

        if (dot == std::string_view::npos) {
            break;
        }
        start = dot + 1;
    }
    return current;
}

/// Reads a path and requires it to be a string.
[[nodiscard]] std::string stringAt(const json::Value& root, std::string_view path) {
    const json::Value* found = at(root, path);
    REQUIRE(found != nullptr);
    auto text = found->asString();
    REQUIRE(text.has_value());
    return std::string{*text};
}

}  // namespace

// ---------------------------------------------------------------------------
// Request parsing
// ---------------------------------------------------------------------------

TEST_CASE("a minimal chat request parses", "[api]") {
    auto request = CompletionRequest::parse(
            R"({"model":"gemma","messages":[{"role":"user","content":"hi"}]})", true);
    REQUIRE(request.has_value());
    CHECK(request->model == "gemma");
    REQUIRE(request->messages.size() == 1);
    CHECK(request->messages[0].role == ChatRole::User);
    CHECK(request->messages[0].content == "hi");
    CHECK_FALSE(request->stream);
}

TEST_CASE("every role a client may send is mapped", "[api]") {
    auto request = CompletionRequest::parse(
            R"({"messages":[
                {"role":"system","content":"a"},
                {"role":"user","content":"b"},
                {"role":"assistant","content":"c"},
                {"role":"developer","content":"d"}]})",
            true);
    REQUIRE(request.has_value());
    REQUIRE(request->messages.size() == 4);
    CHECK(request->messages[0].role == ChatRole::System);
    CHECK(request->messages[1].role == ChatRole::User);
    // Gemma calls it "model"; OpenAI calls it "assistant".
    CHECK(request->messages[2].role == ChatRole::Model);
    // "developer" is the newer name for a system message.
    CHECK(request->messages[3].role == ChatRole::System);
}

TEST_CASE("an unknown role is refused rather than guessed", "[api]") {
    auto request = CompletionRequest::parse(
            R"({"messages":[{"role":"wizard","content":"hi"}]})", true);
    REQUIRE_FALSE(request.has_value());
    CHECK_THAT(request.error().message(), Catch::Matchers::ContainsSubstring("wizard"));
}

TEST_CASE("content may be an array of text parts", "[api]") {
    auto request = CompletionRequest::parse(
            R"({"messages":[{"role":"user","content":[
                {"type":"text","text":"first "},
                {"type":"text","text":"second"}]}]})",
            true);
    REQUIRE(request.has_value());
    CHECK(request->messages[0].content == "first second");
}

TEST_CASE("a non-text content part is refused, not dropped", "[api]") {
    // Silently ignoring an image would answer a question the user did not ask.
    auto request = CompletionRequest::parse(
            R"({"messages":[{"role":"user","content":[
                {"type":"image_url","image_url":{"url":"http://x/y.png"}}]}]})",
            true);
    REQUIRE_FALSE(request.has_value());
    CHECK(request.error().code() == ErrorCode::Unsupported);
    CHECK_THAT(request.error().message(), Catch::Matchers::ContainsSubstring("text-only"));
}

TEST_CASE("stop accepts a string or an array", "[api]") {
    auto single = CompletionRequest::parse(
            R"({"messages":[{"role":"user","content":"x"}],"stop":"END"})", true);
    REQUIRE(single.has_value());
    REQUIRE(single->stopStrings.size() == 1);
    CHECK(single->stopStrings[0] == "END");

    auto many = CompletionRequest::parse(
            R"({"messages":[{"role":"user","content":"x"}],"stop":["A","B"]})", true);
    REQUIRE(many.has_value());
    REQUIRE(many->stopStrings.size() == 2);
    CHECK(many->stopStrings[1] == "B");

    // Explicit null is how some clients say "no stop sequences".
    auto none = CompletionRequest::parse(
            R"({"messages":[{"role":"user","content":"x"}],"stop":null})", true);
    REQUIRE(none.has_value());
    CHECK(none->stopStrings.empty());
}

TEST_CASE("sampling fields reach the sampler", "[api]") {
    auto request = CompletionRequest::parse(
            R"({"messages":[{"role":"user","content":"x"}],
                "temperature":0.7,"top_p":0.9,"top_k":40,"seed":1234,"max_tokens":99})",
            true);
    REQUIRE(request.has_value());
    CHECK(request->sampling.temperature == 0.7f);
    CHECK(request->sampling.topP == 0.9f);
    CHECK(request->sampling.topK == 40);
    CHECK(request->sampling.seed == 1234);
    CHECK(request->maxTokens == 99);
}

TEST_CASE("max_completion_tokens overrides the older max_tokens", "[api]") {
    // A client migrating to the newer field may send both; the newer one is
    // what it means.
    auto request = CompletionRequest::parse(
            R"({"messages":[{"role":"user","content":"x"}],
                "max_tokens":10,"max_completion_tokens":20})",
            true);
    REQUIRE(request.has_value());
    CHECK(request->maxTokens == 20);
}

TEST_CASE("a zero frequency penalty means no penalty", "[api]") {
    // The scales differ - OpenAI adds to the logit, this engine divides - so
    // the mapping has to be exact at zero or every default request would come
    // out subtly penalized.
    auto request = CompletionRequest::parse(
            R"({"messages":[{"role":"user","content":"x"}],"frequency_penalty":0})", true);
    REQUIRE(request.has_value());
    CHECK(request->sampling.repetitionPenalty == 1.0f);

    auto penalized = CompletionRequest::parse(
            R"({"messages":[{"role":"user","content":"x"}],"frequency_penalty":1.0})", true);
    REQUIRE(penalized.has_value());
    CHECK(penalized->sampling.repetitionPenalty > 1.0f);
}

TEST_CASE("requests this server cannot honour are refused", "[api]") {
    const auto refuses = [](std::string_view body) {
        return !CompletionRequest::parse(body, true).has_value();
    };

    CHECK(refuses(R"({"messages":[]})"));
    CHECK(refuses(R"({"model":"x"})"));
    CHECK(refuses(R"({"messages":[{"role":"user","content":"x"}],"n":2})"));
    CHECK(refuses(R"({"messages":[{"role":"user","content":"x"}],"max_tokens":0})"));
    CHECK(refuses(R"({"messages":[{"role":"user","content":"x"}],"temperature":-1})"));
    CHECK(refuses("not json"));
    CHECK(refuses("[]"));
}

TEST_CASE("the raw completion endpoint takes a prompt", "[api]") {
    auto request = CompletionRequest::parse(R"({"prompt":"once upon a"})", false);
    REQUIRE(request.has_value());
    CHECK(request->prompt == "once upon a");
    CHECK_FALSE(request->chat);

    // A batch of prompts would need several generations; refused rather than
    // answered with only the first.
    auto batch = CompletionRequest::parse(R"({"prompt":["a","b"]})", false);
    REQUIRE_FALSE(batch.has_value());
}

// ---------------------------------------------------------------------------
// Response shapes
// ---------------------------------------------------------------------------

TEST_CASE("a chat completion has the fields clients read", "[api]") {
    const std::string body =
            completionBody("chatcmpl-1", "gemma", "Paris", runtime::StopReason::StopToken,
                           Usage{.promptTokens = 21, .completionTokens = 5}, true);
    const json::Value root = parsed(body);

    CHECK(stringAt(root, "id") == "chatcmpl-1");
    CHECK(stringAt(root, "object") == "chat.completion");
    CHECK(stringAt(root, "model") == "gemma");
    CHECK(stringAt(root, "choices.0.message.role") == "assistant");
    CHECK(stringAt(root, "choices.0.message.content") == "Paris");
    CHECK(stringAt(root, "choices.0.finish_reason") == "stop");

    const json::Value* total = at(root, "usage.total_tokens");
    REQUIRE(total != nullptr);
    CHECK(total->asUInt().value_or(0) == 26);
}

TEST_CASE("a raw completion carries text rather than a message", "[api]") {
    const std::string body =
            completionBody("cmpl-1", "gemma", " yellow", runtime::StopReason::Length,
                           Usage{.promptTokens = 10, .completionTokens = 2}, false);
    const json::Value root = parsed(body);

    CHECK(stringAt(root, "object") == "text_completion");
    CHECK(stringAt(root, "choices.0.text") == " yellow");
    // Hitting the token limit is "length", which is how a client knows the
    // answer was cut off rather than finished.
    CHECK(stringAt(root, "choices.0.finish_reason") == "length");
}

TEST_CASE("the first stream chunk carries the role", "[api]") {
    const std::string body = chunkBody("id", "gemma", "Par", true, nullptr, true);
    const json::Value root = parsed(body);

    CHECK(stringAt(root, "object") == "chat.completion.chunk");
    CHECK(stringAt(root, "choices.0.delta.role") == "assistant");
    CHECK(stringAt(root, "choices.0.delta.content") == "Par");

    const json::Value* reason = at(root, "choices.0.finish_reason");
    REQUIRE(reason != nullptr);
    // Null rather than absent: clients check for a value here to know the
    // stream is still running.
    CHECK(reason->isNull());
}

TEST_CASE("later chunks omit the role", "[api]") {
    const json::Value root = parsed(chunkBody("id", "gemma", "is", false, nullptr, true));
    CHECK(at(root, "choices.0.delta.role") == nullptr);
    CHECK(stringAt(root, "choices.0.delta.content") == "is");
}

TEST_CASE("the final chunk carries a finish reason and no content", "[api]") {
    const auto reason = runtime::StopReason::StopString;
    const json::Value root = parsed(chunkBody("id", "gemma", "", false, &reason, true));

    CHECK(stringAt(root, "choices.0.finish_reason") == "stop");
    CHECK(at(root, "choices.0.delta.content") == nullptr);
}

TEST_CASE("every stop reason maps to one OpenAI value", "[api]") {
    CHECK(finishReason(runtime::StopReason::StopToken) == "stop");
    CHECK(finishReason(runtime::StopReason::StopString) == "stop");
    CHECK(finishReason(runtime::StopReason::Cancelled) == "stop");
    // Both of these mean the answer was truncated, which is what a client acts
    // on - it does not care which limit was hit.
    CHECK(finishReason(runtime::StopReason::Length) == "length");
    CHECK(finishReason(runtime::StopReason::ContextFull) == "length");
}

TEST_CASE("the model list is shaped as a list object", "[api]") {
    const json::Value root = parsed(modelsBody("gemma4"));
    CHECK(stringAt(root, "object") == "list");
    CHECK(stringAt(root, "data.0.id") == "gemma4");
    CHECK(stringAt(root, "data.0.object") == "model");
}

TEST_CASE("completion ids are distinct and prefixed by endpoint", "[api]") {
    const std::string chat = makeCompletionId(true);
    const std::string raw = makeCompletionId(false);
    CHECK(chat.starts_with("chatcmpl-"));
    CHECK(raw.starts_with("cmpl-"));
    CHECK(makeCompletionId(true) != chat);
}

// ---------------------------------------------------------------------------
// Error envelope
// ---------------------------------------------------------------------------

TEST_CASE("errors use the envelope clients render", "[api]") {
    const Response response = Response::error(400, "something went wrong");
    CHECK(response.status == 400);

    const json::Value root = parsed(response.body);
    // A bare string body shows as an empty message in every SDK.
    CHECK(stringAt(root, "error.message") == "something went wrong");
    CHECK(stringAt(root, "error.type") == "invalid_request_error");
}

TEST_CASE("error messages with quotes and newlines stay valid JSON", "[api]") {
    // Error text comes from std::format over model paths and user input, so it
    // will eventually contain both.
    const Response response =
            Response::error(400, "bad \"value\" at\nline 2\tcolumn 3\\end");
    const json::Value root = parsed(response.body);
    CHECK(stringAt(root, "error.message") == "bad \"value\" at\nline 2\tcolumn 3\\end");
}
