#include "Api.h"

#include <atomic>
#include <chrono>
#include <format>

#include "tf/core/json/Json.h"

namespace tf::server {
namespace {

[[nodiscard]] i64 unixTime() {
    return std::chrono::duration_cast<std::chrono::seconds>(
                   std::chrono::system_clock::now().time_since_epoch())
            .count();
}

[[nodiscard]] Result<ChatRole> parseRole(std::string_view role) {
    if (role == "user") {
        return ChatRole::User;
    }
    if (role == "assistant") {
        // Gemma calls this role "model"; the OpenAI name for it is "assistant".
        return ChatRole::Model;
    }
    if (role == "system" || role == "developer") {
        // "developer" is the newer name for the same thing; treating them
        // differently would surprise a client that switched.
        return ChatRole::System;
    }
    return makeError(ErrorCode::InvalidArgument, "unknown message role '{}'", role);
}

/// Content may be a string or an array of typed parts. The array form is how
/// multimodal requests arrive; the text parts are joined and anything else is
/// refused rather than dropped, since silently ignoring an image would answer a
/// question the user did not ask.
[[nodiscard]] Result<std::string> parseContent(const json::Value& value) {
    if (const auto text = value.asString()) {
        return std::string{*text};
    }

    const auto parts = value.asArray();
    if (!parts) {
        return makeError(ErrorCode::InvalidArgument,
                         "message content must be a string or an array of parts");
    }

    std::string joined;
    for (const json::Value& part : **parts) {
        const json::Value* type = part.find("type");
        const auto kind = type != nullptr ? type->asString() : Result<std::string_view>{""};
        if (kind && *kind != "text") {
            return makeError(ErrorCode::Unsupported,
                             "content parts of type '{}' are not supported; this build is "
                             "text-only",
                             *kind);
        }
        const json::Value* text = part.find("text");
        if (text != nullptr) {
            if (const auto value2 = text->asString()) {
                joined += *value2;
            }
        }
    }
    return joined;
}

/// `stop` is a string or an array of up to four strings.
[[nodiscard]] Result<std::vector<std::string>> parseStop(const json::Value& value) {
    std::vector<std::string> stops;
    if (value.isNull()) {
        return stops;
    }
    if (const auto single = value.asString()) {
        stops.emplace_back(*single);
        return stops;
    }
    TF_TRY(const json::Array* array, value.asArray());
    for (const json::Value& entry : *array) {
        TF_TRY(const std::string_view text, entry.asString());
        stops.emplace_back(text);
    }
    return stops;
}

}  // namespace

Result<CompletionRequest> CompletionRequest::parse(std::string_view json, bool chat) {
    TF_TRY(const json::Value root, json::parse(json));
    if (!root.isObject()) {
        return makeError(ErrorCode::InvalidArgument, "the request body must be a JSON object");
    }

    CompletionRequest request;
    request.chat = chat;

    if (const json::Value* model = root.find("model")) {
        if (const auto name = model->asString()) {
            request.model = std::string{*name};
        }
    }

    if (chat) {
        TF_TRY(const json::Value* messages, root.at("messages"));
        TF_TRY(const json::Array* array, messages->asArray());
        if (array->empty()) {
            return makeError(ErrorCode::InvalidArgument, "messages must not be empty");
        }
        for (const json::Value& entry : *array) {
            TF_TRY(const json::Value* roleValue, entry.at("role"));
            TF_TRY(const std::string_view roleText, roleValue->asString());
            TF_TRY(const ChatRole role, parseRole(roleText));

            TF_TRY(const json::Value* contentValue, entry.at("content"));
            TF_TRY(std::string content, parseContent(*contentValue));

            request.messages.push_back(ChatMessage{role, std::move(content)});
        }
    } else {
        TF_TRY(const json::Value* prompt, root.at("prompt"));
        if (const auto text = prompt->asString()) {
            request.prompt = std::string{*text};
        } else {
            // The array form means a batch of prompts, which this server does
            // not run.
            return makeError(ErrorCode::Unsupported,
                             "prompt must be a string; batched prompts are not supported");
        }
    }

    if (const json::Value* value = root.find("temperature")) {
        TF_TRY(const double temperature, value->asDouble());
        request.sampling.temperature = static_cast<float>(temperature);
    }
    if (const json::Value* value = root.find("top_p")) {
        TF_TRY(const double topP, value->asDouble());
        request.sampling.topP = static_cast<float>(topP);
    }
    if (const json::Value* value = root.find("top_k")) {
        // Not in the OpenAI schema, but every local server accepts it and
        // clients that know they are talking to one send it.
        TF_TRY(const u64 topK, value->asUInt());
        request.sampling.topK = static_cast<u32>(topK);
    }
    if (const json::Value* value = root.find("seed")) {
        TF_TRY(request.sampling.seed, value->asUInt());
    }
    if (const json::Value* value = root.find("frequency_penalty")) {
        // OpenAI's scale is -2..2 added to the logit; this engine divides by a
        // factor around 1. Mapping 0 to 1.0 keeps the common "no penalty" case
        // exact and moves in the right direction elsewhere.
        TF_TRY(const double penalty, value->asDouble());
        request.sampling.repetitionPenalty = 1.0f + static_cast<float>(penalty) * 0.5f;
        if (request.sampling.repetitionPenalty <= 0.0f) {
            request.sampling.repetitionPenalty = 0.01f;
        }
    }
    if (const json::Value* value = root.find("max_tokens")) {
        TF_TRY(request.maxTokens, value->asUInt());
    }
    // The newer name for the same field. Checked second so it wins when both
    // are sent.
    if (const json::Value* value = root.find("max_completion_tokens")) {
        TF_TRY(request.maxTokens, value->asUInt());
    }
    if (const json::Value* value = root.find("stream")) {
        TF_TRY(request.stream, value->asBool());
    }
    if (const json::Value* value = root.find("stop")) {
        TF_TRY(request.stopStrings, parseStop(*value));
    }
    if (const json::Value* value = root.find("n")) {
        TF_TRY(const u64 count, value->asUInt());
        request.choices = static_cast<u32>(count);
    }

    if (request.choices != 1) {
        return makeError(ErrorCode::Unsupported,
                         "n must be 1; this server generates one completion per request");
    }
    if (request.maxTokens == 0) {
        return makeError(ErrorCode::InvalidArgument, "max_tokens must be at least 1");
    }
    TF_CHECK(request.sampling.validate());
    return request;
}

std::string_view finishReason(runtime::StopReason reason) {
    switch (reason) {
        case runtime::StopReason::StopToken:
        case runtime::StopReason::StopString:
        case runtime::StopReason::Cancelled:
            return "stop";
        case runtime::StopReason::Length:
        case runtime::StopReason::ContextFull:
            // Both mean the answer was cut short, which is what clients act on.
            return "length";
    }
    return "stop";
}

std::string makeCompletionId(bool chat) {
    static std::atomic<u64> counter{0};
    const u64 index = counter.fetch_add(1, std::memory_order_relaxed);
    return std::format("{}-{:016x}{:04x}", chat ? "chatcmpl" : "cmpl",
                       static_cast<u64>(unixTime()), index & 0xFFFFu);
}

std::string completionBody(std::string_view id, std::string_view model,
                           std::string_view content, runtime::StopReason reason,
                           const Usage& usage, bool chat) {
    json::Value choice = json::Value::makeObject();
    choice.set("index", 0);
    if (chat) {
        json::Value message = json::Value::makeObject();
        message.set("role", "assistant");
        message.set("content", std::string{content});
        choice.set("message", std::move(message));
    } else {
        choice.set("text", std::string{content});
    }
    choice.set("logprobs", json::Value{});
    choice.set("finish_reason", std::string{finishReason(reason)});

    json::Value choices = json::Value::makeArray();
    choices.push(std::move(choice));

    json::Value usageValue = json::Value::makeObject();
    usageValue.set("prompt_tokens", usage.promptTokens);
    usageValue.set("completion_tokens", usage.completionTokens);
    usageValue.set("total_tokens", usage.promptTokens + usage.completionTokens);

    json::Value root = json::Value::makeObject();
    root.set("id", std::string{id});
    root.set("object", chat ? "chat.completion" : "text_completion");
    root.set("created", unixTime());
    root.set("model", std::string{model});
    root.set("choices", std::move(choices));
    root.set("usage", std::move(usageValue));
    return root.dump();
}

std::string chunkBody(std::string_view id, std::string_view model, std::string_view content,
                      bool role, const runtime::StopReason* reason, bool chat) {
    json::Value choice = json::Value::makeObject();
    choice.set("index", 0);

    if (chat) {
        json::Value delta = json::Value::makeObject();
        if (role) {
            delta.set("role", "assistant");
        }
        if (!content.empty()) {
            delta.set("content", std::string{content});
        }
        choice.set("delta", std::move(delta));
    } else {
        choice.set("text", std::string{content});
    }

    if (reason != nullptr) {
        choice.set("finish_reason", std::string{finishReason(*reason)});
    } else {
        choice.set("finish_reason", json::Value{});
    }

    json::Value choices = json::Value::makeArray();
    choices.push(std::move(choice));

    json::Value root = json::Value::makeObject();
    root.set("id", std::string{id});
    root.set("object", chat ? "chat.completion.chunk" : "text_completion");
    root.set("created", unixTime());
    root.set("model", std::string{model});
    root.set("choices", std::move(choices));
    return root.dump();
}

std::string modelsBody(std::string_view model) {
    json::Value entry = json::Value::makeObject();
    entry.set("id", std::string{model});
    entry.set("object", "model");
    entry.set("created", unixTime());
    entry.set("owned_by", "local");

    json::Value data = json::Value::makeArray();
    data.push(std::move(entry));

    json::Value root = json::Value::makeObject();
    root.set("object", "list");
    root.set("data", std::move(data));
    return root.dump();
}

}  // namespace tf::server
