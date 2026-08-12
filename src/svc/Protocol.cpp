#include "Protocol.h"

#include <windows.h>

#include <array>
#include <format>

#include "tf/core/json/Json.h"

namespace tf::svc {
namespace {

constexpr std::array<std::pair<MessageKind, std::string_view>, 12> kNames{{
        {MessageKind::Hello, "hello"},
        {MessageKind::Generate, "generate"},
        {MessageKind::Cancel, "cancel"},
        {MessageKind::Reset, "reset"},
        {MessageKind::Status, "status"},
        {MessageKind::Shutdown, "shutdown"},
        {MessageKind::Ready, "ready"},
        {MessageKind::Token, "token"},
        {MessageKind::Thinking, "thinking"},
        {MessageKind::Done, "done"},
        {MessageKind::Error, "error"},
        {MessageKind::StatusReport, "status_report"},
}};

[[nodiscard]] Result<ChatRole> parseRole(std::string_view role) {
    if (role == "user") {
        return ChatRole::User;
    }
    if (role == "assistant" || role == "model") {
        return ChatRole::Model;
    }
    if (role == "system") {
        return ChatRole::System;
    }
    return makeError(ErrorCode::InvalidArgument, "unknown role '{}'", role);
}

[[nodiscard]] std::string_view roleName(ChatRole role) {
    switch (role) {
        case ChatRole::User: return "user";
        case ChatRole::Model: return "assistant";
        case ChatRole::System: return "system";
    }
    return "user";
}

/// Reads an optional field, leaving the target alone when absent. Every
/// generate field is optional so a front end can send only what it cares about
/// and inherit the defaults for the rest.
template <class T>
void readOptional(const json::Value& root, std::string_view key, T& target) {
    const json::Value* value = root.find(key);
    if (value == nullptr) {
        return;
    }
    if constexpr (std::is_same_v<T, bool>) {
        if (const auto parsed = value->asBool()) {
            target = *parsed;
        }
    } else if constexpr (std::is_same_v<T, float>) {
        if (const auto parsed = value->asDouble()) {
            target = static_cast<float>(*parsed);
        }
    } else if constexpr (std::is_same_v<T, double>) {
        if (const auto parsed = value->asDouble()) {
            target = *parsed;
        }
    } else if constexpr (std::is_same_v<T, std::string>) {
        if (const auto parsed = value->asString()) {
            target = std::string{*parsed};
        }
    } else {
        if (const auto parsed = value->asUInt()) {
            target = static_cast<T>(*parsed);
        }
    }
}

}  // namespace

std::string defaultPipeName() {
    // The session's own pipe. Two users on one machine each get a service
    // rather than fighting over one model, and a stale pipe from another
    // account cannot be connected to by mistake.
    wchar_t user[256] = {};
    DWORD size = static_cast<DWORD>(std::size(user));
    if (::GetUserNameW(user, &size) == 0 || size <= 1) {
        return R"(\\.\pipe\turbofieldfare-decode)";
    }

    std::string narrowed;
    narrowed.reserve(size);
    for (DWORD i = 0; i + 1 < size; ++i) {
        const wchar_t c = user[i];
        // Only characters that are safe in a pipe name; anything else becomes
        // an underscore rather than being dropped, so two users cannot collide.
        narrowed += (c < 128 && (std::isalnum(static_cast<int>(c)) != 0))
                            ? static_cast<char>(c)
                            : '_';
    }
    return std::format(R"(\\.\pipe\turbofieldfare-decode-{})", narrowed);
}

std::string_view toString(MessageKind kind) noexcept {
    for (const auto& [value, name] : kNames) {
        if (value == kind) {
            return name;
        }
    }
    return "?";
}

Result<MessageKind> parseKind(std::string_view text) {
    for (const auto& [value, name] : kNames) {
        if (name == text) {
            return value;
        }
    }
    return makeError(ErrorCode::InvalidArgument, "unknown message type '{}'", text);
}

std::string Message::encode() const {
    json::Value root = json::Value::makeObject();
    root.set("type", std::string{toString(kind)});
    root.set("version", version);

    switch (kind) {
        case MessageKind::Generate: {
            root.set("id", generate.id);
            root.set("chat", generate.chat);
            if (generate.chat) {
                json::Value messages = json::Value::makeArray();
                for (const ChatMessage& message : generate.messages) {
                    json::Value entry = json::Value::makeObject();
                    entry.set("role", std::string{roleName(message.role)});
                    entry.set("content", message.content);
                    messages.push(std::move(entry));
                }
                root.set("messages", std::move(messages));
            } else {
                root.set("prompt", generate.prompt);
            }
            root.set("temperature", static_cast<double>(generate.sampling.temperature));
            root.set("top_k", generate.sampling.topK);
            root.set("top_p", static_cast<double>(generate.sampling.topP));
            root.set("repeat_penalty",
                     static_cast<double>(generate.sampling.repetitionPenalty));
            root.set("repeat_window", generate.sampling.repetitionWindow);
            root.set("seed", generate.sampling.seed);
            root.set("max_tokens", generate.maxTokens);
            root.set("include_thinking", generate.includeThinking);

            json::Value stops = json::Value::makeArray();
            for (const std::string& stop : generate.stopStrings) {
                stops.push(stop);
            }
            root.set("stop", std::move(stops));
            break;
        }
        case MessageKind::Cancel:
            root.set("id", id);
            break;
        case MessageKind::Ready:
            root.set("model", ready.model);
            root.set("context_length", ready.contextLength);
            root.set("device", ready.device);
            root.set("fully_resident", ready.fullyResident);
            break;
        case MessageKind::Token:
        case MessageKind::Thinking:
            root.set("id", id);
            root.set("text", text);
            break;
        case MessageKind::Error:
            root.set("id", id);
            root.set("message", text);
            break;
        case MessageKind::Done:
            root.set("id", done.id);
            root.set("reason", done.reason);
            root.set("prompt_tokens", done.promptTokens);
            root.set("cached_prompt_tokens", done.cachedPromptTokens);
            root.set("generated_tokens", done.generatedTokens);
            root.set("prefill_seconds", done.prefillSeconds);
            root.set("decode_seconds", done.decodeSeconds);
            break;
        case MessageKind::StatusReport:
            root.set("model", status.model);
            root.set("context_length", status.contextLength);
            root.set("position", status.position);
            root.set("requests", status.requests);
            root.set("prompt_tokens", status.promptTokens);
            root.set("cached_prompt_tokens", status.cachedPromptTokens);
            root.set("generated_tokens", status.generatedTokens);
            root.set("expert_hit_rate", status.expertHitRate);
            root.set("expert_bytes_read", status.expertBytesRead);
            root.set("busy", status.busy);
            break;
        case MessageKind::Hello:
        case MessageKind::Reset:
        case MessageKind::Status:
        case MessageKind::Shutdown:
            break;
    }
    return root.dump();
}

Result<Message> Message::decode(std::string_view json) {
    TF_TRY(const json::Value root, json::parse(json));
    if (!root.isObject()) {
        return makeError(ErrorCode::InvalidArgument, "a message must be a JSON object");
    }

    Message message;
    TF_TRY(const json::Value* type, root.at("type"));
    TF_TRY(const std::string_view typeName, type->asString());
    TF_TRY(message.kind, parseKind(typeName));
    readOptional(root, "version", message.version);

    switch (message.kind) {
        case MessageKind::Generate: {
            readOptional(root, "id", message.generate.id);
            readOptional(root, "chat", message.generate.chat);
            readOptional(root, "prompt", message.generate.prompt);

            if (const json::Value* messages = root.find("messages")) {
                TF_TRY(const json::Array* array, messages->asArray());
                for (const json::Value& entry : *array) {
                    TF_TRY(const json::Value* roleValue, entry.at("role"));
                    TF_TRY(const std::string_view roleText, roleValue->asString());
                    TF_TRY(const ChatRole role, parseRole(roleText));

                    TF_TRY(const json::Value* contentValue, entry.at("content"));
                    TF_TRY(const std::string_view content, contentValue->asString());

                    message.generate.messages.push_back(
                            ChatMessage{role, std::string{content}});
                }
            }

            readOptional(root, "temperature", message.generate.sampling.temperature);
            readOptional(root, "top_k", message.generate.sampling.topK);
            readOptional(root, "top_p", message.generate.sampling.topP);
            readOptional(root, "repeat_penalty",
                         message.generate.sampling.repetitionPenalty);
            readOptional(root, "repeat_window", message.generate.sampling.repetitionWindow);
            readOptional(root, "seed", message.generate.sampling.seed);
            readOptional(root, "max_tokens", message.generate.maxTokens);
            readOptional(root, "include_thinking", message.generate.includeThinking);

            if (const json::Value* stops = root.find("stop")) {
                TF_TRY(const json::Array* array, stops->asArray());
                for (const json::Value& entry : *array) {
                    TF_TRY(const std::string_view stop, entry.asString());
                    message.generate.stopStrings.emplace_back(stop);
                }
            }

            if (message.generate.chat && message.generate.messages.empty()) {
                return makeError(ErrorCode::InvalidArgument,
                                 "a chat generate request needs at least one message");
            }
            if (!message.generate.chat && message.generate.prompt.empty()) {
                return makeError(ErrorCode::InvalidArgument,
                                 "a raw generate request needs a prompt");
            }
            TF_CHECK(message.generate.sampling.validate());
            break;
        }
        case MessageKind::Cancel:
            readOptional(root, "id", message.id);
            break;
        case MessageKind::Ready:
            readOptional(root, "model", message.ready.model);
            readOptional(root, "context_length", message.ready.contextLength);
            readOptional(root, "device", message.ready.device);
            readOptional(root, "fully_resident", message.ready.fullyResident);
            message.ready.version = message.version;
            break;
        case MessageKind::Token:
        case MessageKind::Thinking:
            readOptional(root, "id", message.id);
            readOptional(root, "text", message.text);
            break;
        case MessageKind::Error:
            readOptional(root, "id", message.id);
            readOptional(root, "message", message.text);
            break;
        case MessageKind::Done:
            readOptional(root, "id", message.done.id);
            readOptional(root, "reason", message.done.reason);
            readOptional(root, "prompt_tokens", message.done.promptTokens);
            readOptional(root, "cached_prompt_tokens", message.done.cachedPromptTokens);
            readOptional(root, "generated_tokens", message.done.generatedTokens);
            readOptional(root, "prefill_seconds", message.done.prefillSeconds);
            readOptional(root, "decode_seconds", message.done.decodeSeconds);
            break;
        case MessageKind::StatusReport:
            readOptional(root, "model", message.status.model);
            readOptional(root, "context_length", message.status.contextLength);
            readOptional(root, "position", message.status.position);
            readOptional(root, "requests", message.status.requests);
            readOptional(root, "prompt_tokens", message.status.promptTokens);
            readOptional(root, "cached_prompt_tokens", message.status.cachedPromptTokens);
            readOptional(root, "generated_tokens", message.status.generatedTokens);
            readOptional(root, "expert_hit_rate", message.status.expertHitRate);
            readOptional(root, "expert_bytes_read", message.status.expertBytesRead);
            readOptional(root, "busy", message.status.busy);
            break;
        case MessageKind::Hello:
        case MessageKind::Reset:
        case MessageKind::Status:
        case MessageKind::Shutdown:
            break;
    }
    return message;
}

Message makeHello() { return Message{.kind = MessageKind::Hello}; }

Message makeReady(ReadyInfo info) {
    Message message{.kind = MessageKind::Ready};
    message.ready = std::move(info);
    return message;
}

Message makeToken(u64 id, std::string text) {
    Message message{.kind = MessageKind::Token};
    message.id = id;
    message.text = std::move(text);
    return message;
}

Message makeThinking(u64 id, std::string text) {
    Message message{.kind = MessageKind::Thinking};
    message.id = id;
    message.text = std::move(text);
    return message;
}

Message makeDone(DoneInfo info) {
    Message message{.kind = MessageKind::Done};
    message.done = std::move(info);
    return message;
}

Message makeError(u64 id, std::string text) {
    Message message{.kind = MessageKind::Error};
    message.id = id;
    message.text = std::move(text);
    return message;
}

Message makeStatusReport(StatusInfo info) {
    Message message{.kind = MessageKind::StatusReport};
    message.status = std::move(info);
    return message;
}

std::string frame(std::string_view payload) {
    const auto length = static_cast<u32>(payload.size());
    std::string out;
    out.reserve(payload.size() + 4);
    // Little-endian explicitly, so the framing does not depend on the host.
    out += static_cast<char>(length & 0xFF);
    out += static_cast<char>((length >> 8) & 0xFF);
    out += static_cast<char>((length >> 16) & 0xFF);
    out += static_cast<char>((length >> 24) & 0xFF);
    out += payload;
    return out;
}

}  // namespace tf::svc
