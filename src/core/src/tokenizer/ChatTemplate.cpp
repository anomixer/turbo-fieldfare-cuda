#include "tf/core/tokenizer/ChatTemplate.h"

#include <algorithm>

namespace tf {
namespace {

constexpr std::string_view kTurnOpen = "<|turn>";
constexpr std::string_view kTurnClose = "<turn|>";
constexpr std::string_view kBeginOfSequence = "<bos>";
constexpr std::string_view kNewline = "\n";

[[nodiscard]] std::string_view roleName(ChatRole role) {
    switch (role) {
        case ChatRole::Model:
            return "model";
        case ChatRole::System:
        case ChatRole::User:
            return "user";
    }
    return "user";
}

}  // namespace

Result<ChatTemplate> ChatTemplate::create(const Tokenizer& tokenizer) {
    ChatTemplate rendered;
    rendered.tokenizer_ = &tokenizer;

    // Resolved from the installed vocabulary rather than hardcoded, so a
    // checkpoint that renumbers them still works.
    TF_TRY(rendered.bos_, tokenizer.idFor(kBeginOfSequence));
    TF_TRY(rendered.turnOpen_, tokenizer.idFor(kTurnOpen));
    TF_TRY(rendered.turnClose_, tokenizer.idFor(kTurnClose));
    TF_TRY(rendered.newline_, tokenizer.idFor(kNewline));

    rendered.stopTokens_.push_back(rendered.turnClose_);
    if (const auto eos = tokenizer.idFor("<eos>"); eos.has_value()) {
        rendered.stopTokens_.push_back(*eos);
    }
    return rendered;
}

Result<std::vector<u32>> ChatTemplate::render(const std::vector<ChatMessage>& messages,
                                              bool addGenerationPrompt) const {
    if (tokenizer_ == nullptr) {
        return makeError(ErrorCode::InvalidArgument, "chat template is not initialized");
    }

    std::vector<u32> ids{bos_};

    // Gemma has no system turn, so system guidance is folded into the first
    // user message with a blank line between, as the bundled template does.
    std::string pendingSystem;

    for (const auto& message : messages) {
        if (message.role == ChatRole::System) {
            if (!pendingSystem.empty()) {
                pendingSystem += "\n\n";
            }
            pendingSystem += message.content;
            continue;
        }

        std::string content = message.content;
        if (message.role == ChatRole::User && !pendingSystem.empty()) {
            content = pendingSystem + "\n\n" + content;
            pendingSystem.clear();
        }

        ids.push_back(turnOpen_);
        for (const u32 id : tokenizer_->encode(roleName(message.role))) {
            ids.push_back(id);
        }
        ids.push_back(newline_);
        for (const u32 id : tokenizer_->encode(content)) {
            ids.push_back(id);
        }
        ids.push_back(turnClose_);
        ids.push_back(newline_);
    }

    // A conversation that was only system messages still needs somewhere to put
    // them, so emit a user turn carrying just that text.
    if (!pendingSystem.empty()) {
        ids.push_back(turnOpen_);
        for (const u32 id : tokenizer_->encode("user")) {
            ids.push_back(id);
        }
        ids.push_back(newline_);
        for (const u32 id : tokenizer_->encode(pendingSystem)) {
            ids.push_back(id);
        }
        ids.push_back(turnClose_);
        ids.push_back(newline_);
    }

    if (addGenerationPrompt) {
        ids.push_back(turnOpen_);
        for (const u32 id : tokenizer_->encode("model")) {
            ids.push_back(id);
        }
        ids.push_back(newline_);
    }

    return ids;
}

}  // namespace tf
