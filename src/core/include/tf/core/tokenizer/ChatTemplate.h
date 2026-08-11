#pragma once

#include <string>
#include <vector>

#include "tf/core/base/Error.h"
#include "tf/core/base/Types.h"
#include "tf/core/tokenizer/Tokenizer.h"

namespace tf {

enum class ChatRole {
    /// Gemma has no separate system role. System guidance is prepended to the
    /// first user turn, which is what the bundled template does.
    System,
    User,
    Model,
};

struct ChatMessage {
    ChatRole role = ChatRole::User;
    std::string content;
};

/// Renders Gemma's instruction format.
///
/// This is not optional: the checkpoint is instruction tuned, and raw
/// completion text is out of distribution for it. The layout is
///
///   <bos><|turn>user\n{content}<turn|>\n<|turn>model\n
///
/// and generation ends at <turn|>. Only the text-only, no-tool subset of the
/// bundled chat_template.jinja is implemented, matching what the runtime
/// supports.
class ChatTemplate {
public:
    [[nodiscard]] static Result<ChatTemplate> create(const Tokenizer& tokenizer);

    /// Token ids for a conversation, ending with the opening of a model turn so
    /// generation continues from there.
    [[nodiscard]] Result<std::vector<u32>> render(const std::vector<ChatMessage>& messages,
                                                  bool addGenerationPrompt = true) const;

    /// Ids that end a model turn. Generation should stop on any of them.
    [[nodiscard]] const std::vector<u32>& stopTokens() const noexcept { return stopTokens_; }

    [[nodiscard]] u32 beginOfSequence() const noexcept { return bos_; }

private:
    const Tokenizer* tokenizer_ = nullptr;
    u32 bos_ = 0;
    u32 turnOpen_ = 0;
    u32 turnClose_ = 0;
    u32 newline_ = 0;
    std::vector<u32> stopTokens_;
};

}  // namespace tf
