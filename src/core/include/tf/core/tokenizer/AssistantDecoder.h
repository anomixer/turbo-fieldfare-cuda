#pragma once

#include <string>
#include <vector>

#include "tf/core/base/Types.h"
#include "tf/core/tokenizer/Tokenizer.h"

namespace tf {

/// Separates the assistant's answer from its thinking.
///
/// Gemma 4 does not reply with plain text. A turn looks like
///
///     <|channel>thought\n  ...thinking...  <channel|>  ...answer...  <turn|>
///
/// mirroring how a turn itself is `<|turn>role\n ... <turn|>`. Even with
/// thinking disabled the model still opens and closes an empty thought channel,
/// so a front end that prints every decoded token shows
/// "<|channel>thought\n<channel|>Paris" where the user asked for "Paris".
///
/// Splitting on token ids rather than on the decoded text matters: the markers
/// are single tokens, so there is nothing to parse and no way for ordinary text
/// that happens to spell "<channel|>" to be mistaken for one.
class AssistantDecoder {
public:
    /// Looks the markers up by name. A checkpoint without them decodes as plain
    /// text, which is the right behaviour rather than an error.
    explicit AssistantDecoder(const Tokenizer& tokenizer);

    struct Update {
        /// Text belonging to the answer.
        std::string content;
        /// Text belonging to the thinking channel, including its name.
        std::string thinking;
    };

    /// Routes one token. Marker tokens produce no text on either side.
    [[nodiscard]] Update push(u32 token);

    /// True while inside a channel, so the caller can tell "still thinking"
    /// from "producing nothing".
    [[nodiscard]] bool inChannel() const noexcept { return inChannel_; }

    /// True when this checkpoint has the channel markers at all.
    [[nodiscard]] bool structured() const noexcept { return structured_; }

    void reset() noexcept { inChannel_ = false; }

private:
    const Tokenizer* tokenizer_ = nullptr;
    u32 channelOpen_ = 0;
    u32 channelClose_ = 0;
    bool structured_ = false;
    bool inChannel_ = false;
};

}  // namespace tf
