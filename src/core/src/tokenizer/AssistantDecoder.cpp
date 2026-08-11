#include "tf/core/tokenizer/AssistantDecoder.h"

namespace tf {

AssistantDecoder::AssistantDecoder(const Tokenizer& tokenizer) : tokenizer_(&tokenizer) {
    const auto open = tokenizer.idFor("<|channel>");
    const auto close = tokenizer.idFor("<channel|>");
    if (open && close) {
        channelOpen_ = *open;
        channelClose_ = *close;
        structured_ = true;
    }
}

AssistantDecoder::Update AssistantDecoder::push(u32 token) {
    if (!structured_) {
        // No markers in this checkpoint: everything is the answer.
        return Update{.content = tokenizer_->decodeOne(token, /*skipSpecialTokens=*/true)};
    }

    if (token == channelOpen_) {
        inChannel_ = true;
        return {};
    }
    if (token == channelClose_) {
        inChannel_ = false;
        return {};
    }

    // Special tokens other than the markers - an end-of-turn that slipped
    // through, say - are not text the user asked for.
    std::string piece = tokenizer_->decodeOne(token, /*skipSpecialTokens=*/true);
    if (piece.empty()) {
        return {};
    }
    if (inChannel_) {
        return Update{.thinking = std::move(piece)};
    }
    return Update{.content = std::move(piece)};
}

}  // namespace tf
