#pragma once

#include <string>
#include <vector>

#include "Http.h"
#include "tf/core/base/Error.h"
#include "tf/core/base/Types.h"
#include "tf/core/tokenizer/ChatTemplate.h"
#include "tf/runtime/Generator.h"

/// The OpenAI-compatible request and response shapes.
///
/// Kept separate from the HTTP plumbing and from inference so the wire format
/// can be tested without a socket or a GPU - which matters, because a field
/// spelled wrong here shows up as a client that silently ignores a setting
/// rather than as an error.
namespace tf::server {

/// What a client asked for, in this project's own terms.
struct CompletionRequest {
    /// Echoed back. Clients display it and some route on it.
    std::string model;

    /// Set for /v1/chat/completions. Empty for the raw completion endpoint.
    std::vector<ChatMessage> messages;
    /// Set for /v1/completions.
    std::string prompt;
    bool chat = true;

    runtime::SamplingParams sampling;
    u64 maxTokens = 512;
    std::vector<std::string> stopStrings;
    bool stream = false;

    /// `n` above one would need independent sampling streams over one KV cache.
    /// Refused rather than silently answered with a single choice.
    u32 choices = 1;

    [[nodiscard]] static Result<CompletionRequest> parse(std::string_view json, bool chat);
};

/// What OpenAI calls the reason a completion ended.
[[nodiscard]] std::string_view finishReason(runtime::StopReason reason);

struct Usage {
    u64 promptTokens = 0;
    u64 completionTokens = 0;
};

/// Body of a non-streaming response.
[[nodiscard]] std::string completionBody(std::string_view id, std::string_view model,
                                         std::string_view content, runtime::StopReason reason,
                                         const Usage& usage, bool chat);

/// One `data:` payload of a streaming response.
///
/// `content` empty with a reason set is the final chunk; `role` true marks the
/// first, which every client expects to carry the assistant role and nothing
/// else.
[[nodiscard]] std::string chunkBody(std::string_view id, std::string_view model,
                                    std::string_view content, bool role,
                                    const runtime::StopReason* reason, bool chat);

/// Body of GET /v1/models.
[[nodiscard]] std::string modelsBody(std::string_view model);

/// An id in the shape clients expect, unique within a run.
[[nodiscard]] std::string makeCompletionId(bool chat);

}  // namespace tf::server
