#pragma once

#include <string>
#include <string_view>
#include <vector>

#include "tf/core/base/Error.h"
#include "tf/core/base/Types.h"
#include "tf/core/tokenizer/ChatTemplate.h"
#include "tf/runtime/Sampler.h"

/// The wire protocol between the decode service and its front ends.
///
/// The service exists so that the GUI does not own a CUDA context and 13 GiB of
/// VRAM. A XAML crash, a hung UI thread or a shader-compile stall would
/// otherwise take the model with it, and reloading costs about eight seconds -
/// long enough that users would learn not to close the window.
///
/// Framing is a four-byte little-endian length followed by UTF-8 JSON. Named
/// pipes can preserve message boundaries themselves, but a message larger than
/// the pipe buffer then arrives as ERROR_MORE_DATA and has to be reassembled
/// anyway, so explicit framing is both simpler and the same amount of work.
namespace tf::svc {

/// Bumped when a change would make an older peer misread a message. The service
/// refuses a mismatch rather than guessing, because a silently misparsed
/// generate request is worse than a clear refusal.
inline constexpr u32 kProtocolVersion = 1;

/// Default pipe. Per-user by convention rather than by ACL: two sessions on one
/// machine would otherwise fight over one model.
[[nodiscard]] std::string defaultPipeName();

enum class MessageKind {
    // Client to service.
    Hello,
    Generate,
    Cancel,
    Reset,
    Status,
    Shutdown,
    // Service to client.
    Ready,
    Token,
    Thinking,
    Done,
    Error,
    StatusReport,
};

[[nodiscard]] std::string_view toString(MessageKind kind) noexcept;
[[nodiscard]] Result<MessageKind> parseKind(std::string_view text);

/// One request. `id` is echoed on every message about it, so a front end can
/// tell a late token from an old request apart from the current one.
struct GenerateRequest {
    u64 id = 0;
    /// Chat turns. Empty means the raw prompt below is used instead.
    std::vector<ChatMessage> messages;
    std::string prompt;
    bool chat = true;

    runtime::SamplingParams sampling;
    u64 maxTokens = 512;
    std::vector<std::string> stopStrings;
    /// Send the thinking channel as well as the answer.
    bool includeThinking = false;
};

struct ReadyInfo {
    u32 version = kProtocolVersion;
    std::string model;
    u64 contextLength = 0;
    /// Device the model is running on, for the status bar.
    std::string device;
    /// True when nothing streams from disk, which the inspector shows.
    bool fullyResident = true;
};

struct DoneInfo {
    u64 id = 0;
    std::string reason;
    u64 promptTokens = 0;
    u64 cachedPromptTokens = 0;
    u64 generatedTokens = 0;
    double prefillSeconds = 0.0;
    double decodeSeconds = 0.0;
};

struct StatusInfo {
    std::string model;
    u64 contextLength = 0;
    u64 position = 0;
    u64 requests = 0;
    u64 promptTokens = 0;
    u64 cachedPromptTokens = 0;
    u64 generatedTokens = 0;
    /// Zero when nothing streams.
    double expertHitRate = 0.0;
    u64 expertBytesRead = 0;
    bool busy = false;
};

/// A decoded message. Only the field matching `kind` is meaningful.
struct Message {
    MessageKind kind = MessageKind::Hello;
    u32 version = kProtocolVersion;

    GenerateRequest generate;
    ReadyInfo ready;
    DoneInfo done;
    StatusInfo status;

    /// Token, Thinking: the text. Error: the message. Cancel: unused.
    std::string text;
    /// Token, Thinking, Error, Cancel: which request this concerns.
    u64 id = 0;

    [[nodiscard]] std::string encode() const;
    [[nodiscard]] static Result<Message> decode(std::string_view json);
};

// Convenience constructors, so call sites read as what they send.
[[nodiscard]] Message makeHello();
[[nodiscard]] Message makeReady(ReadyInfo info);
[[nodiscard]] Message makeToken(u64 id, std::string text);
[[nodiscard]] Message makeThinking(u64 id, std::string text);
[[nodiscard]] Message makeDone(DoneInfo info);
[[nodiscard]] Message makeError(u64 id, std::string message);
[[nodiscard]] Message makeStatusReport(StatusInfo info);

/// Prefixes `payload` with its length. Separate from encode() so a caller can
/// frame something it did not build.
[[nodiscard]] std::string frame(std::string_view payload);

/// Largest message accepted. A prompt has to fit comfortably; nothing else
/// travels this way, so anything larger is a bad or hostile peer.
inline constexpr u32 kMaxMessageBytes = 32 * 1024 * 1024;

}  // namespace tf::svc
