#pragma once

#include <functional>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "tf/core/base/Error.h"
#include "tf/core/base/Types.h"
#include "tf/core/tokenizer/AssistantDecoder.h"
#include "tf/core/tokenizer/Tokenizer.h"
#include "tf/gpu/Backend.h"
#include "tf/runtime/ForwardRunner.h"
#include "tf/runtime/KVCache.h"
#include "tf/runtime/Model.h"
#include "tf/runtime/Sampler.h"
#include "tf/runtime/StopMatcher.h"

namespace tf::runtime {

/// Why generation ended. The caller usually wants to distinguish these: a
/// length stop means the answer was cut off, the others mean it finished.
enum class StopReason {
    /// An end-of-turn or end-of-sequence token was produced.
    StopToken,
    /// One of the caller's stop strings appeared in the output.
    StopString,
    /// `maxTokens` was reached.
    Length,
    /// The KV cache ran out of context.
    ContextFull,
    /// The callback asked to stop.
    Cancelled,
};

[[nodiscard]] std::string_view describe(StopReason reason);

struct GenerationOptions {
    SamplingParams sampling;

    /// Ceiling on generated tokens, not counting the prompt.
    u64 maxTokens = 512;

    /// Text sequences that end generation. Matched across token boundaries.
    std::vector<std::string> stopStrings;

    /// Token ids that end generation, usually the chat template's.
    std::vector<u32> stopTokens;

    /// Prompt tokens processed per batched pass. Must not exceed what the
    /// runner and the KV cache were built for.
    u32 prefillChunk = 128;
};

struct GenerationStats {
    u64 promptTokens = 0;
    /// Prompt tokens served from the KV cache rather than reprocessed. In a
    /// multi-turn conversation this is nearly the whole prompt.
    u64 cachedPromptTokens = 0;
    u64 generatedTokens = 0;
    double prefillSeconds = 0.0;
    double decodeSeconds = 0.0;
    StopReason reason = StopReason::Length;
    /// The seed the sampler actually used, so a run can be repeated.
    u64 seed = 0;

    /// Counted over tokens actually processed, since the cached ones cost
    /// nothing and would flatter the figure.
    [[nodiscard]] double prefillTokensPerSecond() const {
        const u64 processed = promptTokens - cachedPromptTokens;
        return prefillSeconds > 0.0 ? static_cast<double>(processed) / prefillSeconds : 0.0;
    }
    [[nodiscard]] double decodeTokensPerSecond() const {
        return decodeSeconds > 0.0 ? static_cast<double>(generatedTokens) / decodeSeconds : 0.0;
    }

    [[nodiscard]] std::string describe() const;
};

/// Runs a prompt to completion, streaming text out as it is produced.
///
/// Owns the KV cache and the forward runner, so one instance is one
/// conversation. The model is borrowed: its weights are the expensive part and
/// several generators can share them.
class Generator {
public:
    // Every special member is out of line, for the same reason ExpertStreamer
    // does it: ForwardRunner owns a unique_ptr to an incomplete type, so even a
    // defaulted default constructor needs its destructor available to unwind if
    // a later member throws.
    Generator();
    ~Generator();

    Generator(const Generator&) = delete;
    Generator& operator=(const Generator&) = delete;
    Generator(Generator&&) noexcept;
    Generator& operator=(Generator&&) noexcept;

    /// `contextLength` sizes the KV cache; `maxPrefillChunk` sizes the batched
    /// scratch and the sliding-window headroom, and caps
    /// GenerationOptions::prefillChunk.
    ///
    /// **`model` and `tokenizer` must outlive the generator and must not move.**
    /// Their addresses are stored, so relocating either afterwards - moving them
    /// into a struct, say - leaves the forward runner reading a moved-from
    /// object. That fails as a crash partway through the first generation, well
    /// away from the move that caused it. Construct them in their final home
    /// first, as tf-decode and the server both do.
    [[nodiscard]] static Result<Generator> create(gpu::IGpuBackend& backend, Model& model,
                                                  const Tokenizer& tokenizer,
                                                  u64 contextLength,
                                                  u32 maxPrefillChunk = 128);

    /// Called with each piece of decoded text as it becomes safe to show.
    /// Returning false stops generation.
    using TextCallback = std::function<bool(std::string_view)>;

    /// Called with the model's thinking as it arrives, when the caller wants
    /// to show or log it. Thinking is never mixed into the answer: see
    /// AssistantDecoder for why the raw token stream cannot simply be printed.
    using ThinkingCallback = std::function<void(std::string_view)>;

    /// Set to receive the thinking channel. Unset, it is discarded.
    void setThinkingCallback(ThinkingCallback callback) {
        onThinking_ = std::move(callback);
    }

    /// Runs `promptIds` and generates until a stop condition.
    ///
    /// Reuses whatever prefix `promptIds` shares with what the cache already
    /// holds, so a chat client resending the whole conversation each turn
    /// reprocesses only the new turn. Use reset() to start over.
    [[nodiscard]] Result<GenerationStats> generate(std::span<const u32> promptIds,
                                                   const GenerationOptions& options,
                                                   const TextCallback& onText);

    /// Clears the KV cache, the prompt cache and the generated-token history.
    /// The expert slot cache is deliberately left warm: its contents are still
    /// a good prediction of what the next prompt will want.
    void reset();

    /// Tokens the KV cache currently represents, prompt and generated together.
    /// This is what the next prompt is matched against.
    [[nodiscard]] std::span<const u32> cachedTokens() const noexcept { return cached_; }

    /// Tokens generated so far this session, which is what the repetition
    /// penalty looks back over.
    [[nodiscard]] std::span<const u32> history() const noexcept { return history_; }

    [[nodiscard]] u64 position() const noexcept;
    [[nodiscard]] u64 contextLength() const noexcept;

    /// Exposed for the diagnostics panel and the benchmarks in M13.
    [[nodiscard]] const ForwardRunner& runner() const noexcept { return runner_; }

private:
    [[nodiscard]] Status prefill(std::span<const u32> promptIds, u32 chunk);

    /// How much of `promptIds` the cache can serve. Bounded both by the shared
    /// prefix and by how far the sliding-window rings can be rewound.
    [[nodiscard]] u64 reusablePrefix(std::span<const u32> promptIds) const;

    Model* model_ = nullptr;
    const Tokenizer* tokenizer_ = nullptr;

    /// Held indirectly because ForwardRunner stores a pointer to it. Moving a
    /// Generator - which create() does on the way out - would otherwise leave
    /// the runner pointing at the old object's cache. The indirection keeps the
    /// cache's address stable across the move.
    std::unique_ptr<KVCacheManager> cache_;
    ForwardRunner runner_;
    u32 maxPrefillChunk_ = 128;

    ThinkingCallback onThinking_;

    /// Every token the KV cache holds, in order. Kept so the next prompt can be
    /// matched against it; this is the whole prompt cache.
    std::vector<u32> cached_;

    /// Every token generated this session. The repetition penalty reads it, and
    /// it is deliberately not reset between turns of one conversation.
    std::vector<u32> history_;
    std::vector<float> logits_;
};

}  // namespace tf::runtime
