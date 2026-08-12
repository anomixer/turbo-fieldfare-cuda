#pragma once

#include <functional>
#include <span>
#include <vector>

#include "tf/core/base/Error.h"
#include "tf/core/base/Types.h"
#include "tf/gpu/Backend.h"
#include "tf/gpu/Kernels.h"
#include "tf/runtime/KVCache.h"
#include "tf/runtime/Model.h"

namespace tf::runtime {

/// Where the time went in one decode step, for the diagnostics panel and M13.
struct StepTimings {
    double attentionMillis = 0.0;
    double sharedExpertMillis = 0.0;
    double routedExpertMillis = 0.0;
    double expertFetchMillis = 0.0;
    double headMillis = 0.0;
    double totalMillis = 0.0;
};

/// Executes the decode path: one token in, a full logit vector out.
///
/// The layer sequence follows docs/MODEL_SEMANTICS.md exactly, including the
/// details that are easy to get subtly wrong - V branching off the raw K
/// projection, the router reading the pre-norm hidden state, and the layer
/// scalar multiplying the residual as well as the branch.
class ForwardRunner {
public:
    // Out of line, like ExpertStreamer's: Scratch is incomplete here, so even
    // the defaulted default constructor needs the destructor available in order
    // to unwind. Held by value inside Generator, which is where this first
    // stopped compiling.
    ForwardRunner();
    ~ForwardRunner();

    ForwardRunner(const ForwardRunner&) = delete;
    ForwardRunner& operator=(const ForwardRunner&) = delete;
    ForwardRunner(ForwardRunner&&) noexcept;
    ForwardRunner& operator=(ForwardRunner&&) noexcept;

    /// `maxPrefillTokens` sizes the scratch buffers, and so caps the chunk
    /// prefillChunk will accept. Every per-token intermediate is allocated at
    /// that width, which is why it is a constructor argument rather than
    /// something the caller varies per call.
    [[nodiscard]] static Result<ForwardRunner> create(gpu::IGpuBackend& backend, Model& model,
                                                      KVCacheManager& cache,
                                                      u32 maxPrefillTokens = 128);

    /// Runs one token through the whole stack, writing its KV at `position` and
    /// leaving softcapped logits in logits().
    ///
    /// Does not advance the cache; the caller decides whether the step counts,
    /// which matters when a sampled token is rejected.
    ///
    /// `computeLogits` false skips the output head. Prefill wants this for
    /// every token but the last: the head multiplies against the whole tied
    /// embedding table, which is 396 MiB of the 1.26 GiB resident set, and
    /// prompt tokens produce logits nobody reads.
    [[nodiscard]] Status decodeStep(u32 tokenId, u64 position, bool computeLogits = true);

    /// Runs a whole chunk of prompt tokens in one pass, writing their KV at
    /// `basePosition + i` and leaving the residual stream holding all of them.
    ///
    /// This exists because decode is bound by weight bandwidth, not arithmetic:
    /// a single token reads all 1.26 GiB of the resident set to produce one
    /// output. A chunk reads the same weights once for every token in it, so
    /// prompt processing runs several times faster per token than generation
    /// does - and unlike decode, the routed experts are grouped, so each
    /// expert's weights are read once for the whole chunk rather than once per
    /// token that selected it.
    ///
    /// Does not run the output head. Prompt tokens other than the last produce
    /// logits nobody reads; call runHeadOnly() afterwards for the last one,
    /// which is what the residual stream is left positioned on.
    ///
    /// `tokenIds` must not exceed maxPrefillTokens().
    [[nodiscard]] Status prefillChunk(std::span<const u32> tokenIds, u64 basePosition);

    /// Largest chunk prefillChunk will accept, fixed at construction.
    [[nodiscard]] u32 maxPrefillTokens() const noexcept { return maxTokens_; }

    /// Device memory the scratch buffers occupy. Worth reporting: it scales
    /// with maxPrefillTokens, and on a card whose VRAM barely holds the model
    /// it is the difference between loading and not.
    [[nodiscard]] u64 scratchBytes() const noexcept { return scratchBytes_; }

    /// What the scratch will cost before any of it is allocated, so the
    /// residency planner can reserve it rather than discover it too late.
    ///
    /// Deliberately an over-estimate of a few percent: it is better for the
    /// planner to stream one more layer than for the allocation to fail after
    /// the weights are already in VRAM.
    [[nodiscard]] static u64 estimateScratchBytes(const ArchInfo& arch, u32 maxPrefillTokens);

    /// Greedy token from the current logits.
    [[nodiscard]] Result<u32> greedyToken();

    /// Copies the logits to the host. Allocating 262144 floats per call, so
    /// intended for sampling and tests rather than the inner loop.
    [[nodiscard]] Result<std::vector<float>> readLogits();

    [[nodiscard]] gpu::DeviceView logits() const;

    [[nodiscard]] const StepTimings& lastStepTimings() const noexcept { return timings_; }

    /// Enables per-phase GPU timing. Costs a few synchronizations per step, so
    /// it is off unless asked for.
    void setTimingEnabled(bool enabled) noexcept { timingEnabled_ = enabled; }

    /// Called with the residual stream after each layer. Downloads the hidden
    /// state and stalls the pipeline, so this is a debugging tool only - but it
    /// is the fastest way to localize a layer that saturates, collapses or
    /// produces non-finite values.
    using LayerObserver = std::function<void(u64 layer, std::span<const float> hidden)>;
    void setLayerObserver(LayerObserver observer) { observer_ = std::move(observer); }

    /// Runs only the first `count` layers before the head. A bisection tool:
    /// with zero layers the result exercises the embedding, final norm and tied
    /// head alone, which should return the input token, since the head is the
    /// embedding transposed and a vector is most similar to itself.
    void setLayerLimit(u64 count) noexcept { layerLimit_ = count; }

    /// Restores the default of running every layer. Distinct from a limit of
    /// zero, which runs none.
    void clearLayerLimit() noexcept { layerLimit_ = kNoLayerLimit; }

    static constexpr u64 kNoLayerLimit = ~u64{0};

    // ---- Debug access ----------------------------------------------------
    //
    // Enough visibility to diff the GPU against the CPU reference stage by
    // stage. Without it a mismatch localizes only to "somewhere in 30 layers".

    /// Overwrites the residual stream, so a layer can be driven from a known
    /// input instead of from a real embedding.
    [[nodiscard]] Status setHiddenState(std::span<const float> values);

    [[nodiscard]] Result<std::vector<float>> readHiddenState();

    /// Reads a named intermediate. Valid names are the scratch buffers:
    /// "hidden", "normed", "query", "keyRaw", "key", "value", "attention",
    /// "attentionOut", "sharedBranch", "routedBranch", "branchInput",
    /// "routerNormed", "routerScores", "routerWeights", "expertOutputs",
    /// "denseGate", "denseUp", "denseAct".
    [[nodiscard]] Result<std::vector<float>> readScratch(std::string_view name,
                                                         usize count);

    /// Router expert ids selected by the most recent layer, [tokens][topK].
    ///
    /// The backing vector is sized for a full prefill chunk, so this returns
    /// only the part the last call actually filled - for a decode step, one
    /// token's topK.
    [[nodiscard]] std::span<const u32> lastRouterIndices() const noexcept {
        return std::span<const u32>{routerIndices_}.first(
                static_cast<usize>(tokens_) * model_->arch().topKExperts);
    }

    /// Runs a single layer against the current hidden state, leaving the result
    /// in it. Used by the reference comparison.
    [[nodiscard]] Status runSingleLayer(u64 layer, u64 position);

    /// Runs only the output head - final norm, tied embedding head, softcap -
    /// against the last token of the current residual stream. Lets the head be
    /// compared against the reference without a full 30-layer forward pass, and
    /// is how prefill produces the logits for its final token.
    [[nodiscard]] Status runHeadOnly();

private:
    struct Scratch;

    [[nodiscard]] Status runLayer(u64 layer, u64 basePosition);
    [[nodiscard]] Status runAttention(u64 layer, u64 basePosition);
    [[nodiscard]] Status runSharedExpert(u64 layer);
    [[nodiscard]] Status runHead();

    /// One GEMV per selected expert. Correct for any token count but only
    /// sensible for one: with n tokens it reads each expert n times.
    [[nodiscard]] Status runRoutedExpertsSingle(u64 layer);

    /// One GEMM per distinct expert the chunk selected, over the tokens that
    /// selected it. Reads each expert's weights once per chunk.
    [[nodiscard]] Status runRoutedExpertsGrouped(u64 layer);

    /// Runs one expert over `count` gathered rows and accumulates the result
    /// into the routed branch.
    [[nodiscard]] Status runOneExpertGroup(u64 layer, u32 expert, u32 slot, u32 count);

    gpu::IGpuBackend* backend_ = nullptr;
    Model* model_ = nullptr;
    KVCacheManager* cache_ = nullptr;

    /// Kernels and expert uploads run on separate streams so the shared-expert
    /// branch can occupy the GPU while the CPU reads experts from disk.
    gpu::StreamPtr compute_;
    gpu::StreamPtr copy_;
    gpu::EventPtr uploadsReady_;

    std::unique_ptr<Scratch> scratch_;

    /// Tokens in the batch currently being run: 1 for decode, up to maxTokens_
    /// for a prefill chunk. Every scratch buffer is indexed by it.
    u32 tokens_ = 1;
    u32 maxTokens_ = 1;
    u64 scratchBytes_ = 0;

    /// Router selections read back per layer, [tokens_][topK]. The host needs
    /// them to address expert weights, whether those are streamed or resident.
    std::vector<u32> routerIndices_;
    /// Router weights read back alongside, same shape. Only the grouped path
    /// needs these on the host, to build each expert's scale vector.
    std::vector<float> routerWeights_;
    /// Scratch for the grouped path: for the expert being processed, the token
    /// rows that selected it and the weight each applies.
    std::vector<u32> groupRows_;
    std::vector<float> groupScales_;
    /// Slot each selected expert occupies, filled by the streamer's plan.
    /// Meaningless, and unused, for a resident layer.
    std::vector<u32> slotForExpert_;

    StepTimings timings_;
    bool timingEnabled_ = false;
    LayerObserver observer_;
    /// kNoLayerLimit runs everything; zero runs no layers at all.
    u64 layerLimit_ = kNoLayerLimit;
};

}  // namespace tf::runtime
