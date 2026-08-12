#include "tf/runtime/ForwardRunner.h"

#include <algorithm>
#include <chrono>
#include <format>

#include "tf/core/math/Float.h"

namespace tf::runtime {
namespace {

/// Activations are fp32. Gemma 4's do not fit in fp16: layer 5's expert branch
/// peaks around 410 and the GeGLU product overflows. See
/// docs/MODEL_SEMANTICS.md.
constexpr u64 kActivationBytes = 4;

gpu::DeviceView viewOf(const gpu::BufferPtr& buffer, u64 offset = 0) {
    return gpu::DeviceView{.buffer = buffer.get(), .offset = offset};
}

}  // namespace

/// Every per-token intermediate. Allocated once at the largest shape any layer
/// needs, since layers differ: a full-attention layer projects Q to 8192 where
/// a sliding one projects to 4096.
struct ForwardRunner::Scratch {
    /// Row indices and per-row scales for the expert group being processed,
    /// staged on the host then uploaded once per expert.
    gpu::BufferPtr groupRows;    ///< u32  [maxTokens]
    gpu::BufferPtr groupScales;  ///< fp32 [maxTokens]
    gpu::BufferPtr groupInput;   ///< fp32 [maxTokens][hidden]
    gpu::BufferPtr groupOutput;  ///< fp32 [maxTokens][hidden]

    gpu::BufferPtr hidden;      ///< the residual stream
    gpu::BufferPtr residual;    ///< saved before the attention branch
    gpu::BufferPtr residual2;   ///< saved before the feed-forward branch
    gpu::BufferPtr normed;

    gpu::BufferPtr query;
    gpu::BufferPtr keyRaw;      ///< k_proj output, before k_norm and RoPE
    gpu::BufferPtr key;
    gpu::BufferPtr value;
    gpu::BufferPtr attention;
    gpu::BufferPtr attentionOut;

    gpu::BufferPtr sharedBranch;  ///< h1
    gpu::BufferPtr routedBranch;  ///< h2
    gpu::BufferPtr branchInput;   ///< pre_feedforward_layernorm_2 output

    gpu::BufferPtr denseGate;
    gpu::BufferPtr denseUp;
    gpu::BufferPtr denseAct;

    gpu::BufferPtr routerNormed;
    gpu::BufferPtr routerScores;
    gpu::BufferPtr routerIndices;  ///< u32 [topK]
    gpu::BufferPtr routerWeights;  ///< fp32 [topK]

    gpu::BufferPtr expertGate;
    gpu::BufferPtr expertUp;
    gpu::BufferPtr expertAct;
    gpu::BufferPtr expertOutputs;  ///< [topK][hidden]

    gpu::BufferPtr logits;
    gpu::BufferPtr token;  ///< u32 [1]
};

ForwardRunner::ForwardRunner() = default;
ForwardRunner::~ForwardRunner() = default;
ForwardRunner::ForwardRunner(ForwardRunner&&) noexcept = default;
ForwardRunner& ForwardRunner::operator=(ForwardRunner&&) noexcept = default;

u64 ForwardRunner::estimateScratchBytes(const ArchInfo& arch, u32 maxPrefillTokens) {
    u64 maxQuery = 0;
    u64 maxKeyValue = 0;
    for (u64 layer = 0; layer < arch.numLayers; ++layer) {
        maxQuery = std::max(maxQuery, arch.qProjOutFeatures(layer));
        maxKeyValue = std::max(maxKeyValue, arch.kvProjOutFeatures(layer));
    }

    const u64 n = maxPrefillTokens;
    const u64 expertRows = std::max(n, arch.topKExperts);

    // Grouped by what create() allocates, in the same order, so a buffer added
    // there is visibly missing here.
    u64 elements = 0;
    elements += 4 * n * arch.hiddenSize;             // hidden, residual x2, normed
    elements += 2 * n * maxQuery;                    // query, attention
    elements += 3 * n * maxKeyValue;                 // keyRaw, key, value
    elements += 4 * n * arch.hiddenSize;             // attentionOut, branches, branchInput
    elements += 3 * n * arch.intermediateSize;       // dense gate, up, act
    elements += n * arch.hiddenSize;                 // routerNormed
    elements += n * arch.numExperts;                 // routerScores
    elements += 2 * n * arch.topKExperts;            // router indices and weights
    elements += 3 * expertRows * arch.moeIntermediateSize;
    elements += expertRows * arch.hiddenSize;
    elements += 2 * n * arch.hiddenSize;             // group input and output
    elements += 2 * n;                               // group scales and rows
    elements += arch.vocabSize;                      // logits

    // A tenth on top, so the planner errs toward streaming one more layer
    // rather than toward an allocation that fails after the weights are in.
    const u64 bytes = elements * kActivationBytes;
    return bytes + bytes / 10;
}

Result<ForwardRunner> ForwardRunner::create(gpu::IGpuBackend& backend, Model& model,
                                            KVCacheManager& cache, u32 maxPrefillTokens) {
    const ArchInfo& arch = model.arch();

    if (maxPrefillTokens == 0) {
        return makeError(ErrorCode::InvalidArgument, "a prefill chunk of zero tokens");
    }

    ForwardRunner runner;
    runner.backend_ = &backend;
    runner.model_ = &model;
    runner.cache_ = &cache;
    runner.maxTokens_ = maxPrefillTokens;

    TF_TRY(runner.compute_, backend.createStream("compute"));
    TF_TRY(runner.copy_, backend.createStream("expert-copy"));
    TF_TRY(runner.uploadsReady_, backend.createEvent(/*withTiming=*/false));

    // Widest shapes across all layers, so one allocation serves every layer.
    u64 maxQuery = 0;
    u64 maxKeyValue = 0;
    for (u64 layer = 0; layer < arch.numLayers; ++layer) {
        maxQuery = std::max(maxQuery, arch.qProjOutFeatures(layer));
        maxKeyValue = std::max(maxKeyValue, arch.kvProjOutFeatures(layer));
    }

    auto scratch = std::make_unique<Scratch>();
    u64 allocated = 0;
    const auto alloc = [&](gpu::BufferPtr& target, u64 elements,
                           std::string name) -> Status {
        allocated += elements * kActivationBytes;
        TF_TRY(target, backend.allocate(gpu::MemoryKind::Device, elements * kActivationBytes,
                                        std::move(name)));
        return {};
    };

    // Everything a token owns is allocated `n` wide, so decode and prefill run
    // the same code against the same buffers and only the extent differs.
    const u64 n = maxPrefillTokens;

    TF_CHECK(alloc(scratch->hidden, n * arch.hiddenSize, "hidden"));
    TF_CHECK(alloc(scratch->residual, n * arch.hiddenSize, "residual"));
    TF_CHECK(alloc(scratch->residual2, n * arch.hiddenSize, "residual2"));
    TF_CHECK(alloc(scratch->normed, n * arch.hiddenSize, "normed"));

    TF_CHECK(alloc(scratch->query, n * maxQuery, "query"));
    TF_CHECK(alloc(scratch->keyRaw, n * maxKeyValue, "key-raw"));
    TF_CHECK(alloc(scratch->key, n * maxKeyValue, "key"));
    TF_CHECK(alloc(scratch->value, n * maxKeyValue, "value"));
    TF_CHECK(alloc(scratch->attention, n * maxQuery, "attention"));
    TF_CHECK(alloc(scratch->attentionOut, n * arch.hiddenSize, "attention-out"));

    TF_CHECK(alloc(scratch->sharedBranch, n * arch.hiddenSize, "shared-branch"));
    TF_CHECK(alloc(scratch->routedBranch, n * arch.hiddenSize, "routed-branch"));
    TF_CHECK(alloc(scratch->branchInput, n * arch.hiddenSize, "branch-input"));

    TF_CHECK(alloc(scratch->denseGate, n * arch.intermediateSize, "dense-gate"));
    TF_CHECK(alloc(scratch->denseUp, n * arch.intermediateSize, "dense-up"));
    TF_CHECK(alloc(scratch->denseAct, n * arch.intermediateSize, "dense-act"));

    TF_CHECK(alloc(scratch->routerNormed, n * arch.hiddenSize, "router-normed"));
    TF_CHECK(alloc(scratch->routerScores, n * arch.numExperts, "router-scores"));
    allocated += n * arch.topKExperts * sizeof(u32);
    TF_TRY(scratch->routerIndices,
           backend.allocate(gpu::MemoryKind::Device, n * arch.topKExperts * sizeof(u32),
                            "router-indices"));
    TF_CHECK(alloc(scratch->routerWeights, n * arch.topKExperts, "router-weights"));

    // The expert intermediates are indexed by gathered row rather than by
    // token, and one expert can be chosen by every token in the chunk, so they
    // are also n wide. The single-token path additionally needs topK slots of
    // expertOutputs, which n covers whenever n >= topK.
    const u64 expertRows = std::max(n, arch.topKExperts);
    TF_CHECK(alloc(scratch->expertGate, expertRows * arch.moeIntermediateSize, "expert-gate"));
    TF_CHECK(alloc(scratch->expertUp, expertRows * arch.moeIntermediateSize, "expert-up"));
    TF_CHECK(alloc(scratch->expertAct, expertRows * arch.moeIntermediateSize, "expert-act"));
    TF_CHECK(alloc(scratch->expertOutputs, expertRows * arch.hiddenSize, "expert-outputs"));

    TF_CHECK(alloc(scratch->groupInput, n * arch.hiddenSize, "group-input"));
    TF_CHECK(alloc(scratch->groupOutput, n * arch.hiddenSize, "group-output"));
    TF_CHECK(alloc(scratch->groupScales, n, "group-scales"));
    allocated += n * sizeof(u32);
    TF_TRY(scratch->groupRows,
           backend.allocate(gpu::MemoryKind::Device, n * sizeof(u32), "group-rows"));

    TF_CHECK(alloc(scratch->logits, arch.vocabSize, "logits"));
    allocated += sizeof(u32);
    TF_TRY(scratch->token,
           backend.allocate(gpu::MemoryKind::Device, sizeof(u32), "token"));

    runner.scratchBytes_ = allocated;
    runner.scratch_ = std::move(scratch);
    runner.routerIndices_.assign(static_cast<usize>(n * arch.topKExperts), 0);
    runner.routerWeights_.assign(static_cast<usize>(n * arch.topKExperts), 0.0f);
    return runner;
}

gpu::DeviceView ForwardRunner::logits() const { return viewOf(scratch_->logits); }

Status ForwardRunner::runAttention(u64 layer, u64 basePosition) {
    const ArchInfo& arch = model_->arch();
    const LayerWeights& weights = model_->layer(layer);
    gpu::IKernels& kernels = backend_->kernels();
    Scratch& s = *scratch_;

    const auto heads = static_cast<u32>(arch.numHeads);
    const auto kvHeads = static_cast<u32>(arch.kvHeadsFor(layer));
    const auto headDim = static_cast<u32>(arch.headDimFor(layer));
    const auto kvWidth = static_cast<u32>(arch.kvProjOutFeatures(layer));

    const float partialRotary = arch.isFullAttention(layer)
                                        ? static_cast<float>(arch.partialRotaryFactor)
                                        : 1.0f;

    // Q: project, per-head norm, then rotate. The per-head norms treat every
    // (token, head) pair as an independent row, which is exactly what the
    // [tokens][heads][headDim] layout gives.
    TF_CHECK(kernels.dequantGemm(*compute_,
                                 gpu::DequantGemmArgs{.weights = weights.qProj.weights,
                                                      .input = viewOf(s.normed),
                                                      .output = viewOf(s.query),
                                                      .tokens = tokens_}));
    TF_CHECK(kernels.rmsNorm(*compute_, gpu::RmsNormArgs{.input = viewOf(s.query),
                                                         .weight = weights.qNorm,
                                                         .output = viewOf(s.query),
                                                         .rows = tokens_ * heads,
                                                         .count = headDim,
                                                         .eps = static_cast<float>(
                                                                 arch.rmsNormEps)}));
    TF_CHECK(kernels.rope(*compute_,
                          gpu::RopeArgs{.data = viewOf(s.query),
                                        .tokens = tokens_,
                                        .heads = heads,
                                        .headDim = headDim,
                                        .position = basePosition,
                                        .theta = static_cast<float>(arch.ropeThetaFor(layer)),
                                        .partialRotaryFactor = partialRotary}));

    // K: project into keyRaw, which V also reads. Keeping the raw projection is
    // the whole point - V must not see k_norm or RoPE.
    TF_CHECK(kernels.dequantGemm(*compute_,
                                 gpu::DequantGemmArgs{.weights = weights.kProj.weights,
                                                      .input = viewOf(s.normed),
                                                      .output = viewOf(s.keyRaw),
                                                      .tokens = tokens_}));

    // V first, while keyRaw still holds the unmodified projection.
    if (weights.vProj.valid()) {
        TF_CHECK(kernels.dequantGemm(*compute_,
                                     gpu::DequantGemmArgs{.weights = weights.vProj.weights,
                                                          .input = viewOf(s.normed),
                                                          .output = viewOf(s.value),
                                                          .tokens = tokens_}));
    } else {
        // attention_k_eq_v: V is the raw K projection.
        TF_CHECK(backend_->enqueueCopy(*compute_, *s.value, 0, *s.keyRaw, 0,
                                       u64{tokens_} * kvWidth * kActivationBytes));
    }
    // v_norm has no learnable weight at all, which is why no weight view is
    // passed rather than a weight of ones.
    TF_CHECK(kernels.rmsNorm(*compute_,
                             gpu::RmsNormArgs{.input = viewOf(s.value),
                                              .weight = {},
                                              .output = viewOf(s.value),
                                              .rows = tokens_ * kvHeads,
                                              .count = headDim,
                                              .eps = static_cast<float>(arch.rmsNormEps)}));

    TF_CHECK(kernels.rmsNorm(*compute_, gpu::RmsNormArgs{.input = viewOf(s.keyRaw),
                                                         .weight = weights.kNorm,
                                                         .output = viewOf(s.key),
                                                         .rows = tokens_ * kvHeads,
                                                         .count = headDim,
                                                         .eps = static_cast<float>(
                                                                 arch.rmsNormEps)}));
    TF_CHECK(kernels.rope(*compute_,
                          gpu::RopeArgs{.data = viewOf(s.key),
                                        .tokens = tokens_,
                                        .heads = kvHeads,
                                        .headDim = headDim,
                                        .position = basePosition,
                                        .theta = static_cast<float>(arch.ropeThetaFor(layer)),
                                        .partialRotaryFactor = partialRotary}));

    // The whole chunk's keys and values must be in the cache before attention
    // runs, since a later token in the chunk attends to an earlier one.
    const LayerKVCache kv = cache_->layer(layer);
    TF_CHECK(kernels.kvWrite(*compute_, gpu::KvWriteArgs{.key = viewOf(s.key),
                                                         .value = viewOf(s.value),
                                                         .keyCache = kv.keys,
                                                         .valueCache = kv.values,
                                                         .tokens = tokens_,
                                                         .kvHeads = kv.kvHeads,
                                                         .headDim = kv.headDim,
                                                         .capacity = kv.capacity,
                                                         .position = basePosition,
                                                         .circular = kv.circular}));

    TF_CHECK(kernels.attention(
            *compute_,
            gpu::AttentionArgs{
                    .queries = viewOf(s.query),
                    .keyCache = kv.keys,
                    .valueCache = kv.values,
                    .output = viewOf(s.attention),
                    .tokens = tokens_,
                    .numHeads = heads,
                    .kvHeads = kvHeads,
                    .headDim = headDim,
                    .capacity = kv.capacity,
                    .basePosition = basePosition,
                    // Only the sliding layers window; the full ones see
                    // everything.
                    .slidingWindow = arch.isFullAttention(layer)
                                             ? 0u
                                             : static_cast<u32>(arch.slidingWindow),
                    .circular = kv.circular,
                    // Gemma 4 uses 1.0: q_norm and k_norm already normalize.
                    .scale = 1.0f}));

    TF_CHECK(kernels.dequantGemm(*compute_,
                                 gpu::DequantGemmArgs{.weights = weights.oProj.weights,
                                                      .input = viewOf(s.attention),
                                                      .output = viewOf(s.attentionOut),
                                                      .tokens = tokens_}));
    return {};
}

Status ForwardRunner::runSharedExpert(u64 layer) {
    const ArchInfo& arch = model_->arch();
    const LayerWeights& weights = model_->layer(layer);
    gpu::IKernels& kernels = backend_->kernels();
    Scratch& s = *scratch_;

    const auto width = static_cast<u32>(arch.intermediateSize);

    TF_CHECK(kernels.dequantGemm(*compute_,
                                 gpu::DequantGemmArgs{.weights = weights.gateProj.weights,
                                                      .input = viewOf(s.normed),
                                                      .output = viewOf(s.denseGate),
                                                      .tokens = tokens_}));
    TF_CHECK(kernels.dequantGemm(*compute_,
                                 gpu::DequantGemmArgs{.weights = weights.upProj.weights,
                                                      .input = viewOf(s.normed),
                                                      .output = viewOf(s.denseUp),
                                                      .tokens = tokens_}));
    // The gate is the projection that passes through GELU. Elementwise, so the
    // batch is just a longer vector.
    TF_CHECK(kernels.geglu(*compute_, gpu::GegluArgs{.gate = viewOf(s.denseGate),
                                                     .up = viewOf(s.denseUp),
                                                     .output = viewOf(s.denseAct),
                                                     .count = tokens_ * width}));
    TF_CHECK(kernels.dequantGemm(*compute_,
                                 gpu::DequantGemmArgs{.weights = weights.downProj.weights,
                                                      .input = viewOf(s.denseAct),
                                                      .output = viewOf(s.sharedBranch),
                                                      .tokens = tokens_}));
    return {};
}

Status ForwardRunner::runRoutedExpertsSingle(u64 layer) {
    const ArchInfo& arch = model_->arch();
    gpu::IKernels& kernels = backend_->kernels();
    Scratch& s = *scratch_;

    const auto width = static_cast<u32>(arch.moeIntermediateSize);
    const u64 hiddenBytes = arch.hiddenSize * kActivationBytes;

    for (u64 k = 0; k < arch.topKExperts; ++k) {
        const u32 expert = routerIndices_[static_cast<usize>(k)];
        // For a streamed layer the slot came from the plan; for a resident one
        // the expert is addressed directly and the slot is ignored.
        const u32 slot = k < slotForExpert_.size() ? slotForExpert_[static_cast<usize>(k)] : 0;
        TF_TRY(const auto projections, model_->expertWeights(layer, expert, slot));

        TF_CHECK(kernels.dequantGemv(
                *compute_, gpu::DequantGemvArgs{.weights = projections[0].weights,
                                                .input = viewOf(s.branchInput),
                                                .output = viewOf(s.expertGate)}));
        TF_CHECK(kernels.dequantGemv(
                *compute_, gpu::DequantGemvArgs{.weights = projections[1].weights,
                                                .input = viewOf(s.branchInput),
                                                .output = viewOf(s.expertUp)}));
        TF_CHECK(kernels.geglu(*compute_, gpu::GegluArgs{.gate = viewOf(s.expertGate),
                                                         .up = viewOf(s.expertUp),
                                                         .output = viewOf(s.expertAct),
                                                         .count = width}));
        TF_CHECK(kernels.dequantGemv(
                *compute_,
                gpu::DequantGemvArgs{.weights = projections[2].weights,
                                     .input = viewOf(s.expertAct),
                                     .output = viewOf(s.expertOutputs, k * hiddenBytes)}));
    }

    TF_CHECK(kernels.moeCombine(
            *compute_, gpu::MoeCombineArgs{.expertOutputs = viewOf(s.expertOutputs),
                                           .weights = viewOf(s.routerWeights),
                                           .output = viewOf(s.routedBranch),
                                           .topK = static_cast<u32>(arch.topKExperts),
                                           .hidden = static_cast<u32>(arch.hiddenSize)}));
    return {};
}

Status ForwardRunner::runOneExpertGroup(u64 layer, u32 expert, u32 slot, u32 count) {
    const ArchInfo& arch = model_->arch();
    gpu::IKernels& kernels = backend_->kernels();
    Scratch& s = *scratch_;

    const auto width = static_cast<u32>(arch.moeIntermediateSize);
    const auto hidden = static_cast<u32>(arch.hiddenSize);

    TF_TRY(const auto projections, model_->expertWeights(layer, expert, slot));

    TF_CHECK(backend_->enqueueUpload(
            *compute_, *s.groupRows, 0,
            ByteSpan{reinterpret_cast<const u8*>(groupRows_.data()), count * sizeof(u32)}));
    TF_CHECK(backend_->enqueueUpload(
            *compute_, *s.groupScales, 0,
            ByteSpan{reinterpret_cast<const u8*>(groupScales_.data()),
                     count * sizeof(float)}));

    TF_CHECK(kernels.gatherRows(*compute_,
                                gpu::GatherRowsArgs{.input = viewOf(s.branchInput),
                                                    .rows = viewOf(s.groupRows),
                                                    .output = viewOf(s.groupInput),
                                                    .count = count,
                                                    .width = hidden}));

    TF_CHECK(kernels.dequantGemm(*compute_,
                                 gpu::DequantGemmArgs{.weights = projections[0].weights,
                                                      .input = viewOf(s.groupInput),
                                                      .output = viewOf(s.expertGate),
                                                      .tokens = count}));
    TF_CHECK(kernels.dequantGemm(*compute_,
                                 gpu::DequantGemmArgs{.weights = projections[1].weights,
                                                      .input = viewOf(s.groupInput),
                                                      .output = viewOf(s.expertUp),
                                                      .tokens = count}));
    TF_CHECK(kernels.geglu(*compute_, gpu::GegluArgs{.gate = viewOf(s.expertGate),
                                                     .up = viewOf(s.expertUp),
                                                     .output = viewOf(s.expertAct),
                                                     .count = count * width}));
    TF_CHECK(kernels.dequantGemm(*compute_,
                                 gpu::DequantGemmArgs{.weights = projections[2].weights,
                                                      .input = viewOf(s.expertAct),
                                                      .output = viewOf(s.groupOutput),
                                                      .tokens = count}));

    // The router weight is folded in here rather than in a separate combine:
    // each token accumulates its experts' contributions into the branch as they
    // are produced, so nothing has to hold [tokens][topK][hidden].
    TF_CHECK(kernels.scatterAddRows(
            *compute_, gpu::ScatterAddRowsArgs{.input = viewOf(s.groupOutput),
                                               .rows = viewOf(s.groupRows),
                                               .scales = viewOf(s.groupScales),
                                               .output = viewOf(s.routedBranch),
                                               .count = count,
                                               .width = hidden}));
    return {};
}

Status ForwardRunner::runRoutedExpertsGrouped(u64 layer) {
    const ArchInfo& arch = model_->arch();
    gpu::IKernels& kernels = backend_->kernels();
    Scratch& s = *scratch_;

    const auto topK = static_cast<u32>(arch.topKExperts);
    const auto numExperts = static_cast<u32>(arch.numExperts);

    // Invert the routing: for each expert, which tokens chose it and with what
    // weight. This is the whole point of the grouped path - the alternative is
    // reading an expert's 3.19 MiB of weights once per token that selected it.
    std::vector<std::vector<u32>> tokensFor(numExperts);
    for (u32 token = 0; token < tokens_; ++token) {
        for (u32 k = 0; k < topK; ++k) {
            const u32 expert = routerIndices_[static_cast<usize>(token) * topK + k];
            if (expert >= numExperts) {
                return makeError(ErrorCode::GpuFailure,
                                 "layer {}: router selected expert {} of {}", layer, expert,
                                 numExperts);
            }
            tokensFor[expert].push_back(token);
        }
    }

    std::vector<u32> present;
    present.reserve(numExperts);
    for (u32 expert = 0; expert < numExperts; ++expert) {
        if (!tokensFor[expert].empty()) {
            present.push_back(expert);
        }
    }

    // The branch accumulates, so it starts empty.
    TF_CHECK(kernels.fillZero(
            *compute_, gpu::FillZeroArgs{.output = viewOf(s.routedBranch),
                                         .count = u64{tokens_} * arch.hiddenSize}));

    const auto runGroup = [&](u32 expert, u32 slot) -> Status {
        const auto& rows = tokensFor[expert];
        groupRows_.assign(rows.begin(), rows.end());
        groupScales_.clear();
        groupScales_.reserve(rows.size());
        for (const u32 token : rows) {
            // Find which of this token's k slots holds this expert; its weight
            // sits at the same index.
            for (u32 k = 0; k < topK; ++k) {
                const usize index = static_cast<usize>(token) * topK + k;
                if (routerIndices_[index] == expert) {
                    groupScales_.push_back(routerWeights_[index]);
                    break;
                }
            }
        }
        return runOneExpertGroup(layer, expert, slot, static_cast<u32>(rows.size()));
    };

    if (model_->layer(layer).expertsResident) {
        for (const u32 expert : present) {
            TF_CHECK(runGroup(expert, 0));
        }
        return {};
    }

    // A streamed layer holds only `slotsPerLayer` experts at a time, and a
    // chunk routinely touches every one of the 128. So the experts are fetched
    // and consumed a slotful at a time. Each expert is still read from disk at
    // most once for the whole chunk, which is what makes streamed prefill cost
    // a fraction of what streamed decode does per token.
    const auto slots = static_cast<usize>(model_->streamer().slotsPerLayer());
    for (usize start = 0; start < present.size(); start += slots) {
        const usize end = std::min(start + slots, present.size());
        const std::vector<u32> batch(present.begin() + static_cast<isize>(start),
                                     present.begin() + static_cast<isize>(end));

        TF_TRY(const ExpertCachePlan plan, model_->streamer().plan(layer, batch));
        TF_CHECK(model_->streamer().fetch(plan, *copy_));
        TF_CHECK(uploadsReady_->record(*copy_));
        TF_CHECK(uploadsReady_->wait(*compute_));

        for (usize i = 0; i < plan.experts.size(); ++i) {
            TF_CHECK(runGroup(plan.experts[i], plan.slots[i]));
        }

        // The next batch reuses these slots, so the kernels reading them must
        // finish before the fetch overwrites them.
        TF_CHECK(compute_->synchronize());
    }
    return {};
}

Status ForwardRunner::runLayer(u64 layer, u64 basePosition) {
    const ArchInfo& arch = model_->arch();
    const LayerWeights& weights = model_->layer(layer);
    gpu::IKernels& kernels = backend_->kernels();
    Scratch& s = *scratch_;

    const auto hidden = static_cast<u32>(arch.hiddenSize);
    const auto eps = static_cast<float>(arch.rmsNormEps);
    const auto topK = static_cast<u32>(arch.topKExperts);
    // The residual stream and everything derived from it is [tokens_][hidden],
    // so the norms take one row per token and the elementwise ops one long run.
    const u32 rows = tokens_;
    const u32 span = tokens_ * hidden;
    const u64 spanBytes = u64{span} * kActivationBytes;

    // ---- Attention branch ------------------------------------------------
    TF_CHECK(backend_->enqueueCopy(*compute_, *s.residual, 0, *s.hidden, 0, spanBytes));
    TF_CHECK(kernels.rmsNorm(*compute_, gpu::RmsNormArgs{.input = viewOf(s.hidden),
                                                         .weight = weights.inputNorm,
                                                         .output = viewOf(s.normed),
                                                         .rows = rows,
                                                         .count = hidden,
                                                         .eps = eps}));
    TF_CHECK(runAttention(layer, basePosition));

    // post_attention_layernorm applies to the branch, not the sum.
    TF_CHECK(kernels.rmsNorm(*compute_,
                             gpu::RmsNormArgs{.input = viewOf(s.attentionOut),
                                              .weight = weights.postAttentionNorm,
                                              .output = viewOf(s.attentionOut),
                                              .rows = rows,
                                              .count = hidden,
                                              .eps = eps}));
    TF_CHECK(kernels.add(*compute_, gpu::AddArgs{.a = viewOf(s.residual),
                                                 .b = viewOf(s.attentionOut),
                                                 .output = viewOf(s.hidden),
                                                 .count = span}));

    // ---- Feed-forward branches ------------------------------------------
    TF_CHECK(backend_->enqueueCopy(*compute_, *s.residual2, 0, *s.hidden, 0, spanBytes));

    // The router reads the pre-norm hidden state, not the normed branch input.
    TF_CHECK(kernels.rmsNorm(*compute_,
                             gpu::RmsNormArgs{.input = viewOf(s.hidden),
                                              .weight = weights.routerScaleFolded,
                                              .output = viewOf(s.routerNormed),
                                              .rows = rows,
                                              .count = hidden,
                                              .eps = eps}));
    TF_CHECK(kernels.dequantGemm(*compute_,
                                 gpu::DequantGemmArgs{.weights = weights.routerProj.weights,
                                                      .input = viewOf(s.routerNormed),
                                                      .output = viewOf(s.routerScores),
                                                      .tokens = tokens_}));
    TF_CHECK(kernels.routerTopK(
            *compute_,
            gpu::RouterTopKArgs{.scores = viewOf(s.routerScores),
                                .perExpertScale = weights.perExpertScale,
                                .outIndices = viewOf(s.routerIndices),
                                .outWeights = viewOf(s.routerWeights),
                                .tokens = tokens_,
                                .numExperts = static_cast<u32>(arch.numExperts),
                                .topK = topK}));

    // Both branches read the same hidden state, through different norms.
    TF_CHECK(kernels.rmsNorm(*compute_,
                             gpu::RmsNormArgs{.input = viewOf(s.hidden),
                                              .weight = weights.preFeedforwardNorm,
                                              .output = viewOf(s.normed),
                                              .rows = rows,
                                              .count = hidden,
                                              .eps = eps}));
    TF_CHECK(kernels.rmsNorm(*compute_,
                             gpu::RmsNormArgs{.input = viewOf(s.hidden),
                                              .weight = weights.preFeedforwardNorm2,
                                              .output = viewOf(s.branchInput),
                                              .rows = rows,
                                              .count = hidden,
                                              .eps = eps}));

    // The host needs the expert ids to address weights, so this readback is
    // unavoidable. It is the one hard synchronization per layer.
    const usize selections = static_cast<usize>(tokens_) * topK;
    TF_CHECK(backend_->enqueueDownload(
            *compute_,
            MutableByteSpan{reinterpret_cast<u8*>(routerIndices_.data()),
                            selections * sizeof(u32)},
            *s.routerIndices, 0));
    if (tokens_ > 1) {
        // The grouped path needs the weights on the host too, to build each
        // expert's scale vector. One extra 4 KiB download against a
        // synchronization that has to happen anyway.
        TF_CHECK(backend_->enqueueDownload(
                *compute_,
                MutableByteSpan{reinterpret_cast<u8*>(routerWeights_.data()),
                                selections * sizeof(float)},
                *s.routerWeights, 0));
    }
    TF_CHECK(compute_->synchronize());

    for (usize i = 0; i < selections; ++i) {
        if (routerIndices_[i] >= arch.numExperts) {
            return makeError(ErrorCode::GpuFailure,
                             "layer {}: router selected expert {} of {}", layer,
                             routerIndices_[i], arch.numExperts);
        }
    }

    // Enqueue the shared expert before touching the disk, so the GPU has work
    // to do while the CPU reads. This is where the overlap comes from.
    TF_CHECK(runSharedExpert(layer));

    if (tokens_ > 1) {
        // A chunk selects far more than topK distinct experts, so grouping by
        // expert - and, for a streamed layer, fetching in slot-sized batches -
        // replaces the decode path's single fetch entirely.
        const auto fetchStart = std::chrono::steady_clock::now();
        TF_CHECK(runRoutedExpertsGrouped(layer));
        if (timingEnabled_ && !weights.expertsResident) {
            timings_.expertFetchMillis += std::chrono::duration<double, std::milli>(
                                                  std::chrono::steady_clock::now() - fetchStart)
                                                  .count();
        }
    } else {
        if (!weights.expertsResident) {
            const auto fetchStart = std::chrono::steady_clock::now();

            TF_TRY(const ExpertCachePlan plan,
                   model_->streamer().plan(layer,
                                           std::span<const u32>{routerIndices_.data(), topK}));
            TF_CHECK(model_->streamer().fetch(plan, *copy_));

            // Slots are addressed by the plan, so reorder the indices to match
            // how runRoutedExpertsSingle will ask for them.
            for (usize k = 0; k < plan.slots.size(); ++k) {
                routerIndices_[k] = plan.experts[k];
            }
            slotForExpert_ = plan.slots;

            TF_CHECK(uploadsReady_->record(*copy_));
            TF_CHECK(uploadsReady_->wait(*compute_));

            if (timingEnabled_) {
                timings_.expertFetchMillis +=
                        std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - fetchStart)
                                .count();
            }
        } else {
            slotForExpert_.assign(topK, 0);
        }

        TF_CHECK(runRoutedExpertsSingle(layer));
    }

    // Each branch takes its own post norm before they are summed.
    TF_CHECK(kernels.rmsNorm(*compute_,
                             gpu::RmsNormArgs{.input = viewOf(s.sharedBranch),
                                              .weight = weights.postFeedforwardNorm1,
                                              .output = viewOf(s.sharedBranch),
                                              .rows = rows,
                                              .count = hidden,
                                              .eps = eps}));
    TF_CHECK(kernels.rmsNorm(*compute_,
                             gpu::RmsNormArgs{.input = viewOf(s.routedBranch),
                                              .weight = weights.postFeedforwardNorm2,
                                              .output = viewOf(s.routedBranch),
                                              .rows = rows,
                                              .count = hidden,
                                              .eps = eps}));
    TF_CHECK(kernels.add(*compute_, gpu::AddArgs{.a = viewOf(s.sharedBranch),
                                                 .b = viewOf(s.routedBranch),
                                                 .output = viewOf(s.hidden),
                                                 .count = span}));
    TF_CHECK(kernels.rmsNorm(*compute_,
                             gpu::RmsNormArgs{.input = viewOf(s.hidden),
                                              .weight = weights.postFeedforwardNorm,
                                              .output = viewOf(s.hidden),
                                              .rows = rows,
                                              .count = hidden,
                                              .eps = eps}));
    TF_CHECK(kernels.add(*compute_, gpu::AddArgs{.a = viewOf(s.residual2),
                                                 .b = viewOf(s.hidden),
                                                 .output = viewOf(s.hidden),
                                                 .count = span}));

    // The layer scalar multiplies the residual as well as the branch.
    TF_CHECK(kernels.scale(*compute_, gpu::ScaleArgs{.input = viewOf(s.hidden),
                                                     .output = viewOf(s.hidden),
                                                     .count = span,
                                                     .scalar = weights.layerScalar}));
    return {};
}

Status ForwardRunner::runHead() {
    const ArchInfo& arch = model_->arch();
    gpu::IKernels& kernels = backend_->kernels();
    Scratch& s = *scratch_;

    // Only the last token of the batch gets logits. For decode that is the only
    // token; for a prefill chunk the earlier tokens' predictions are already
    // known - they are the prompt - so computing them would read the whole
    // 396 MiB embedding table to produce numbers nobody looks at.
    const u64 lastToken = u64{tokens_ - 1} * arch.hiddenSize * kActivationBytes;

    TF_CHECK(kernels.rmsNorm(*compute_,
                             gpu::RmsNormArgs{.input = viewOf(s.hidden, lastToken),
                                              .weight = model_->finalNorm(),
                                              .output = viewOf(s.normed),
                                              .rows = 1,
                                              .count = static_cast<u32>(arch.hiddenSize),
                                              .eps = static_cast<float>(arch.rmsNormEps)}));

    // Tied weights: the embedding table transposed is the output head. No
    // sqrt(hiddenSize) scale here - that applies only on the way in.
    TF_CHECK(kernels.dequantGemv(
            *compute_, gpu::DequantGemvArgs{.weights = model_->embedding().weights,
                                            .input = viewOf(s.normed),
                                            .output = viewOf(s.logits)}));

    if (arch.finalLogitSoftcap > 0.0) {
        TF_CHECK(kernels.logitSoftcap(
                *compute_,
                gpu::LogitSoftcapArgs{.input = viewOf(s.logits),
                                      .output = viewOf(s.logits),
                                      .count = static_cast<u32>(arch.vocabSize),
                                      .cap = static_cast<float>(arch.finalLogitSoftcap)}));
    }
    return {};
}

Status ForwardRunner::decodeStep(u32 tokenId, u64 position, bool computeLogits) {
    const ArchInfo& arch = model_->arch();
    gpu::IKernels& kernels = backend_->kernels();

    const auto start = std::chrono::steady_clock::now();
    timings_ = StepTimings{};
    tokens_ = 1;

    if (tokenId >= arch.vocabSize) {
        return makeError(ErrorCode::InvalidArgument, "token {} is outside the vocabulary of {}",
                         tokenId, arch.vocabSize);
    }
    if (!cache_->canAccept(1) && position >= cache_->contextLength()) {
        return makeError(ErrorCode::InvalidArgument,
                         "position {} exceeds the {} token context", position,
                         cache_->contextLength());
    }

    // Embedding lookup, scaled by sqrt(hiddenSize) on the way in.
    TF_CHECK(kernels.embedLookup(
            *compute_, gpu::EmbedLookupArgs{.table = model_->embedding().weights,
                                            .output = viewOf(scratch_->hidden),
                                            .tokenId = tokenId}));

    const u64 layerCount = std::min(layerLimit_, arch.numLayers);

    for (u64 layer = 0; layer < layerCount; ++layer) {
        TF_CHECK(runLayer(layer, position));

        if (observer_) {
            std::vector<float> values(static_cast<usize>(arch.hiddenSize));
            TF_CHECK(backend_->enqueueDownload(
                    *compute_,
                    MutableByteSpan{reinterpret_cast<u8*>(values.data()),
                                    values.size() * sizeof(float)},
                    *scratch_->hidden, 0));
            TF_CHECK(compute_->synchronize());
            observer_(layer, values);
        }
    }

    if (computeLogits) {
        TF_CHECK(runHead());
    }
    TF_CHECK(compute_->synchronize());

    timings_.totalMillis =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
                    .count();
    return {};
}

Status ForwardRunner::prefillChunk(std::span<const u32> tokenIds, u64 basePosition) {
    const ArchInfo& arch = model_->arch();
    gpu::IKernels& kernels = backend_->kernels();

    const auto start = std::chrono::steady_clock::now();
    timings_ = StepTimings{};

    if (tokenIds.empty()) {
        return makeError(ErrorCode::InvalidArgument, "a prefill chunk of no tokens");
    }
    if (tokenIds.size() > maxTokens_) {
        return makeError(ErrorCode::InvalidArgument,
                         "a chunk of {} tokens exceeds the {} this runner was built for",
                         tokenIds.size(), maxTokens_);
    }
    // The sliding-window rings must have been sized for this chunk. Writing a
    // wider one would overwrite history the chunk's own first tokens still
    // need, and the result reads as plausible text rather than as a failure -
    // so this is checked rather than assumed.
    if (tokenIds.size() > cache_->maxPrefillTokens()) {
        return makeError(ErrorCode::InvalidArgument,
                         "a chunk of {} tokens exceeds the {} the KV cache rings were sized "
                         "for; the sliding window would lose rows this chunk still reads",
                         tokenIds.size(), cache_->maxPrefillTokens());
    }
    for (const u32 tokenId : tokenIds) {
        if (tokenId >= arch.vocabSize) {
            return makeError(ErrorCode::InvalidArgument,
                             "token {} is outside the vocabulary of {}", tokenId,
                             arch.vocabSize);
        }
    }
    if (basePosition + tokenIds.size() > cache_->contextLength()) {
        return makeError(ErrorCode::InvalidArgument,
                         "positions {}..{} exceed the {} token context", basePosition,
                         basePosition + tokenIds.size() - 1, cache_->contextLength());
    }

    tokens_ = static_cast<u32>(tokenIds.size());

    // The embedding is a row lookup per token, so there is nothing to batch:
    // each reads 2816 of the table's 262144 rows and they do not share work.
    const u64 hiddenBytes = arch.hiddenSize * kActivationBytes;
    for (u32 i = 0; i < tokens_; ++i) {
        TF_CHECK(kernels.embedLookup(
                *compute_,
                gpu::EmbedLookupArgs{.table = model_->embedding().weights,
                                     .output = viewOf(scratch_->hidden, i * hiddenBytes),
                                     .tokenId = tokenIds[i]}));
    }

    const u64 layerCount = std::min(layerLimit_, arch.numLayers);
    for (u64 layer = 0; layer < layerCount; ++layer) {
        TF_CHECK(runLayer(layer, basePosition));
    }

    TF_CHECK(compute_->synchronize());

    timings_.totalMillis =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
                    .count();
    return {};
}

Result<u32> ForwardRunner::greedyToken() {
    const ArchInfo& arch = model_->arch();
    gpu::IKernels& kernels = backend_->kernels();

    TF_CHECK(kernels.argmax(*compute_,
                            gpu::ArgmaxArgs{.input = viewOf(scratch_->logits),
                                            .output = viewOf(scratch_->token),
                                            .count = static_cast<u32>(arch.vocabSize)}));

    u32 token = 0;
    TF_CHECK(backend_->enqueueDownload(
            *compute_, MutableByteSpan{reinterpret_cast<u8*>(&token), sizeof(token)},
            *scratch_->token, 0));
    TF_CHECK(compute_->synchronize());
    return token;
}

Status ForwardRunner::setHiddenState(std::span<const float> values) {
    const ArchInfo& arch = model_->arch();
    if (values.size() != arch.hiddenSize) {
        return makeError(ErrorCode::InvalidArgument,
                         "hidden state has {} values but the model expects {}", values.size(),
                         arch.hiddenSize);
    }

    // One state means one token, so the head and the layer ops that follow read
    // the extent that was actually written rather than whatever batch size the
    // previous call left behind.
    tokens_ = 1;

    TF_CHECK(backend_->enqueueUpload(
            *compute_, *scratch_->hidden, 0,
            ByteSpan{reinterpret_cast<const u8*>(values.data()), values.size() * sizeof(float)}));
    TF_CHECK(compute_->synchronize());
    return {};
}

Result<std::vector<float>> ForwardRunner::readHiddenState() {
    return readScratch("hidden", static_cast<usize>(model_->arch().hiddenSize));
}

Result<std::vector<float>> ForwardRunner::readScratch(std::string_view name, usize count) {
    Scratch& s = *scratch_;

    // Router indices are the only u32 buffer here, and lastRouterIndices covers
    // it, so everything reachable by name is fp32.
    const gpu::Buffer* buffer = nullptr;
    if (name == "hidden") {
        buffer = s.hidden.get();
    } else if (name == "normed") {
        buffer = s.normed.get();
    } else if (name == "query") {
        buffer = s.query.get();
    } else if (name == "keyRaw") {
        buffer = s.keyRaw.get();
    } else if (name == "key") {
        buffer = s.key.get();
    } else if (name == "value") {
        buffer = s.value.get();
    } else if (name == "attention") {
        buffer = s.attention.get();
    } else if (name == "attentionOut") {
        buffer = s.attentionOut.get();
    } else if (name == "sharedBranch") {
        buffer = s.sharedBranch.get();
    } else if (name == "routedBranch") {
        buffer = s.routedBranch.get();
    } else if (name == "branchInput") {
        buffer = s.branchInput.get();
    } else if (name == "routerNormed") {
        buffer = s.routerNormed.get();
    } else if (name == "routerScores") {
        buffer = s.routerScores.get();
    } else if (name == "routerWeights") {
        buffer = s.routerWeights.get();
    } else if (name == "expertOutputs") {
        buffer = s.expertOutputs.get();
    } else if (name == "denseGate") {
        buffer = s.denseGate.get();
    } else if (name == "denseUp") {
        buffer = s.denseUp.get();
    } else if (name == "denseAct") {
        buffer = s.denseAct.get();
    } else {
        return makeError(ErrorCode::NotFound, "no scratch buffer named '{}'", name);
    }

    std::vector<float> values(count);
    TF_CHECK(backend_->enqueueDownload(
            *compute_,
            MutableByteSpan{reinterpret_cast<u8*>(values.data()), count * sizeof(float)},
            *buffer, 0));
    TF_CHECK(compute_->synchronize());
    return values;
}

Status ForwardRunner::runSingleLayer(u64 layer, u64 position) {
    TF_CHECK(runLayer(layer, position));
    TF_CHECK(compute_->synchronize());
    return {};
}

Status ForwardRunner::runHeadOnly() {
    TF_CHECK(runHead());
    TF_CHECK(compute_->synchronize());
    return {};
}

Result<std::vector<float>> ForwardRunner::readLogits() {
    const ArchInfo& arch = model_->arch();

    std::vector<float> values(static_cast<usize>(arch.vocabSize));
    TF_CHECK(backend_->enqueueDownload(
            *compute_,
            MutableByteSpan{reinterpret_cast<u8*>(values.data()),
                            values.size() * sizeof(float)},
            *scratch_->logits, 0));
    TF_CHECK(compute_->synchronize());
    return values;
}

}  // namespace tf::runtime
