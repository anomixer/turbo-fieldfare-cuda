#pragma once

#include "tf/core/base/Error.h"
#include "tf/core/base/Types.h"
#include "tf/core/format/Quantization.h"
#include "tf/gpu/Backend.h"

/// Semantic GPU operations the runtime calls.
///
/// Not a generic "bind these resources and launch" abstraction: CUDA and D3D12
/// share no useful notion of a pipeline binding, so each backend implements
/// these operations however suits it. The runtime only ever names what it wants
/// computed.
///
/// Data conventions, matching upstream and docs/MODEL_SEMANTICS.md:
///   - activations are fp32: Gemma 4's do not fit in fp16
///   - quantized weights are packed u32 with bf16 scales and biases
///   - norm weights and other small parameters are bf16
///   - kernels accumulate in fp32 regardless of storage type
namespace tf::gpu {

/// A location in device memory: a buffer plus a byte offset. Length is implied
/// by the operation's shape parameters, so a view stays cheap to pass around.
struct DeviceView {
    Buffer* buffer = nullptr;
    u64 offset = 0;

    [[nodiscard]] bool valid() const noexcept { return buffer != nullptr; }

    /// Advances by `bytes`, for stepping through a packed layout.
    [[nodiscard]] DeviceView at(u64 bytes) const noexcept {
        return DeviceView{.buffer = buffer, .offset = offset + bytes};
    }
};

/// The three tensors of one MLX affine quantized linear.
struct QuantizedWeights {
    DeviceView packed;  ///< u32
    DeviceView scales;  ///< bf16
    DeviceView biases;  ///< bf16
    QuantizedLinearLayout layout;
};

/// RMS normalization over `rows` independent vectors of `count` elements.
///
/// One operation covers all three uses: the hidden-state norms (rows = 1), the
/// per-head q_norm and k_norm (rows = numHeads, count = headDim), and v_norm,
/// which passes an invalid `weight` to mean no learnable scale at all.
///
/// Gemma 4 applies plain `weight`, not `1 + weight`.
struct RmsNormArgs {
    DeviceView input;   ///< fp32 [rows][count]
    DeviceView weight;  ///< bf16 [count], invalid for the no-scale form
    DeviceView output;  ///< fp32 [rows][count]
    u32 rows = 1;
    u32 count = 0;
    float eps = 1e-6f;
};

/// out[o] = sum_i dequantize(W[o][i]) * x[i]
///
/// Handles both the 4-bit weights and the 8-bit router projection; the width
/// comes from `weights.layout.spec`.
struct DequantGemvArgs {
    QuantizedWeights weights;
    DeviceView input;   ///< fp32 [inFeatures]
    DeviceView output;  ///< fp32 [outFeatures]
};

/// out[t][o] = sum_i dequantize(W[o][i]) * x[t][i], for `tokens` tokens.
///
/// Same weights and same layout as the GEMV, batched over tokens. Prefill uses
/// this instead of a GEMV per token because the matrix is read once for the
/// whole batch rather than once per token, which is the whole reason prompt
/// processing can outrun decode.
struct DequantGemmArgs {
    QuantizedWeights weights;
    DeviceView input;   ///< fp32 [tokens][inFeatures]
    DeviceView output;  ///< fp32 [tokens][outFeatures]
    u32 tokens = 1;
};

/// Row lookup from the quantized embedding table, scaled by sqrt(hiddenSize).
struct EmbedLookupArgs {
    QuantizedWeights table;
    DeviceView output;  ///< fp32 [hiddenSize]
    u32 tokenId = 0;
};

/// Rotary embedding over `heads` heads of `headDim` elements, in place.
///
/// `partialRotaryFactor` below 1.0 rotates only the leading fraction of each
/// head and passes the rest through, which is what the full-attention layers
/// need.
struct RopeArgs {
    DeviceView data;  ///< fp32 [tokens][heads][headDim], modified in place
    u32 tokens = 1;
    u32 heads = 0;
    u32 headDim = 0;
    /// Token `t` rotates at `position + t`.
    u64 position = 0;
    float theta = 10000.0f;
    float partialRotaryFactor = 1.0f;
};

/// gelu_approx(gate) * up, elementwise. The gate is what passes through GELU.
struct GegluArgs {
    DeviceView gate;    ///< fp32 [count]
    DeviceView up;      ///< fp32 [count]
    DeviceView output;  ///< fp32 [count]
    u32 count = 0;
};

/// output = a + b, elementwise. Used for residual joins.
struct AddArgs {
    DeviceView a;
    DeviceView b;
    DeviceView output;
    u32 count = 0;
};

/// output = input * scalar. Carries the per-layer `layer_scalar`, which is read
/// once at load rather than dereferenced per token.
struct ScaleArgs {
    DeviceView input;
    DeviceView output;
    u32 count = 0;
    float scalar = 1.0f;
};

/// tanh(x / cap) * cap over the logits.
struct LogitSoftcapArgs {
    DeviceView input;   ///< fp32 [count]
    DeviceView output;  ///< fp32 [count]
    u32 count = 0;
    float cap = 30.0f;
};

/// Index of the maximum, ties resolved to the lowest index so greedy decoding
/// is deterministic.
struct ArgmaxArgs {
    DeviceView input;   ///< fp32 [count]
    DeviceView output;  ///< u32 [1]
    u32 count = 0;
};

/// Writes one token's K and V into the cache.
///
/// The 25 sliding-window layers use a ring: logical position p lands at row
/// p % capacity, and older rows are overwritten. The 5 full-attention layers
/// append linearly.
struct KvWriteArgs {
    DeviceView key;         ///< fp32 [tokens][kvHeads][headDim]
    DeviceView value;       ///< fp32 [tokens][kvHeads][headDim]
    DeviceView keyCache;    ///< fp16 [kvHeads][capacity][headDim]
    DeviceView valueCache;  ///< fp16 [kvHeads][capacity][headDim]
    u32 tokens = 1;
    u32 kvHeads = 0;
    u32 headDim = 0;
    u32 capacity = 0;
    /// Token `t` is written at `position + t`.
    u64 position = 0;
    bool circular = false;
};

/// Causal attention over the KV cache for `tokens` queries.
///
/// Grouped-query: `numHeads` query heads share `kvHeads` cache heads, with head
/// h reading KV head h / (numHeads / kvHeads).
///
/// Query `t` sits at position `basePosition + t` and attends to everything up
/// to and including itself, so causality needs no mask tensor and a decode step
/// is just `tokens == 1`. The caller must have written all of the chunk's own
/// keys and values before this runs.
struct AttentionArgs {
    DeviceView queries;     ///< fp32 [tokens][numHeads][headDim]
    DeviceView keyCache;    ///< fp16 [kvHeads][capacity][headDim]
    DeviceView valueCache;  ///< fp16 [kvHeads][capacity][headDim]
    DeviceView output;      ///< fp32 [tokens][numHeads][headDim]
    u32 tokens = 1;
    u32 numHeads = 0;
    u32 kvHeads = 0;
    u32 headDim = 0;
    u32 capacity = 0;
    u64 basePosition = 0;
    /// 0 gives full causal attention.
    u32 slidingWindow = 0;
    bool circular = false;
    /// 1.0 for Gemma 4: q_norm and k_norm already normalize.
    float scale = 1.0f;
};

/// Selects the top-k experts by score, softmaxes over just those, then applies
/// the per-expert scale.
///
/// Selection is by raw score and the softmax happens afterwards, so the
/// resulting weights deliberately do not sum to one.
struct RouterTopKArgs {
    DeviceView scores;          ///< fp32 [tokens][numExperts]
    DeviceView perExpertScale;  ///< bf16 [numExperts]
    DeviceView outIndices;      ///< u32  [tokens][topK]
    DeviceView outWeights;      ///< fp32 [tokens][topK]
    u32 tokens = 1;
    u32 numExperts = 0;
    u32 topK = 0;
};

/// output[i] = input[rows[i]], over `width` elements per row.
///
/// Prefill runs one expert GEMM over all the tokens that selected that expert,
/// rather than one GEMV per (token, expert). Gathering those tokens into a
/// contiguous block is what makes the batching possible - it is the difference
/// between reading an expert's weights once per chunk and once per token.
struct GatherRowsArgs {
    DeviceView input;    ///< fp32 [any][width]
    DeviceView rows;     ///< u32  [count]
    DeviceView output;   ///< fp32 [count][width]
    u32 count = 0;
    u32 width = 0;
};

/// output[rows[i]] += scales[i] * input[i]
///
/// The gather's inverse, folding in the router weight. A token selects a given
/// expert at most once, so within one launch no two source rows share a
/// destination and the accumulation needs no atomics.
struct ScatterAddRowsArgs {
    DeviceView input;    ///< fp32 [count][width]
    DeviceView rows;     ///< u32  [count]
    DeviceView scales;   ///< fp32 [count]
    DeviceView output;   ///< fp32 [any][width]
    u32 count = 0;
    u32 width = 0;
};

/// Zeroes `count` floats. The routed branch is accumulated into, so it starts
/// empty rather than being overwritten.
struct FillZeroArgs {
    DeviceView output;
    u64 count = 0;
};

/// out[d] = sum_k weights[k] * expertOutputs[k][d]
struct MoeCombineArgs {
    DeviceView expertOutputs;  ///< fp32 [topK][hidden]
    DeviceView weights;        ///< fp32 [topK]
    DeviceView output;         ///< fp32 [hidden]
    u32 topK = 0;
    u32 hidden = 0;
};

class IKernels {
public:
    virtual ~IKernels() = default;

    [[nodiscard]] virtual Status rmsNorm(Stream& stream, const RmsNormArgs& args) = 0;
    [[nodiscard]] virtual Status dequantGemv(Stream& stream, const DequantGemvArgs& args) = 0;
    [[nodiscard]] virtual Status dequantGemm(Stream& stream, const DequantGemmArgs& args) = 0;

    /// Token granularity the batched GEMM works in. A prefill chunk that is a
    /// multiple of this launches no partly-idle blocks.
    [[nodiscard]] virtual u32 gemmTokenTile() const = 0;

    [[nodiscard]] virtual Status embedLookup(Stream& stream, const EmbedLookupArgs& args) = 0;
    [[nodiscard]] virtual Status rope(Stream& stream, const RopeArgs& args) = 0;
    [[nodiscard]] virtual Status geglu(Stream& stream, const GegluArgs& args) = 0;
    [[nodiscard]] virtual Status add(Stream& stream, const AddArgs& args) = 0;
    [[nodiscard]] virtual Status scale(Stream& stream, const ScaleArgs& args) = 0;
    [[nodiscard]] virtual Status logitSoftcap(Stream& stream,
                                              const LogitSoftcapArgs& args) = 0;
    [[nodiscard]] virtual Status argmax(Stream& stream, const ArgmaxArgs& args) = 0;

    [[nodiscard]] virtual Status kvWrite(Stream& stream, const KvWriteArgs& args) = 0;
    [[nodiscard]] virtual Status attention(Stream& stream, const AttentionArgs& args) = 0;
    [[nodiscard]] virtual Status routerTopK(Stream& stream, const RouterTopKArgs& args) = 0;
    [[nodiscard]] virtual Status gatherRows(Stream& stream, const GatherRowsArgs& args) = 0;
    [[nodiscard]] virtual Status scatterAddRows(Stream& stream,
                                                const ScatterAddRowsArgs& args) = 0;
    [[nodiscard]] virtual Status fillZero(Stream& stream, const FillZeroArgs& args) = 0;
    [[nodiscard]] virtual Status moeCombine(Stream& stream, const MoeCombineArgs& args) = 0;
};

}  // namespace tf::gpu
