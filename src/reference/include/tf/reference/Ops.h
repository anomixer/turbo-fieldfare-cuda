#pragma once

#include <span>
#include <vector>

#include "tf/core/base/Error.h"
#include "tf/core/base/Types.h"
#include "tf/core/format/Quantization.h"
#include "tf/core/math/Float.h"

/// Scalar CPU reference implementations of every operation the CUDA kernels
/// will perform.
///
/// These are written for clarity and exactness, not speed: they run in f64 or
/// f32 with no fast-math, no blocking and no vectorization, so that when a
/// kernel disagrees the reference is not the suspect. Semantics follow
/// docs/MODEL_SEMANTICS.md, which was derived from the MLX implementation of
/// the exact pinned checkpoint.
namespace tf::reference {

// ---------------------------------------------------------------------------
// Quantization
// ---------------------------------------------------------------------------

/// Unpacks one MLX affine quantized value.
///
/// `packed` holds 32/bits values per word, low-order first. Dequantization is
/// affine: `value = q * scale + bias`, per group of `groupSize` contiguous
/// inputs. The bias is additive, not a zero point to subtract.
[[nodiscard]] float dequantizeElement(std::span<const u32> packed,
                                      std::span<const bf16> scales,
                                      std::span<const bf16> biases, u64 row, u64 column,
                                      u64 inFeatures, QuantSpec spec);

/// Dequantizes a whole [outFeatures, inFeatures] matrix to row-major floats.
/// Intended for tests and debugging, not for any hot path.
[[nodiscard]] std::vector<float> dequantizeMatrix(std::span<const u32> packed,
                                                  std::span<const bf16> scales,
                                                  std::span<const bf16> biases,
                                                  const QuantizedLinearLayout& layout);

/// Quantizes a matrix to MLX affine form, choosing per-group scale and bias
/// from the group's min and max. Used to build test fixtures whose exact
/// dequantized values are known.
struct QuantizedMatrix {
    std::vector<u32> packed;
    std::vector<bf16> scales;
    std::vector<bf16> biases;
};

[[nodiscard]] QuantizedMatrix quantizeMatrix(std::span<const float> values,
                                             const QuantizedLinearLayout& layout);

/// out[o] = sum_i dequantize(W[o][i]) * x[i]
[[nodiscard]] std::vector<float> dequantGemv(std::span<const u32> packed,
                                             std::span<const bf16> scales,
                                             std::span<const bf16> biases,
                                             std::span<const float> x,
                                             const QuantizedLinearLayout& layout);

/// Row lookup from a quantized embedding table, scaled by sqrt(hiddenSize).
[[nodiscard]] std::vector<float> embedLookup(std::span<const u32> packed,
                                             std::span<const bf16> scales,
                                             std::span<const bf16> biases, u32 tokenId,
                                             const QuantizedLinearLayout& layout);

// ---------------------------------------------------------------------------
// Normalization
// ---------------------------------------------------------------------------

/// Gemma 4 RMSNorm: `x * rsqrt(mean(x^2) + eps) * weight`.
///
/// Plain `weight`, NOT `1 + weight`. Gemma 1/2/3 use the unit-offset form and
/// Gemma 4 does not; see docs/MODEL_SEMANTICS.md.
[[nodiscard]] std::vector<float> rmsNorm(std::span<const float> x,
                                         std::span<const float> weight, double eps);

/// RMS normalization with no learnable scale, used for the V projection.
/// Distinct from a weight that happens to be all ones only in that no weight
/// tensor exists.
[[nodiscard]] std::vector<float> rmsNormNoScale(std::span<const float> x, double eps);

// ---------------------------------------------------------------------------
// Rotary embeddings
// ---------------------------------------------------------------------------

/// Rotary embedding parameters.
///
/// Partial rotation is subtler than it looks. MLX's ProportionalRoPE builds a
/// frequency table of length headDim/2 whose leading rotatedDims/2 entries are
/// real and whose remainder are infinite, then rotates over the *whole* head.
/// So the pairing and the exponent denominator both stay tied to headDim, and
/// only the number of pairs that actually turn is reduced:
///
///   pair j joins dimensions j and j + headDim/2
///   pair j rotates only while j < rotatedDims/2
///   angle = position / theta^(2j / headDim)
///
/// Deriving the exponent from rotatedDims instead, or pairing across
/// rotatedDims/2, gives a different rotation entirely - and one that still
/// looks plausible, because it is a valid rotation of the wrong axes.
struct RopeParams {
    u64 headDim = 0;
    double theta = 10000.0;
    /// Fraction of headDim that rotates. 1.0 rotates every pair; the
    /// full-attention layers use 0.25.
    double partialRotaryFactor = 1.0;

    [[nodiscard]] u64 rotatedDims() const {
        // Must stay even: the rotation works on pairs.
        const auto dims = static_cast<u64>(static_cast<double>(headDim) * partialRotaryFactor);
        return dims - (dims % 2);
    }

    /// Pairs that actually rotate.
    [[nodiscard]] u64 rotatedPairs() const { return rotatedDims() / 2; }
};

/// Applies RoPE in place over one head of `headDim` values at `position`, using
/// the non-interleaved convention described above.
void applyRope(std::span<float> head, u64 position, const RopeParams& params);

// ---------------------------------------------------------------------------
// Activations
// ---------------------------------------------------------------------------

/// tanh-approximation GELU, i.e. `gelu_pytorch_tanh`.
[[nodiscard]] float geluApprox(float x);

/// `gelu_approx(gate) * up`, the GeGLU used by both the dense MLP and every
/// routed expert. The gate is the projection that passes through GELU.
[[nodiscard]] std::vector<float> geglu(std::span<const float> gate,
                                       std::span<const float> up);

// ---------------------------------------------------------------------------
// Attention
// ---------------------------------------------------------------------------

struct AttentionParams {
    u64 numHeads = 0;
    u64 numKVHeads = 0;
    u64 headDim = 0;
    /// Gemma 4 sets this to 1.0: q_norm and k_norm already normalize, so the
    /// usual 1/sqrt(headDim) is absent.
    float scale = 1.0f;
    /// 0 disables the window, giving full causal attention.
    u64 slidingWindow = 0;

    /// Rows per KV head in the cache. 0 means the cache is exactly as long as
    /// the history, which is the simple linear case.
    u64 capacity = 0;
    /// True for the 25 sliding-window layers, whose cache is a ring: logical
    /// position p lives at row p % capacity, and rows are overwritten once the
    /// history exceeds the ring. False for the 5 full-attention layers, which
    /// append linearly.
    bool circular = false;

    /// Rows allocated per KV head.
    [[nodiscard]] u64 rowsPerHead(u64 cachedLength) const {
        return capacity != 0 ? capacity : cachedLength;
    }

    /// Cache row holding logical position `position`.
    [[nodiscard]] u64 slotFor(u64 position, u64 cachedLength) const {
        const u64 rows = rowsPerHead(cachedLength);
        return circular && rows != 0 ? position % rows : position;
    }
};

/// Single-query (decode step) attention over a KV cache.
///
/// `keys` and `values` are [numKVHeads][rowsPerHead][headDim] row-major.
/// `queries` is [numHeads][headDim]. Returns [numHeads][headDim].
/// `queryPosition` is the absolute position of the query token, which the
/// sliding window is measured against.
[[nodiscard]] std::vector<float> decodeAttention(std::span<const float> queries,
                                                 std::span<const float> keys,
                                                 std::span<const float> values,
                                                 u64 cachedLength, u64 queryPosition,
                                                 const AttentionParams& params);

/// Weighted sum over the selected experts: out[d] = sum_k weights[k] * out[k][d].
///
/// `expertOutputs` is [topK][hidden] row-major.
[[nodiscard]] std::vector<float> moeCombine(std::span<const float> expertOutputs,
                                            std::span<const float> weights, u64 topK,
                                            u64 hidden);

/// Numerically stable softmax over one vector.
[[nodiscard]] std::vector<float> softmax(std::span<const float> logits);

// ---------------------------------------------------------------------------
// Mixture of experts
// ---------------------------------------------------------------------------

struct RouterResult {
    std::vector<u32> indices;
    std::vector<float> weights;
};

/// Selects the top-k experts by score, then softmaxes over only those k and
/// multiplies by the per-expert scale.
///
/// Selection is by raw score; the softmax happens after selection, not before.
[[nodiscard]] RouterResult routerTopK(std::span<const float> scores,
                                      std::span<const float> perExpertScale, u64 topK);

/// The per-layer router norm weight, `router.scale * hiddenSize^-0.5`, folded
/// once at load rather than recomputed per token.
[[nodiscard]] std::vector<float> foldRouterScale(std::span<const float> routerScale,
                                                 u64 hiddenSize);

// ---------------------------------------------------------------------------
// Output head
// ---------------------------------------------------------------------------

/// `tanh(x / cap) * cap`, applied to the final logits.
[[nodiscard]] std::vector<float> logitSoftcap(std::span<const float> logits, float cap);

/// Index of the largest value, ties resolved to the lowest index so that greedy
/// decoding is deterministic.
[[nodiscard]] u32 argmax(std::span<const float> values);

}  // namespace tf::reference
