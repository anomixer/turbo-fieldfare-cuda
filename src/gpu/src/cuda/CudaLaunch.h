#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "tf/core/base/Types.h"

/// Launch entry points implemented in CudaKernels.cu.
///
/// The boundary exists because nvcc compiles at C++20 while the rest of the
/// project is C++23: std::expected, and therefore tf::Result, cannot appear in
/// a .cu translation unit. So .cu files hold kernels and these plain launchers,
/// and every argument check, error message and Result lives in
/// CudaKernelDispatch.cpp on the C++23 side.
///
/// Storage types, and why they differ:
///
///   activations  float  - Gemma 4's activations do not fit in fp16. The
///                         checkpoint is bfloat16, whose exponent range matches
///                         fp32; fp16 saturates at 65504. Layer 5's expert
///                         input reaches a peak of 410 and its projections
///                         around 360, so the GeGLU product overflows. See
///                         docs/MODEL_SEMANTICS.md.
///   KV cache     __half - keys and values are post-normalization and bounded
///                         near unit RMS, so fp16 is safe here and halves the
///                         largest runtime allocation.
///   weights      u32 + __nv_bfloat16 scales and biases, as stored.
///
/// Each function returns the result of cudaGetLastError() immediately after the
/// launch, which catches a bad configuration. Execution errors surface later,
/// at the next synchronization.
namespace tf::gpu::launch {

cudaError_t rmsNorm(cudaStream_t stream, const float* input, const __nv_bfloat16* weight,
                    float* output, u32 rows, u32 count, float eps);

cudaError_t dequantGemv(cudaStream_t stream, u32 bits, const u32* packed,
                        const __nv_bfloat16* scales, const __nv_bfloat16* biases,
                        const float* input, float* output, u32 outFeatures, u32 inFeatures,
                        u32 groupSize);

/// Batched form of the above: `input` and `output` are [tokens][features] row
/// major, so one token is the layout the decode path already uses.
cudaError_t dequantGemm(cudaStream_t stream, u32 bits, const u32* packed,
                        const __nv_bfloat16* scales, const __nv_bfloat16* biases,
                        const float* input, float* output, u32 tokens, u32 outFeatures,
                        u32 inFeatures, u32 groupSize);

/// Token count the GEMM blocks over. Exposed so the prefill chunk size can be
/// chosen as a multiple of it and avoid launching mostly-idle blocks.
u32 dequantGemmTokenTile();

cudaError_t embedLookup(cudaStream_t stream, u32 bits, const u32* packed,
                        const __nv_bfloat16* scales, const __nv_bfloat16* biases,
                        float* output, u32 tokenId, u32 hiddenSize, u32 groupSize,
                        float embedScale);

/// Token `t` of the batch rotates at position `basePosition + t`.
cudaError_t rope(cudaStream_t stream, float* data, u32 tokens, u32 heads, u32 headDim,
                 u32 rotated, u64 basePosition, float theta);

cudaError_t geglu(cudaStream_t stream, const float* gate, const float* up, float* output,
                  u32 count);

cudaError_t add(cudaStream_t stream, const float* a, const float* b, float* output,
                u32 count);

cudaError_t scale(cudaStream_t stream, const float* input, float* output, u32 count,
                  float scalar);

cudaError_t logitSoftcap(cudaStream_t stream, const float* input, float* output, u32 count,
                         float cap);

cudaError_t argmax(cudaStream_t stream, const float* input, u32* output, u32 count);

cudaError_t kvWrite(cudaStream_t stream, const float* key, const float* value,
                    __half* keyCache, __half* valueCache, u32 tokens, u32 kvHeads, u32 headDim,
                    u32 capacity, u64 basePosition, bool circular);

/// Causal attention for `tokens` queries at positions `basePosition + t`. Each
/// query masks itself, so a prefill chunk needs no explicit mask tensor and a
/// decode step is the one-token case.
cudaError_t attention(cudaStream_t stream, const float* queries, const __half* keyCache,
                      const __half* valueCache, float* output, u32 tokens, u32 numHeads,
                      u32 kvHeads, u32 headDim, u32 capacity, u64 basePosition,
                      u32 slidingWindow, bool circular, float scale);

cudaError_t routerTopK(cudaStream_t stream, const float* scores,
                       const __nv_bfloat16* perExpertScale, u32* outIndices,
                       float* outWeights, u32 tokens, u32 numExperts, u32 topK);

/// output[i] = input[rows[i]], and its weighted inverse. Together these let the
/// prefill MoE run one GEMM per expert over the tokens that chose it.
cudaError_t gatherRows(cudaStream_t stream, const float* input, const u32* rows, float* output,
                       u32 count, u32 width);

cudaError_t scatterAddRows(cudaStream_t stream, const float* input, const u32* rows,
                           const float* scales, float* output, u32 count, u32 width);

cudaError_t fillZero(cudaStream_t stream, float* data, u64 count);

cudaError_t moeCombine(cudaStream_t stream, const float* expertOutputs, const float* weights,
                       float* output, u32 topK, u32 hidden);

}  // namespace tf::gpu::launch
