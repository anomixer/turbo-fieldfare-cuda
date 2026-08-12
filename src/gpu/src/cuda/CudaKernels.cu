// CUDA kernels for the decode path.
//
// Translated from upstream's Metal shaders. The correspondence is mechanical:
//
//   simd_sum(x)                    -> warpReduceSum via __shfl_xor_sync
//   simdgroup_index_in_threadgroup -> threadIdx.x / 32
//   thread_index_in_simdgroup      -> threadIdx.x % 32
//   threadgroup T[]                -> __shared__ T[]
//   threadgroup_barrier            -> __syncthreads()
//   function constants             -> template parameters
//
// A CUDA warp is always 32 lanes, exactly what Metal's simdgroup assumes, so
// the reductions port without a width parameter.
//
// This translation unit contains kernels and their launchers only. Everything
// that needs C++23 - argument validation, error messages, Result - lives in
// CudaKernelDispatch.cpp, because nvcc compiles at C++20 and std::expected is
// not available to it.
//
// Every kernel is checked against the scalar reference in src/reference by
// tests/gpu/KernelTests.cpp. Correctness first; M13 is where they get tuned.

#include <math_constants.h>

#include "CudaLaunch.h"

namespace tf::gpu {
namespace {

constexpr u32 kWarpSize = 32;
constexpr u32 kBlock = 256;

[[nodiscard]] u32 blocksFor(u32 count, u32 blockSize) {
    return (count + blockSize - 1) / blockSize;
}

// ---------------------------------------------------------------------------
// Reductions
// ---------------------------------------------------------------------------

/// Sum across a warp. Every lane ends up holding the total.
__device__ __forceinline__ float warpReduceSum(float value) {
#pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        value += __shfl_xor_sync(0xFFFFFFFFu, value, offset);
    }
    return value;
}

/// Sum across a block: one warp reduction, then a second over the per-warp
/// partials. Every thread ends up holding the total.
template <u32 kBlockSize>
__device__ __forceinline__ float blockReduceSum(float value) {
    static_assert(kBlockSize % kWarpSize == 0);
    constexpr u32 kWarps = kBlockSize / kWarpSize;

    __shared__ float partials[kWarps];

    const u32 lane = threadIdx.x % kWarpSize;
    const u32 warp = threadIdx.x / kWarpSize;

    value = warpReduceSum(value);
    if (lane == 0) {
        partials[warp] = value;
    }
    __syncthreads();

    if (warp == 0) {
        float total = (threadIdx.x < kWarps) ? partials[threadIdx.x] : 0.0f;
        total = warpReduceSum(total);
        if (lane == 0) {
            partials[0] = total;
        }
    }
    __syncthreads();
    return partials[0];
}

// ---------------------------------------------------------------------------
// RMSNorm
// ---------------------------------------------------------------------------

/// One block per row. `kHasWeight` false is the no-scale form used by v_norm,
/// where no weight tensor exists at all.
template <u32 kBlockSize, bool kHasWeight>
__global__ void rmsNormKernel(const float* __restrict__ input,
                              const __nv_bfloat16* __restrict__ weight,
                              float* __restrict__ output, u32 count, float eps) {
    const u32 row = blockIdx.x;
    const float* rowIn = input + static_cast<size_t>(row) * count;
    float* rowOut = output + static_cast<size_t>(row) * count;

    float sumSquares = 0.0f;
    for (u32 i = threadIdx.x; i < count; i += kBlockSize) {
        const float value = rowIn[i];
        sumSquares += value * value;
    }
    sumSquares = blockReduceSum<kBlockSize>(sumSquares);

    const float inverseRms = rsqrtf(sumSquares / static_cast<float>(count) + eps);

    for (u32 i = threadIdx.x; i < count; i += kBlockSize) {
        float value = rowIn[i] * inverseRms;
        if constexpr (kHasWeight) {
            // Gemma 4 multiplies by the stored weight directly. Gemma 1/2/3
            // used (1 + weight); applying that here would inflate every norm.
            value *= __bfloat162float(weight[i]);
        }
        rowOut[i] = value;
    }
}

// ---------------------------------------------------------------------------
// Elementwise
// ---------------------------------------------------------------------------

__global__ void gegluKernel(const float* __restrict__ gate, const float* __restrict__ up,
                            float* __restrict__ output, u32 count) {
    const u32 index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    const float g = gate[index];
    const float u = up[index];

    // gelu_pytorch_tanh. tanhf, not the fast intrinsic, so this stays within
    // tolerance of the CPU reference.
    constexpr float kSqrt2OverPi = 0.7978845608028654f;
    constexpr float kCoefficient = 0.044715f;
    const float inner = kSqrt2OverPi * (g + kCoefficient * g * g * g);
    const float activated = 0.5f * g * (1.0f + tanhf(inner));

    output[index] = activated * u;
}

__global__ void addKernel(const float* __restrict__ a, const float* __restrict__ b,
                          float* __restrict__ output, u32 count) {
    const u32 index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        output[index] = a[index] + b[index];
    }
}

__global__ void scaleKernel(const float* __restrict__ input, float* __restrict__ output,
                            u32 count, float scalar) {
    const u32 index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        output[index] = input[index] * scalar;
    }
}

__global__ void logitSoftcapKernel(const float* __restrict__ input,
                                   float* __restrict__ output, u32 count, float cap) {
    const u32 index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        output[index] = tanhf(input[index] / cap) * cap;
    }
}

// ---------------------------------------------------------------------------
// Argmax
// ---------------------------------------------------------------------------

/// One block striding the whole vector. The vocabulary is 262144, so a block of
/// 256 makes 1024 passes - cheap beside the head GEMV that produced the logits,
/// and it avoids a second launch to combine partial results.
template <u32 kBlockSize>
__global__ void argmaxKernel(const float* __restrict__ input, u32* __restrict__ output,
                             u32 count) {
    __shared__ float bestValues[kBlockSize];
    __shared__ u32 bestIndices[kBlockSize];

    float bestValue = -CUDART_INF_F;
    u32 bestIndex = 0;

    for (u32 i = threadIdx.x; i < count; i += kBlockSize) {
        const float value = input[i];
        // Strictly greater keeps the lowest index on a tie, which is what makes
        // greedy decoding reproducible.
        if (value > bestValue) {
            bestValue = value;
            bestIndex = i;
        }
    }

    bestValues[threadIdx.x] = bestValue;
    bestIndices[threadIdx.x] = bestIndex;
    __syncthreads();

    for (u32 stride = kBlockSize / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            const u32 other = threadIdx.x + stride;
            const bool takeOther = bestValues[other] > bestValues[threadIdx.x] ||
                                   (bestValues[other] == bestValues[threadIdx.x] &&
                                    bestIndices[other] < bestIndices[threadIdx.x]);
            if (takeOther) {
                bestValues[threadIdx.x] = bestValues[other];
                bestIndices[threadIdx.x] = bestIndices[other];
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        *output = bestIndices[0];
    }
}

// ---------------------------------------------------------------------------
// RoPE
// ---------------------------------------------------------------------------

/// One block per (head, token); each thread rotates one dimension pair.
///
/// Token `t` of the batch sits at position `basePosition + t`, which is the
/// only thing that differs between a decode step and a prefill chunk - decode
/// is simply the one-token case.
///
/// Non-interleaved, and partial rotation does NOT shrink the pairing: pair j
/// always joins dimensions j and j + headDim/2, the exponent always divides by
/// headDim, and only the count of rotating pairs is reduced. That is what MLX's
/// ProportionalRoPE does - it builds a headDim/2 frequency table whose tail is
/// infinite and rotates over the whole head. Pairing across rotated/2 instead
/// would be a valid rotation of the wrong axes.
__global__ void ropeKernel(float* __restrict__ data, u32 heads, u32 headDim, u32 rotated,
                           u64 basePosition, float theta) {
    const u32 head = blockIdx.x;
    const u32 token = blockIdx.y;
    const u64 position = basePosition + token;
    const u32 half = headDim / 2;
    const u32 rotatedPairs = rotated / 2;

    float* headData =
            data + (static_cast<size_t>(token) * heads + head) * headDim;

    for (u32 i = threadIdx.x; i < rotatedPairs; i += blockDim.x) {
        const float exponent = static_cast<float>(2 * i) / static_cast<float>(headDim);
        // powf and sincosf rather than the fast intrinsics: position can reach
        // 262144, where __sincosf loses far too much of the angle.
        const float frequency = powf(theta, -exponent);
        const float angle = static_cast<float>(position) * frequency;

        float sine = 0.0f;
        float cosine = 0.0f;
        sincosf(angle, &sine, &cosine);

        const float lower = headData[i];
        const float upper = headData[i + half];

        headData[i] = lower * cosine - upper * sine;
        headData[i + half] = lower * sine + upper * cosine;
    }
}

// ---------------------------------------------------------------------------
// Quantized GEMV
// ---------------------------------------------------------------------------

/// Dequantizes and accumulates one output row per warp.
///
/// MLX affine layout: `kBits` values packed per u32 low-order first, one bf16
/// scale and bias per group of `groupSize` inputs. Dequantization is
/// `q * scale + bias` - affine, not a zero point to subtract. Scales may be
/// negative.
///
/// Lanes stride over whole packed words, so a scale/bias pair is loaded once
/// per word rather than once per value.
template <u32 kBits, u32 kWarpsPerBlock>
__global__ void dequantGemvKernel(const u32* __restrict__ packed,
                                  const __nv_bfloat16* __restrict__ scales,
                                  const __nv_bfloat16* __restrict__ biases,
                                  const float* __restrict__ input,
                                  float* __restrict__ output, u32 outFeatures,
                                  u32 inFeatures, u32 groupSize) {
    constexpr u32 kValuesPerWord = 32 / kBits;
    constexpr u32 kMask = (1u << kBits) - 1u;

    const u32 warp = threadIdx.x / kWarpSize;
    const u32 lane = threadIdx.x % kWarpSize;
    const u32 row = blockIdx.x * kWarpsPerBlock + warp;
    if (row >= outFeatures) {
        return;
    }

    const u32 wordsPerRow = inFeatures / kValuesPerWord;
    const u32 groupsPerRow = inFeatures / groupSize;
    const u32 wordsPerGroup = groupSize / kValuesPerWord;

    const u32* rowPacked = packed + static_cast<size_t>(row) * wordsPerRow;
    const __nv_bfloat16* rowScales = scales + static_cast<size_t>(row) * groupsPerRow;
    const __nv_bfloat16* rowBiases = biases + static_cast<size_t>(row) * groupsPerRow;

    float accumulator = 0.0f;

    for (u32 word = lane; word < wordsPerRow; word += kWarpSize) {
        const u32 group = word / wordsPerGroup;
        const float scale = __bfloat162float(rowScales[group]);
        const float bias = __bfloat162float(rowBiases[group]);

        const u32 bits = rowPacked[word];
        const u32 base = word * kValuesPerWord;

#pragma unroll
        for (u32 slot = 0; slot < kValuesPerWord; ++slot) {
            const u32 quantized = (bits >> (slot * kBits)) & kMask;
            const float weight = static_cast<float>(quantized) * scale + bias;
            accumulator += weight * input[base + slot];
        }
    }

    accumulator = warpReduceSum(accumulator);
    if (lane == 0) {
        output[row] = accumulator;
    }
}

// ---------------------------------------------------------------------------
// Quantized GEMM (batched GEMV)
// ---------------------------------------------------------------------------

/// Tokens processed per block. Every one costs an accumulator register per
/// thread and a row of the shared activation tile, so this trades registers and
/// shared memory for weight reuse.
constexpr u32 kTokenTile = 16;

/// Output rows each warp owns.
///
/// The first version of this kernel used one, and measured neither
/// bandwidth-bound nor compute-bound: 2.8 TFLOP/s against a card that does 24,
/// and 4.3 GB/s against 448. What limits it is shared-memory reads. Per packed
/// word a lane issues kTokenTile activation reads and does kTokenTile *
/// kValuesPerWord multiply-adds, a ratio of four FMAs per LDS.128. Giving each
/// lane a second row reuses the same activation read for twice the arithmetic
/// without touching the shared traffic at all.
///
/// Measured at 128 tokens over a 2816x2816 projection, as a multiple of the
/// GEMV launches it replaces: 1 row 3.4x, 2 rows 5.4x, 4 rows 10.8x, 8 rows
/// 10.9x. Eight buys three percent for twice the accumulator registers and more
/// waste on a narrow matrix like the router, so four is the setting.
constexpr u32 kRowsPerWarp = 4;

/// Input features staged per iteration. 256 is one packed word per lane at
/// 4 bits, and 16 KiB of shared memory at the tile width above.
constexpr u32 kFeatureChunk = 256;

/// out[t][o] = sum_i dequantize(W[o][i]) * x[t][i], for `tokens` tokens.
///
/// The point of this kernel over calling the GEMV once per token: prefill is
/// bound by weight bandwidth, not arithmetic. A GEMV reads the whole matrix to
/// produce one output column, so N tokens read it N times. Here each lane loads
/// a packed word once, dequantizes it once, and feeds it to kTokenTile tokens -
/// the same DRAM traffic does kTokenTile times the work.
///
/// The activation tile is staged token-major so that the kValuesPerWord
/// features belonging to one lane's word are contiguous, and can be read as
/// float4. Feature-major would put those features `kTokenTile` apart and, since
/// the resulting bank stride is a multiple of eight, collapse onto four banks.
///
/// Each lane also carries kRowsPerWarp output rows, so one activation read
/// feeds that many rows of arithmetic. Registers are the limit: the
/// accumulators alone are kRowsPerWarp * kTokenTile floats.
template <u32 kBits, u32 kWarpsPerBlock, u32 kRows>
__global__ void dequantGemmKernel(const u32* __restrict__ packed,
                                  const __nv_bfloat16* __restrict__ scales,
                                  const __nv_bfloat16* __restrict__ biases,
                                  const float* __restrict__ input,
                                  float* __restrict__ output, u32 tokens, u32 outFeatures,
                                  u32 inFeatures, u32 groupSize) {
    constexpr u32 kValuesPerWord = 32 / kBits;
    constexpr u32 kMask = (1u << kBits) - 1u;
    constexpr u32 kVectorsPerWord = kValuesPerWord / 4;

    __shared__ float tile[kTokenTile][kFeatureChunk];

    const u32 warp = threadIdx.x / kWarpSize;
    const u32 lane = threadIdx.x % kWarpSize;
    const u32 firstRow = (blockIdx.x * kWarpsPerBlock + warp) * kRows;
    const u32 tokenBase = blockIdx.y * kTokenTile;

    const u32 wordsPerRow = inFeatures / kValuesPerWord;
    const u32 groupsPerRow = inFeatures / groupSize;
    const u32 wordsPerGroup = groupSize / kValuesPerWord;

    // Rows past the end still cooperate on the shared load and the barriers;
    // only their arithmetic and stores are skipped.
    bool active[kRows];
    const u32* rowPacked[kRows];
    const __nv_bfloat16* rowScales[kRows];
    const __nv_bfloat16* rowBiases[kRows];
#pragma unroll
    for (u32 r = 0; r < kRows; ++r) {
        const u32 row = firstRow + r;
        active[r] = row < outFeatures;
        const u32 safeRow = active[r] ? row : 0;
        rowPacked[r] = packed + static_cast<size_t>(safeRow) * wordsPerRow;
        rowScales[r] = scales + static_cast<size_t>(safeRow) * groupsPerRow;
        rowBiases[r] = biases + static_cast<size_t>(safeRow) * groupsPerRow;
    }
    const bool anyActive = active[0];

    float accumulator[kRows][kTokenTile];
#pragma unroll
    for (u32 r = 0; r < kRows; ++r) {
#pragma unroll
        for (u32 t = 0; t < kTokenTile; ++t) {
            accumulator[r][t] = 0.0f;
        }
    }

    constexpr u32 kTileElements = kTokenTile * kFeatureChunk;
    const u32 threads = kWarpsPerBlock * kWarpSize;

    for (u32 chunk = 0; chunk < inFeatures; chunk += kFeatureChunk) {
        __syncthreads();

        // Feature index varies fastest across threads, so global reads coalesce
        // and shared writes land in consecutive banks.
        for (u32 index = threadIdx.x; index < kTileElements; index += threads) {
            const u32 t = index / kFeatureChunk;
            const u32 feature = index % kFeatureChunk;
            const u32 token = tokenBase + t;
            const u32 column = chunk + feature;
            tile[t][feature] = (token < tokens && column < inFeatures)
                                       ? input[static_cast<size_t>(token) * inFeatures + column]
                                       : 0.0f;
        }

        __syncthreads();

        if (!anyActive) {
            continue;
        }

        const u32 wordBase = chunk / kValuesPerWord;
        constexpr u32 kWordsPerChunk = kFeatureChunk / kValuesPerWord;

        for (u32 word = lane; word < kWordsPerChunk; word += kWarpSize) {
            const u32 globalWord = wordBase + word;
            if (globalWord >= wordsPerRow) {
                break;
            }
            const u32 group = globalWord / wordsPerGroup;

            float weight[kRows][kValuesPerWord];
#pragma unroll
            for (u32 r = 0; r < kRows; ++r) {
                const float scale = __bfloat162float(rowScales[r][group]);
                const float bias = __bfloat162float(rowBiases[r][group]);
                const u32 bits = rowPacked[r][globalWord];
#pragma unroll
                for (u32 slot = 0; slot < kValuesPerWord; ++slot) {
                    const u32 quantized = (bits >> (slot * kBits)) & kMask;
                    weight[r][slot] = static_cast<float>(quantized) * scale + bias;
                }
            }

            const u32 feature = word * kValuesPerWord;
#pragma unroll
            for (u32 t = 0; t < kTokenTile; ++t) {
                // Read once, use for every row this lane owns. That reuse is
                // the whole point of kRows.
                float activation[kValuesPerWord];
                const float4* vectors = reinterpret_cast<const float4*>(&tile[t][feature]);
#pragma unroll
                for (u32 vector = 0; vector < kVectorsPerWord; ++vector) {
                    const float4 value = vectors[vector];
                    activation[vector * 4 + 0] = value.x;
                    activation[vector * 4 + 1] = value.y;
                    activation[vector * 4 + 2] = value.z;
                    activation[vector * 4 + 3] = value.w;
                }
#pragma unroll
                for (u32 r = 0; r < kRows; ++r) {
#pragma unroll
                    for (u32 slot = 0; slot < kValuesPerWord; ++slot) {
                        accumulator[r][t] += weight[r][slot] * activation[slot];
                    }
                }
            }
        }
    }

#pragma unroll
    for (u32 r = 0; r < kRows; ++r) {
        if (!active[r]) {
            continue;
        }
#pragma unroll
        for (u32 t = 0; t < kTokenTile; ++t) {
            const float total = warpReduceSum(accumulator[r][t]);
            const u32 token = tokenBase + t;
            if (lane == 0 && token < tokens) {
                output[static_cast<size_t>(token) * outFeatures + firstRow + r] = total;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Embedding lookup
// ---------------------------------------------------------------------------

/// Reads one row of the quantized embedding table and applies Gemma's
/// sqrt(hiddenSize) input scale.
template <u32 kBits>
__global__ void embedLookupKernel(const u32* __restrict__ packed,
                                  const __nv_bfloat16* __restrict__ scales,
                                  const __nv_bfloat16* __restrict__ biases,
                                  float* __restrict__ output, u32 tokenId, u32 hiddenSize,
                                  u32 groupSize, float embedScale) {
    constexpr u32 kValuesPerWord = 32 / kBits;
    constexpr u32 kMask = (1u << kBits) - 1u;

    const u32 wordsPerRow = hiddenSize / kValuesPerWord;
    const u32 groupsPerRow = hiddenSize / groupSize;

    const u32* rowPacked = packed + static_cast<size_t>(tokenId) * wordsPerRow;
    const __nv_bfloat16* rowScales = scales + static_cast<size_t>(tokenId) * groupsPerRow;
    const __nv_bfloat16* rowBiases = biases + static_cast<size_t>(tokenId) * groupsPerRow;

    for (u32 index = blockIdx.x * blockDim.x + threadIdx.x; index < hiddenSize;
         index += gridDim.x * blockDim.x) {
        const u32 word = index / kValuesPerWord;
        const u32 slot = index % kValuesPerWord;
        const u32 group = index / groupSize;

        const u32 quantized = (rowPacked[word] >> (slot * kBits)) & kMask;
        const float value =
                static_cast<float>(quantized) * __bfloat162float(rowScales[group]) +
                __bfloat162float(rowBiases[group]);
        output[index] = value * embedScale;
    }
}

// ---------------------------------------------------------------------------
// KV cache
// ---------------------------------------------------------------------------

/// Maps a logical token position to its cache row. The sliding-window layers
/// wrap; the full-attention layers do not.
__device__ __forceinline__ u32 cacheSlot(u64 position, u32 capacity, bool circular) {
    return circular ? static_cast<u32>(position % capacity) : static_cast<u32>(position);
}

/// One block per (KV head, token), writing that head's K and V.
__global__ void kvWriteKernel(const float* __restrict__ key,
                              const float* __restrict__ value, __half* __restrict__ keyCache,
                              __half* __restrict__ valueCache, u32 kvHeads, u32 headDim,
                              u32 capacity, u64 basePosition, bool circular) {
    const u32 head = blockIdx.x;
    const u32 token = blockIdx.y;
    const u32 slot = cacheSlot(basePosition + token, capacity, circular);

    const size_t sourceBase =
            (static_cast<size_t>(token) * kvHeads + head) * headDim;
    const size_t cacheBase =
            (static_cast<size_t>(head) * capacity + slot) * headDim;

    for (u32 i = threadIdx.x; i < headDim; i += blockDim.x) {
        keyCache[cacheBase + i] = __float2half(key[sourceBase + i]);
        valueCache[cacheBase + i] = __float2half(value[sourceBase + i]);
    }
}

// ---------------------------------------------------------------------------
// Decode attention
// ---------------------------------------------------------------------------

/// One block per (query head, token). Streaming softmax in a single pass over
/// the history: running maximum and running denominator are rescaled as larger
/// scores appear, so the scores never need to be materialized.
///
/// The alternative - store all scores, find the max, then normalize - would
/// need up to 4096 floats of shared memory per head, more than a block has.
/// It would also cost O(tokens x history) memory during prefill, where this
/// costs none.
///
/// Causality is per token rather than per launch: token `t` sits at position
/// `basePosition + t` and attends to positions up to and including its own.
/// That is what lets a whole prefill chunk go through one launch, and it makes
/// a decode step the one-token case of the same kernel rather than a separate
/// one that could drift from it.
template <u32 kBlockSize>
__global__ void attentionKernel(const float* __restrict__ queries,
                                const __half* __restrict__ keyCache,
                                const __half* __restrict__ valueCache,
                                float* __restrict__ output, u32 numHeads, u32 kvHeads,
                                u32 headDim, u32 capacity, u64 basePosition,
                                u32 slidingWindow, bool circular, float scale) {
    extern __shared__ float shared[];
    float* accumulator = shared;              // headDim
    float* queryCache = shared + headDim;     // headDim

    const u32 head = blockIdx.x;
    if (head >= numHeads) {
        return;
    }
    const u32 token = blockIdx.y;
    const u64 queryPosition = basePosition + token;
    const u64 cachedLength = queryPosition + 1;

    // A sliding window attends to the most recent `slidingWindow` positions
    // inclusive of the query. A ring additionally loses anything older than its
    // own capacity, since those rows have been overwritten.
    u64 firstVisible = 0;
    if (slidingWindow > 0 && cachedLength > slidingWindow) {
        firstVisible = cachedLength - slidingWindow;
    }
    if (circular && cachedLength > capacity) {
        const u64 oldest = cachedLength - capacity;
        firstVisible = firstVisible > oldest ? firstVisible : oldest;
    }

    const u32 groupSize = numHeads / kvHeads;
    const u32 kvHead = head / groupSize;

    // The query is read once per position otherwise; staging it in shared
    // memory turns 4096 global reads per element into one.
    const float* query =
            queries + (static_cast<size_t>(token) * numHeads + head) * headDim;
    for (u32 i = threadIdx.x; i < headDim; i += kBlockSize) {
        queryCache[i] = query[i];
        accumulator[i] = 0.0f;
    }
    __syncthreads();

    __shared__ float sharedMax;
    __shared__ float sharedDenominator;
    if (threadIdx.x == 0) {
        sharedMax = -CUDART_INF_F;
        sharedDenominator = 0.0f;
    }
    __syncthreads();

    for (u64 position = firstVisible; position < cachedLength; ++position) {
        const u32 slot = cacheSlot(position, capacity, circular);
        const size_t rowBase =
                (static_cast<size_t>(kvHead) * capacity + slot) * headDim;

        float dot = 0.0f;
        for (u32 i = threadIdx.x; i < headDim; i += kBlockSize) {
            dot += queryCache[i] * __half2float(keyCache[rowBase + i]);
        }
        dot = blockReduceSum<kBlockSize>(dot) * scale;

        // Rescale the running total whenever a new maximum appears, which is
        // what keeps the exponentials in range without a second pass.
        __shared__ float weight;
        __shared__ float rescale;
        if (threadIdx.x == 0) {
            const float newMax = fmaxf(sharedMax, dot);
            rescale = (sharedMax == -CUDART_INF_F) ? 0.0f : __expf(sharedMax - newMax);
            weight = __expf(dot - newMax);
            sharedMax = newMax;
            sharedDenominator = sharedDenominator * rescale + weight;
        }
        __syncthreads();

        for (u32 i = threadIdx.x; i < headDim; i += kBlockSize) {
            accumulator[i] = accumulator[i] * rescale +
                             weight * __half2float(valueCache[rowBase + i]);
        }
        __syncthreads();
    }

    const float denominator = sharedDenominator;
    float* destination =
            output + (static_cast<size_t>(token) * numHeads + head) * headDim;
    for (u32 i = threadIdx.x; i < headDim; i += kBlockSize) {
        // An empty window leaves the denominator at zero; emit zeros rather
        // than NaN.
        destination[i] = denominator > 0.0f ? accumulator[i] / denominator
                                                         : 0.0f;
    }
}

// ---------------------------------------------------------------------------
// Router
// ---------------------------------------------------------------------------

/// One block per token. Extracts the top-k by repeated argmax over the score
/// vector - k is 8 against 128 experts, so eight cheap reductions beat a sort.
///
/// Selection is by raw score; the softmax runs afterwards over just the
/// selected scores, and the per-expert scale multiplies after that.
template <u32 kBlockSize, u32 kMaxTopK>
__global__ void routerTopKKernel(const float* __restrict__ scores,
                                 const __nv_bfloat16* __restrict__ perExpertScale,
                                 u32* __restrict__ outIndices, float* __restrict__ outWeights,
                                 u32 numExperts, u32 topK) {
    extern __shared__ float working[];  // numExperts

    const u32 token = blockIdx.x;
    scores += static_cast<size_t>(token) * numExperts;
    outIndices += static_cast<size_t>(token) * topK;
    outWeights += static_cast<size_t>(token) * topK;

    for (u32 i = threadIdx.x; i < numExperts; i += kBlockSize) {
        working[i] = scores[i];
    }
    __syncthreads();

    __shared__ float selectedScores[kMaxTopK];
    __shared__ u32 selectedIndices[kMaxTopK];
    __shared__ float bestValues[kBlockSize];
    __shared__ u32 bestIndices[kBlockSize];

    for (u32 round = 0; round < topK; ++round) {
        float bestValue = -CUDART_INF_F;
        u32 bestIndex = 0;
        for (u32 i = threadIdx.x; i < numExperts; i += kBlockSize) {
            if (working[i] > bestValue) {
                bestValue = working[i];
                bestIndex = i;
            }
        }
        bestValues[threadIdx.x] = bestValue;
        bestIndices[threadIdx.x] = bestIndex;
        __syncthreads();

        for (u32 stride = kBlockSize / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                const u32 other = threadIdx.x + stride;
                // Ties resolve to the lower index, matching the reference.
                const bool takeOther = bestValues[other] > bestValues[threadIdx.x] ||
                                       (bestValues[other] == bestValues[threadIdx.x] &&
                                        bestIndices[other] < bestIndices[threadIdx.x]);
                if (takeOther) {
                    bestValues[threadIdx.x] = bestValues[other];
                    bestIndices[threadIdx.x] = bestIndices[other];
                }
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            selectedScores[round] = bestValues[0];
            selectedIndices[round] = bestIndices[0];
            // Remove the winner so the next round finds the following expert.
            working[bestIndices[0]] = -CUDART_INF_F;
        }
        __syncthreads();
    }

    // Softmax over the selected scores only, then the per-expert scale. One
    // thread: topK is 8.
    if (threadIdx.x == 0) {
        float maximum = -CUDART_INF_F;
        for (u32 k = 0; k < topK; ++k) {
            maximum = fmaxf(maximum, selectedScores[k]);
        }
        float total = 0.0f;
        for (u32 k = 0; k < topK; ++k) {
            selectedScores[k] = __expf(selectedScores[k] - maximum);
            total += selectedScores[k];
        }
        for (u32 k = 0; k < topK; ++k) {
            const u32 expert = selectedIndices[k];
            const float weight = (selectedScores[k] / total) *
                                 __bfloat162float(perExpertScale[expert]);
            outIndices[k] = expert;
            outWeights[k] = weight;
        }
    }
}

// ---------------------------------------------------------------------------
// Row gather and scatter
// ---------------------------------------------------------------------------

/// output[i] = input[rows[i]], over `width` elements per row.
///
/// The prefill MoE runs one GEMM per expert rather than one per (token,
/// expert): the tokens that selected a given expert are collected into a
/// contiguous block first, so the expert's weights are read once for all of
/// them. This is that collection step.
__global__ void gatherRowsKernel(const float* __restrict__ input,
                                 const u32* __restrict__ rows, float* __restrict__ output,
                                 u32 width) {
    const u32 row = blockIdx.y;
    const float* source = input + static_cast<size_t>(rows[row]) * width;
    float* destination = output + static_cast<size_t>(row) * width;

    for (u32 i = blockIdx.x * blockDim.x + threadIdx.x; i < width;
         i += gridDim.x * blockDim.x) {
        destination[i] = source[i];
    }
}

/// output[rows[i]] += scales[i] * input[i]
///
/// The inverse of the gather, folding the router weight in on the way. Each
/// destination row belongs to exactly one source row per launch - a token
/// selects an expert at most once - so no atomics are needed.
__global__ void scatterAddRowsKernel(const float* __restrict__ input,
                                     const u32* __restrict__ rows,
                                     const float* __restrict__ scales,
                                     float* __restrict__ output, u32 width) {
    const u32 row = blockIdx.y;
    const float scale = scales[row];
    const float* source = input + static_cast<size_t>(row) * width;
    float* destination = output + static_cast<size_t>(rows[row]) * width;

    for (u32 i = blockIdx.x * blockDim.x + threadIdx.x; i < width;
         i += gridDim.x * blockDim.x) {
        destination[i] += scale * source[i];
    }
}

/// Zeroes a buffer. The scatter accumulates, so the routed branch starts empty.
__global__ void fillZeroKernel(float* __restrict__ data, u64 count) {
    for (u64 i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
         i += static_cast<u64>(gridDim.x) * blockDim.x) {
        data[i] = 0.0f;
    }
}

// ---------------------------------------------------------------------------
// MoE combine
// ---------------------------------------------------------------------------

__global__ void moeCombineKernel(const float* __restrict__ expertOutputs,
                                 const float* __restrict__ weights,
                                 float* __restrict__ output, u32 topK, u32 hidden) {
    const u32 index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= hidden) {
        return;
    }

    float total = 0.0f;
    for (u32 k = 0; k < topK; ++k) {
        total += weights[k] *
                 expertOutputs[static_cast<size_t>(k) * hidden + index];
    }
    output[index] = total;
}

}  // namespace

// ---------------------------------------------------------------------------
// Launchers
// ---------------------------------------------------------------------------

namespace launch {

cudaError_t rmsNorm(cudaStream_t stream, const float* input, const __nv_bfloat16* weight,
                    float* output, u32 rows, u32 count, float eps) {
    if (weight != nullptr) {
        rmsNormKernel<kBlock, true>
                <<<rows, kBlock, 0, stream>>>(input, weight, output, count, eps);
    } else {
        rmsNormKernel<kBlock, false>
                <<<rows, kBlock, 0, stream>>>(input, nullptr, output, count, eps);
    }
    return cudaGetLastError();
}

cudaError_t dequantGemv(cudaStream_t stream, u32 bits, const u32* packed,
                        const __nv_bfloat16* scales, const __nv_bfloat16* biases,
                        const float* input, float* output, u32 outFeatures, u32 inFeatures,
                        u32 groupSize) {
    constexpr u32 kWarpsPerBlock = kBlock / kWarpSize;
    const u32 blocks = blocksFor(outFeatures, kWarpsPerBlock);

    switch (bits) {
        case 4:
            dequantGemvKernel<4, kWarpsPerBlock><<<blocks, kBlock, 0, stream>>>(
                    packed, scales, biases, input, output, outFeatures, inFeatures, groupSize);
            break;
        case 8:
            dequantGemvKernel<8, kWarpsPerBlock><<<blocks, kBlock, 0, stream>>>(
                    packed, scales, biases, input, output, outFeatures, inFeatures, groupSize);
            break;
        default:
            return cudaErrorInvalidValue;
    }
    return cudaGetLastError();
}

cudaError_t dequantGemm(cudaStream_t stream, u32 bits, const u32* packed,
                        const __nv_bfloat16* scales, const __nv_bfloat16* biases,
                        const float* input, float* output, u32 tokens, u32 outFeatures,
                        u32 inFeatures, u32 groupSize) {
    constexpr u32 kWarpsPerBlock = kBlock / kWarpSize;
    const dim3 blocks(blocksFor(outFeatures, kWarpsPerBlock * kRowsPerWarp),
                      blocksFor(tokens, kTokenTile));

    switch (bits) {
        case 4:
            dequantGemmKernel<4, kWarpsPerBlock, kRowsPerWarp><<<blocks, kBlock, 0, stream>>>(
                    packed, scales, biases, input, output, tokens, outFeatures, inFeatures,
                    groupSize);
            break;
        case 8:
            dequantGemmKernel<8, kWarpsPerBlock, kRowsPerWarp><<<blocks, kBlock, 0, stream>>>(
                    packed, scales, biases, input, output, tokens, outFeatures, inFeatures,
                    groupSize);
            break;
        default:
            return cudaErrorInvalidValue;
    }
    return cudaGetLastError();
}

u32 dequantGemmTokenTile() { return kTokenTile; }

cudaError_t embedLookup(cudaStream_t stream, u32 bits, const u32* packed,
                        const __nv_bfloat16* scales, const __nv_bfloat16* biases,
                        float* output, u32 tokenId, u32 hiddenSize, u32 groupSize,
                        float embedScale) {
    const u32 blocks = blocksFor(hiddenSize, kBlock);

    switch (bits) {
        case 4:
            embedLookupKernel<4><<<blocks, kBlock, 0, stream>>>(
                    packed, scales, biases, output, tokenId, hiddenSize, groupSize,
                    embedScale);
            break;
        case 8:
            embedLookupKernel<8><<<blocks, kBlock, 0, stream>>>(
                    packed, scales, biases, output, tokenId, hiddenSize, groupSize,
                    embedScale);
            break;
        default:
            return cudaErrorInvalidValue;
    }
    return cudaGetLastError();
}

cudaError_t rope(cudaStream_t stream, float* data, u32 tokens, u32 heads, u32 headDim,
                 u32 rotated, u64 basePosition, float theta) {
    const u32 rotatedPairs = rotated / 2;
    const u32 threads = rotatedPairs < kBlock ? (rotatedPairs == 0 ? 1 : rotatedPairs) : kBlock;
    ropeKernel<<<dim3(heads, tokens), threads, 0, stream>>>(data, heads, headDim, rotated,
                                                            basePosition, theta);
    return cudaGetLastError();
}

cudaError_t geglu(cudaStream_t stream, const float* gate, const float* up, float* output,
                  u32 count) {
    gegluKernel<<<blocksFor(count, kBlock), kBlock, 0, stream>>>(gate, up, output, count);
    return cudaGetLastError();
}

cudaError_t add(cudaStream_t stream, const float* a, const float* b, float* output,
                u32 count) {
    addKernel<<<blocksFor(count, kBlock), kBlock, 0, stream>>>(a, b, output, count);
    return cudaGetLastError();
}

cudaError_t scale(cudaStream_t stream, const float* input, float* output, u32 count,
                  float scalar) {
    scaleKernel<<<blocksFor(count, kBlock), kBlock, 0, stream>>>(input, output, count, scalar);
    return cudaGetLastError();
}

cudaError_t logitSoftcap(cudaStream_t stream, const float* input, float* output, u32 count,
                         float cap) {
    logitSoftcapKernel<<<blocksFor(count, kBlock), kBlock, 0, stream>>>(input, output, count,
                                                                        cap);
    return cudaGetLastError();
}

cudaError_t argmax(cudaStream_t stream, const float* input, u32* output, u32 count) {
    argmaxKernel<kBlock><<<1, kBlock, 0, stream>>>(input, output, count);
    return cudaGetLastError();
}

cudaError_t kvWrite(cudaStream_t stream, const float* key, const float* value,
                    __half* keyCache, __half* valueCache, u32 tokens, u32 kvHeads, u32 headDim,
                    u32 capacity, u64 basePosition, bool circular) {
    const u32 threads = headDim < kBlock ? headDim : kBlock;
    kvWriteKernel<<<dim3(kvHeads, tokens), threads, 0, stream>>>(
            key, value, keyCache, valueCache, kvHeads, headDim, capacity, basePosition,
            circular);
    return cudaGetLastError();
}

cudaError_t attention(cudaStream_t stream, const float* queries, const __half* keyCache,
                      const __half* valueCache, float* output, u32 tokens, u32 numHeads,
                      u32 kvHeads, u32 headDim, u32 capacity, u64 basePosition,
                      u32 slidingWindow, bool circular, float scale) {
    // Accumulator plus the staged query, both fp32.
    const size_t sharedBytes = static_cast<size_t>(headDim) * 2 * sizeof(float);

    attentionKernel<kBlock><<<dim3(numHeads, tokens), kBlock, sharedBytes, stream>>>(
            queries, keyCache, valueCache, output, numHeads, kvHeads, headDim, capacity,
            basePosition, slidingWindow, circular, scale);
    return cudaGetLastError();
}

cudaError_t routerTopK(cudaStream_t stream, const float* scores,
                       const __nv_bfloat16* perExpertScale, u32* outIndices,
                       float* outWeights, u32 tokens, u32 numExperts, u32 topK) {
    constexpr u32 kMaxTopK = 32;
    if (topK > kMaxTopK) {
        return cudaErrorInvalidValue;
    }
    const size_t sharedBytes = static_cast<size_t>(numExperts) * sizeof(float);

    routerTopKKernel<kBlock, kMaxTopK><<<tokens, kBlock, sharedBytes, stream>>>(
            scores, perExpertScale, outIndices, outWeights, numExperts, topK);
    return cudaGetLastError();
}

cudaError_t gatherRows(cudaStream_t stream, const float* input, const u32* rows, float* output,
                       u32 count, u32 width) {
    if (count == 0) {
        return cudaSuccess;
    }
    gatherRowsKernel<<<dim3(blocksFor(width, kBlock), count), kBlock, 0, stream>>>(
            input, rows, output, width);
    return cudaGetLastError();
}

cudaError_t scatterAddRows(cudaStream_t stream, const float* input, const u32* rows,
                           const float* scales, float* output, u32 count, u32 width) {
    if (count == 0) {
        return cudaSuccess;
    }
    scatterAddRowsKernel<<<dim3(blocksFor(width, kBlock), count), kBlock, 0, stream>>>(
            input, rows, scales, output, width);
    return cudaGetLastError();
}

cudaError_t fillZero(cudaStream_t stream, float* data, u64 count) {
    if (count == 0) {
        return cudaSuccess;
    }
    constexpr u64 kMaxBlocks = 65535;
    const u64 blocks = (count + kBlock - 1) / kBlock;
    fillZeroKernel<<<static_cast<u32>(blocks < kMaxBlocks ? blocks : kMaxBlocks), kBlock, 0,
                     stream>>>(data, count);
    return cudaGetLastError();
}

cudaError_t moeCombine(cudaStream_t stream, const float* expertOutputs,
                       const float* weights, float* output, u32 topK, u32 hidden) {
    moeCombineKernel<<<blocksFor(hidden, kBlock), kBlock, 0, stream>>>(expertOutputs, weights,
                                                                       output, topK, hidden);
    return cudaGetLastError();
}

}  // namespace launch
}  // namespace tf::gpu
