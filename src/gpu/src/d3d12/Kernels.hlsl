// Compute shaders for the D3D12 backend.
//
// Translated from the CUDA kernels in src/gpu/src/cuda/CudaKernels.cu, and
// checked against the same scalar CPU references by the same tests. The
// correspondence is mechanical:
//
//   __shared__ T[]                 -> groupshared T[]
//   __syncthreads()                -> GroupMemoryBarrierWithGroupSync()
//   warpReduceSum via __shfl_xor   -> WaveActiveSum
//   blockReduceSum                 -> WaveActiveSum then a groupshared pass
//   threadIdx.x / 32               -> groupThread / WaveGetLaneCount()
//   thread_index_in_warp           -> WaveGetLaneIndex()
//
// Wave width is read at runtime rather than assumed to be 32. NVIDIA is always
// 32, AMD's RDNA can be either, and Intel varies by generation; every place the
// width matters here is a stride, so nothing needs it to be a constant.
//
// Storage types match the CUDA side exactly:
//   activations  float   - Gemma 4's do not fit in fp16
//   KV cache     half    - post-normalization and bounded near unit RMS
//   weights      uint packed, with bfloat16 scales and biases
//
// bfloat16 has no hardware conversion here, but it does not need one: it is the
// top half of a float, so widening is a shift.

// ---------------------------------------------------------------------------
// Bindings
// ---------------------------------------------------------------------------
//
// Root descriptors, no descriptor heap. Which buffer is which is documented per
// kernel; unused slots are simply not bound.

RWByteAddressBuffer B0 : register(u0);
RWByteAddressBuffer B1 : register(u1);
RWByteAddressBuffer B2 : register(u2);
RWByteAddressBuffer B3 : register(u3);
RWByteAddressBuffer B4 : register(u4);
RWByteAddressBuffer B5 : register(u5);

cbuffer Constants : register(b0) {
    uint4 C0;
    uint4 C1;
    uint4 C2;
    uint4 C3;
};

#define BLOCK 256

// ---------------------------------------------------------------------------
// Conversions
// ---------------------------------------------------------------------------

/// bfloat16 at a two-byte offset. The value is the top 16 bits of the float it
/// represents, so widening is a shift with no table and no rounding.
float loadBf16(RWByteAddressBuffer buffer, uint index) {
    const uint word = buffer.Load(index * 2 & ~3u);
    const uint bits = ((index & 1u) != 0u) ? (word >> 16) : (word & 0xFFFFu);
    return asfloat(bits << 16);
}

float loadHalf(RWByteAddressBuffer buffer, uint index) {
    const uint word = buffer.Load(index * 2 & ~3u);
    const uint bits = ((index & 1u) != 0u) ? (word >> 16) : (word & 0xFFFFu);
    return f16tof32(bits);
}

/// Stores one half without disturbing its neighbour in the same dword.
///
/// A read-modify-write would race whenever two threads share a dword, so this
/// uses an atomic on the packed word instead. The KV writer has adjacent
/// threads writing adjacent halves constantly, so the race is not hypothetical.
void storeHalf(RWByteAddressBuffer buffer, uint index, float value) {
    const uint address = (index * 2) & ~3u;
    const uint bits = f32tof16(value) & 0xFFFFu;
    if ((index & 1u) != 0u) {
        buffer.InterlockedAnd(address, 0x0000FFFFu);
        buffer.InterlockedOr(address, bits << 16);
    } else {
        buffer.InterlockedAnd(address, 0xFFFF0000u);
        buffer.InterlockedOr(address, bits);
    }
}

/// tanh, computed rather than taken from the intrinsic.
///
/// HLSL's tanh is a driver-supplied approximation whose accuracy is not
/// specified, and it measured about 2.5e-5 of absolute error against the CPU
/// reference - enough to fail the elementwise tolerance on GeGLU outputs near
/// zero, where a small absolute error is a large relative one. Building it from
/// exp is both more accurate here and the same on every driver, which matters
/// more than the handful of cycles.
///
/// The exponent is negated on the positive branch so it never overflows.
float stableTanh(float x) {
    const float magnitude = abs(x);
    const float e = exp(-2.0f * magnitude);
    const float result = (1.0f - e) / (1.0f + e);
    return x < 0.0f ? -result : result;
}

/// Sum across the whole group. WaveActiveSum handles the wave; the second pass
/// combines per-wave partials.
groupshared float g_partials[BLOCK / 4];

float groupSum(float value, uint groupThread) {
    const uint lanes = WaveGetLaneCount();
    const uint waves = (BLOCK + lanes - 1) / lanes;
    const uint wave = groupThread / lanes;

    const float waveTotal = WaveActiveSum(value);
    if (WaveIsFirstLane()) {
        g_partials[wave] = waveTotal;
    }
    GroupMemoryBarrierWithGroupSync();

    float total = 0.0f;
    for (uint i = 0; i < waves; ++i) {
        total += g_partials[i];
    }
    return total;
}

// ---------------------------------------------------------------------------
// RMSNorm
// ---------------------------------------------------------------------------
//
// B0 input (float), B1 weight (bf16), B2 output (float).
// C0 = (count, 0, 0, 0), C1.x = eps as float bits.
//
// Gemma 4 applies the stored weight directly, not (1 + weight).

[numthreads(BLOCK, 1, 1)]
void RmsNorm(uint3 group : SV_GroupID, uint3 thread : SV_GroupThreadID) {
    const uint count = C0.x;
    const float eps = asfloat(C1.x);
    const uint row = group.x;
    const uint base = row * count;

    float sumSquares = 0.0f;
    for (uint i = thread.x; i < count; i += BLOCK) {
        const float value = B0.Load<float>((base + i) * 4);
        sumSquares += value * value;
    }
    sumSquares = groupSum(sumSquares, thread.x);

    const float inverseRms = rsqrt(sumSquares / (float)count + eps);
    for (uint j = thread.x; j < count; j += BLOCK) {
        const float value = B0.Load<float>((base + j) * 4) * inverseRms;
        B2.Store<float>((base + j) * 4, value * loadBf16(B1, j));
    }
}

/// v_norm has no learnable weight at all, which is why this exists rather than
/// a weight vector of ones.
[numthreads(BLOCK, 1, 1)]
void RmsNormNoWeight(uint3 group : SV_GroupID, uint3 thread : SV_GroupThreadID) {
    const uint count = C0.x;
    const float eps = asfloat(C1.x);
    const uint base = group.x * count;

    float sumSquares = 0.0f;
    for (uint i = thread.x; i < count; i += BLOCK) {
        const float value = B0.Load<float>((base + i) * 4);
        sumSquares += value * value;
    }
    sumSquares = groupSum(sumSquares, thread.x);

    const float inverseRms = rsqrt(sumSquares / (float)count + eps);
    for (uint j = thread.x; j < count; j += BLOCK) {
        B2.Store<float>((base + j) * 4, B0.Load<float>((base + j) * 4) * inverseRms);
    }
}

// ---------------------------------------------------------------------------
// Elementwise
// ---------------------------------------------------------------------------
//
// B0 a, B1 b, B2 output. C0.x = count, C1.x = a float parameter where used.

[numthreads(BLOCK, 1, 1)]
void Geglu(uint3 id : SV_DispatchThreadID) {
    if (id.x >= C0.x) {
        return;
    }
    const float g = B0.Load<float>(id.x * 4);
    const float u = B1.Load<float>(id.x * 4);

    // gelu_pytorch_tanh, matching the reference exactly. The gate is what
    // passes through the activation; the up projection multiplies afterwards.
    const float kSqrt2OverPi = 0.7978845608028654f;
    const float kCoefficient = 0.044715f;
    const float inner = kSqrt2OverPi * (g + kCoefficient * g * g * g);
    const float activated = 0.5f * g * (1.0f + stableTanh(inner));

    B2.Store<float>(id.x * 4, activated * u);
}

[numthreads(BLOCK, 1, 1)]
void Add(uint3 id : SV_DispatchThreadID) {
    if (id.x >= C0.x) {
        return;
    }
    B2.Store<float>(id.x * 4, B0.Load<float>(id.x * 4) + B1.Load<float>(id.x * 4));
}

[numthreads(BLOCK, 1, 1)]
void Scale(uint3 id : SV_DispatchThreadID) {
    if (id.x >= C0.x) {
        return;
    }
    B2.Store<float>(id.x * 4, B0.Load<float>(id.x * 4) * asfloat(C1.x));
}

[numthreads(BLOCK, 1, 1)]
void LogitSoftcap(uint3 id : SV_DispatchThreadID) {
    if (id.x >= C0.x) {
        return;
    }
    const float cap = asfloat(C1.x);
    B2.Store<float>(id.x * 4, stableTanh(B0.Load<float>(id.x * 4) / cap) * cap);
}

[numthreads(BLOCK, 1, 1)]
void FillZero(uint3 id : SV_DispatchThreadID) {
    if (id.x >= C0.x) {
        return;
    }
    B2.Store<float>(id.x * 4, 0.0f);
}

// ---------------------------------------------------------------------------
// Argmax
// ---------------------------------------------------------------------------
//
// B0 input (float), B2 output (uint). C0.x = count.
//
// One group striding the whole vector. The vocabulary is 262144, so this makes
// 1024 passes - cheap beside the head projection that produced the logits, and
// it avoids a second dispatch to combine partial results.

groupshared float g_bestValue[BLOCK];
groupshared uint g_bestIndex[BLOCK];

[numthreads(BLOCK, 1, 1)]
void Argmax(uint3 thread : SV_GroupThreadID) {
    const uint count = C0.x;

    float bestValue = -1.#INF;
    uint bestIndex = 0;
    for (uint i = thread.x; i < count; i += BLOCK) {
        const float value = B0.Load<float>(i * 4);
        // Strictly greater keeps the lowest index on a tie, which is what makes
        // greedy decoding reproducible - and what the CPU sampler also does.
        if (value > bestValue) {
            bestValue = value;
            bestIndex = i;
        }
    }
    g_bestValue[thread.x] = bestValue;
    g_bestIndex[thread.x] = bestIndex;
    GroupMemoryBarrierWithGroupSync();

    for (uint stride = BLOCK / 2; stride > 0; stride >>= 1) {
        if (thread.x < stride) {
            const uint other = thread.x + stride;
            const bool takeOther = g_bestValue[other] > g_bestValue[thread.x] ||
                                   (g_bestValue[other] == g_bestValue[thread.x] &&
                                    g_bestIndex[other] < g_bestIndex[thread.x]);
            if (takeOther) {
                g_bestValue[thread.x] = g_bestValue[other];
                g_bestIndex[thread.x] = g_bestIndex[other];
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (thread.x == 0) {
        B2.Store(0, g_bestIndex[0]);
    }
}

// ---------------------------------------------------------------------------
// RoPE
// ---------------------------------------------------------------------------
//
// B0 data (float, modified in place).
// C0 = (heads, headDim, rotated, 0), C1.x = theta bits,
// C2 = (basePosition low, basePosition high, 0, 0).
//
// Non-interleaved, and partial rotation does NOT shrink the pairing: pair j
// always joins dimensions j and j + headDim/2, the exponent always divides by
// headDim, and only the count of rotating pairs is reduced. Pairing across
// rotated/2 instead would be a valid rotation of the wrong axes.

[numthreads(BLOCK, 1, 1)]
void Rope(uint3 group : SV_GroupID, uint3 thread : SV_GroupThreadID) {
    const uint heads = C0.x;
    const uint headDim = C0.y;
    const uint rotated = C0.z;
    const float theta = asfloat(C1.x);
    const uint basePosition = C2.x;

    const uint head = group.x;
    const uint token = group.y;
    const float position = (float)(basePosition + token);

    const uint half = headDim / 2;
    const uint rotatedPairs = rotated / 2;
    const uint base = (token * heads + head) * headDim;

    for (uint i = thread.x; i < rotatedPairs; i += BLOCK) {
        const float exponent = (float)(2 * i) / (float)headDim;
        const float frequency = pow(theta, -exponent);
        const float angle = position * frequency;

        float sine;
        float cosine;
        sincos(angle, sine, cosine);

        const float lower = B0.Load<float>((base + i) * 4);
        const float upper = B0.Load<float>((base + i + half) * 4);

        B0.Store<float>((base + i) * 4, lower * cosine - upper * sine);
        B0.Store<float>((base + i + half) * 4, lower * sine + upper * cosine);
    }
}

// ---------------------------------------------------------------------------
// Quantized GEMV
// ---------------------------------------------------------------------------
//
// B0 packed (uint), B1 scales (bf16), B2 biases (bf16), B3 input (float),
// B4 output (float).
// C0 = (outFeatures, inFeatures, groupSize, bits).
//
// One output row per wave. Lanes stride over whole packed words, so a
// scale/bias pair is loaded once per word rather than once per value.
// Dequantization is affine - q * scale + bias - and scales may be negative.

float gemvRow(uint row, uint outFeatures, uint inFeatures, uint groupSize, uint bits,
              uint lane, uint lanes) {
    const uint valuesPerWord = 32 / bits;
    const uint mask = (1u << bits) - 1u;
    const uint wordsPerRow = inFeatures / valuesPerWord;
    const uint groupsPerRow = inFeatures / groupSize;
    const uint wordsPerGroup = groupSize / valuesPerWord;

    float accumulator = 0.0f;
    for (uint word = lane; word < wordsPerRow; word += lanes) {
        const uint group = word / wordsPerGroup;
        const float scale = loadBf16(B1, row * groupsPerRow + group);
        const float bias = loadBf16(B2, row * groupsPerRow + group);

        const uint packed = B0.Load((row * wordsPerRow + word) * 4);
        const uint base = word * valuesPerWord;

        for (uint slot = 0; slot < valuesPerWord; ++slot) {
            const uint quantized = (packed >> (slot * bits)) & mask;
            accumulator += ((float)quantized * scale + bias) * B3.Load<float>((base + slot) * 4);
        }
    }
    return WaveActiveSum(accumulator);
}

[numthreads(BLOCK, 1, 1)]
void DequantGemv4(uint3 group : SV_GroupID, uint3 thread : SV_GroupThreadID) {
    const uint lanes = WaveGetLaneCount();
    const uint wavesPerGroup = BLOCK / lanes;
    const uint row = group.x * wavesPerGroup + thread.x / lanes;
    if (row >= C0.x) {
        return;
    }
    const float total = gemvRow(row, C0.x, C0.y, C0.z, 4, WaveGetLaneIndex(), lanes);
    if (WaveIsFirstLane()) {
        B4.Store<float>(row * 4, total);
    }
}

[numthreads(BLOCK, 1, 1)]
void DequantGemv8(uint3 group : SV_GroupID, uint3 thread : SV_GroupThreadID) {
    const uint lanes = WaveGetLaneCount();
    const uint wavesPerGroup = BLOCK / lanes;
    const uint row = group.x * wavesPerGroup + thread.x / lanes;
    if (row >= C0.x) {
        return;
    }
    const float total = gemvRow(row, C0.x, C0.y, C0.z, 8, WaveGetLaneIndex(), lanes);
    if (WaveIsFirstLane()) {
        B4.Store<float>(row * 4, total);
    }
}

// ---------------------------------------------------------------------------
// Quantized GEMM
// ---------------------------------------------------------------------------
//
// Same buffers as the GEMV, plus C1.x = tokens.
//
// A GEMV reads the whole matrix to produce one output column, so N tokens read
// it N times. Here each lane loads a packed word once and feeds it to
// TOKEN_TILE tokens, and carries ROWS output rows so one activation read serves
// that many rows of arithmetic. The CUDA version measured 10.8x over the GEMV
// launches it replaces with the same two tilings.

#define TOKEN_TILE 16
#define FEATURE_CHUNK 256
#define ROWS 4

groupshared float g_tile[TOKEN_TILE][FEATURE_CHUNK];

void gemmBody(uint3 group, uint3 thread, uint bits) {
    const uint outFeatures = C0.x;
    const uint inFeatures = C0.y;
    const uint groupSize = C0.z;
    const uint tokens = C1.x;

    const uint valuesPerWord = 32 / bits;
    const uint mask = (1u << bits) - 1u;
    const uint wordsPerRow = inFeatures / valuesPerWord;
    const uint groupsPerRow = inFeatures / groupSize;
    const uint wordsPerGroup = groupSize / valuesPerWord;

    const uint lanes = WaveGetLaneCount();
    const uint lane = WaveGetLaneIndex();
    const uint wavesPerGroup = BLOCK / lanes;
    const uint firstRow = (group.x * wavesPerGroup + thread.x / lanes) * ROWS;
    const uint tokenBase = group.y * TOKEN_TILE;

    float accumulator[ROWS][TOKEN_TILE];
    for (uint r0 = 0; r0 < ROWS; ++r0) {
        for (uint t0 = 0; t0 < TOKEN_TILE; ++t0) {
            accumulator[r0][t0] = 0.0f;
        }
    }

    for (uint chunk = 0; chunk < inFeatures; chunk += FEATURE_CHUNK) {
        GroupMemoryBarrierWithGroupSync();

        // Feature index varies fastest across threads, so loads coalesce and
        // the groupshared writes land in consecutive banks.
        for (uint index = thread.x; index < TOKEN_TILE * FEATURE_CHUNK; index += BLOCK) {
            const uint t = index / FEATURE_CHUNK;
            const uint feature = index % FEATURE_CHUNK;
            const uint token = tokenBase + t;
            const uint column = chunk + feature;
            g_tile[t][feature] = (token < tokens && column < inFeatures)
                                         ? B3.Load<float>((token * inFeatures + column) * 4)
                                         : 0.0f;
        }

        GroupMemoryBarrierWithGroupSync();

        if (firstRow >= outFeatures) {
            continue;
        }

        const uint wordBase = chunk / valuesPerWord;
        const uint wordsPerChunk = FEATURE_CHUNK / valuesPerWord;

        for (uint word = lane; word < wordsPerChunk; word += lanes) {
            const uint globalWord = wordBase + word;
            if (globalWord >= wordsPerRow) {
                break;
            }
            const uint quantGroup = globalWord / wordsPerGroup;

            float weight[ROWS][8];
            for (uint r = 0; r < ROWS; ++r) {
                const uint row = min(firstRow + r, outFeatures - 1);
                const float scale = loadBf16(B1, row * groupsPerRow + quantGroup);
                const float bias = loadBf16(B2, row * groupsPerRow + quantGroup);
                const uint packed = B0.Load((row * wordsPerRow + globalWord) * 4);
                for (uint slot = 0; slot < valuesPerWord; ++slot) {
                    weight[r][slot] = (float)((packed >> (slot * bits)) & mask) * scale + bias;
                }
            }

            const uint feature = word * valuesPerWord;
            for (uint t = 0; t < TOKEN_TILE; ++t) {
                // Read once, use for every row this lane owns. That reuse is
                // the whole point of ROWS.
                float activation[8];
                for (uint s = 0; s < valuesPerWord; ++s) {
                    activation[s] = g_tile[t][feature + s];
                }
                for (uint r2 = 0; r2 < ROWS; ++r2) {
                    for (uint s2 = 0; s2 < valuesPerWord; ++s2) {
                        accumulator[r2][t] += weight[r2][s2] * activation[s2];
                    }
                }
            }
        }
    }

    for (uint r3 = 0; r3 < ROWS; ++r3) {
        const uint row = firstRow + r3;
        if (row >= outFeatures) {
            continue;
        }
        for (uint t3 = 0; t3 < TOKEN_TILE; ++t3) {
            const float total = WaveActiveSum(accumulator[r3][t3]);
            const uint token = tokenBase + t3;
            if (WaveIsFirstLane() && token < tokens) {
                B4.Store<float>((token * outFeatures + row) * 4, total);
            }
        }
    }
}

[numthreads(BLOCK, 1, 1)]
void DequantGemm4(uint3 group : SV_GroupID, uint3 thread : SV_GroupThreadID) {
    gemmBody(group, thread, 4);
}

[numthreads(BLOCK, 1, 1)]
void DequantGemm8(uint3 group : SV_GroupID, uint3 thread : SV_GroupThreadID) {
    gemmBody(group, thread, 8);
}

// ---------------------------------------------------------------------------
// Embedding lookup
// ---------------------------------------------------------------------------
//
// B0 packed, B1 scales, B2 biases, B4 output (float).
// C0 = (hiddenSize, groupSize, tokenId, bits), C1.x = embedScale bits.
//
// Scaled by sqrt(hiddenSize) on the way in; the tied output head does not
// apply it on the way out.

void embedBody(uint3 id, uint bits) {
    const uint hiddenSize = C0.x;
    const uint groupSize = C0.y;
    const uint tokenId = C0.z;
    const float embedScale = asfloat(C1.x);

    if (id.x >= hiddenSize) {
        return;
    }

    const uint valuesPerWord = 32 / bits;
    const uint mask = (1u << bits) - 1u;
    const uint wordsPerRow = hiddenSize / valuesPerWord;
    const uint groupsPerRow = hiddenSize / groupSize;

    const uint word = id.x / valuesPerWord;
    const uint slot = id.x % valuesPerWord;
    const uint quantGroup = id.x / groupSize;

    const uint packed = B0.Load((tokenId * wordsPerRow + word) * 4);
    const uint quantized = (packed >> (slot * bits)) & mask;
    const float scale = loadBf16(B1, tokenId * groupsPerRow + quantGroup);
    const float bias = loadBf16(B2, tokenId * groupsPerRow + quantGroup);

    B4.Store<float>(id.x * 4, ((float)quantized * scale + bias) * embedScale);
}

[numthreads(BLOCK, 1, 1)]
void EmbedLookup4(uint3 id : SV_DispatchThreadID) { embedBody(id, 4); }

[numthreads(BLOCK, 1, 1)]
void EmbedLookup8(uint3 id : SV_DispatchThreadID) { embedBody(id, 8); }

// ---------------------------------------------------------------------------
// KV cache write
// ---------------------------------------------------------------------------
//
// B0 key (float), B1 value (float), B2 key cache (half), B3 value cache (half).
// C0 = (kvHeads, headDim, capacity, circular), C1.x = basePosition.
//
// The sliding layers use a ring: logical position p lands at row p % capacity.
// The full-attention layers append linearly.

[numthreads(BLOCK, 1, 1)]
void KvWrite(uint3 group : SV_GroupID, uint3 thread : SV_GroupThreadID) {
    const uint kvHeads = C0.x;
    const uint headDim = C0.y;
    const uint capacity = C0.z;
    const bool circular = C0.w != 0u;
    const uint basePosition = C1.x;

    const uint head = group.x;
    const uint token = group.y;
    const uint position = basePosition + token;
    const uint slot = circular ? (position % capacity) : position;

    const uint sourceBase = (token * kvHeads + head) * headDim;
    const uint cacheBase = (head * capacity + slot) * headDim;

    for (uint i = thread.x; i < headDim; i += BLOCK) {
        storeHalf(B2, cacheBase + i, B0.Load<float>((sourceBase + i) * 4));
        storeHalf(B3, cacheBase + i, B1.Load<float>((sourceBase + i) * 4));
    }
}

// ---------------------------------------------------------------------------
// Attention
// ---------------------------------------------------------------------------
//
// B0 queries (float), B1 key cache (half), B2 value cache (half),
// B4 output (float).
// C0 = (numHeads, kvHeads, headDim, capacity),
// C1 = (basePosition, slidingWindow, circular, 0), C2.x = scale bits.
//
// Streaming softmax in a single pass: the running maximum and denominator are
// rescaled as larger scores appear, so scores are never materialized. Storing
// them would cost up to 4096 floats of groupshared per head, more than a group
// has - and O(tokens x history) during prefill, where this costs none.
//
// Causality is per token: query t sits at basePosition + t and attends to
// everything up to and including itself, so a prefill chunk needs no mask
// tensor and a decode step is the one-token case of the same shader.

groupshared float g_accumulator[512];
groupshared float g_query[512];
groupshared float g_max;
groupshared float g_denominator;
groupshared float g_weight;
groupshared float g_rescale;

[numthreads(BLOCK, 1, 1)]
void Attention(uint3 group : SV_GroupID, uint3 thread : SV_GroupThreadID) {
    const uint numHeads = C0.x;
    const uint kvHeads = C0.y;
    const uint headDim = C0.z;
    const uint capacity = C0.w;
    const uint basePosition = C1.x;
    const uint slidingWindow = C1.y;
    const bool circular = C1.z != 0u;
    const float scale = asfloat(C2.x);

    const uint head = group.x;
    if (head >= numHeads) {
        return;
    }
    const uint token = group.y;
    const uint queryPosition = basePosition + token;
    const uint cachedLength = queryPosition + 1;

    uint firstVisible = 0;
    if (slidingWindow > 0 && cachedLength > slidingWindow) {
        firstVisible = cachedLength - slidingWindow;
    }
    if (circular && cachedLength > capacity) {
        firstVisible = max(firstVisible, cachedLength - capacity);
    }

    const uint groupSize = numHeads / kvHeads;
    const uint kvHead = head / groupSize;
    const uint queryBase = (token * numHeads + head) * headDim;

    for (uint i = thread.x; i < headDim; i += BLOCK) {
        g_query[i] = B0.Load<float>((queryBase + i) * 4);
        g_accumulator[i] = 0.0f;
    }
    if (thread.x == 0) {
        g_max = -1.#INF;
        g_denominator = 0.0f;
    }
    GroupMemoryBarrierWithGroupSync();

    for (uint position = firstVisible; position < cachedLength; ++position) {
        const uint slot = circular ? (position % capacity) : position;
        const uint rowBase = (kvHead * capacity + slot) * headDim;

        float dot = 0.0f;
        for (uint d = thread.x; d < headDim; d += BLOCK) {
            dot += g_query[d] * loadHalf(B1, rowBase + d);
        }
        dot = groupSum(dot, thread.x) * scale;

        if (thread.x == 0) {
            const float newMax = max(g_max, dot);
            g_rescale = isinf(g_max) ? 0.0f : exp(g_max - newMax);
            g_weight = exp(dot - newMax);
            g_max = newMax;
            g_denominator = g_denominator * g_rescale + g_weight;
        }
        GroupMemoryBarrierWithGroupSync();

        for (uint e = thread.x; e < headDim; e += BLOCK) {
            g_accumulator[e] = g_accumulator[e] * g_rescale +
                               g_weight * loadHalf(B2, rowBase + e);
        }
        GroupMemoryBarrierWithGroupSync();
    }

    const float denominator = g_denominator;
    const uint outBase = (token * numHeads + head) * headDim;
    for (uint o = thread.x; o < headDim; o += BLOCK) {
        // An empty window leaves the denominator at zero; emit zeros rather
        // than NaN.
        B4.Store<float>((outBase + o) * 4,
                        denominator > 0.0f ? g_accumulator[o] / denominator : 0.0f);
    }
}

// ---------------------------------------------------------------------------
// Router top-k
// ---------------------------------------------------------------------------
//
// B0 scores (float), B1 per-expert scale (bf16), B2 out indices (uint),
// B3 out weights (float).
// C0 = (numExperts, topK, 0, 0).
//
// Selection is by raw score; the softmax runs afterwards over just the selected
// scores, and the per-expert scale multiplies after that - so the weights
// deliberately do not sum to one.

#define MAX_EXPERTS 256
#define MAX_TOPK 32

groupshared float g_working[MAX_EXPERTS];
groupshared float g_selectedScore[MAX_TOPK];
groupshared uint g_selectedIndex[MAX_TOPK];

[numthreads(BLOCK, 1, 1)]
void RouterTopK(uint3 group : SV_GroupID, uint3 thread : SV_GroupThreadID) {
    const uint numExperts = C0.x;
    const uint topK = C0.y;
    const uint token = group.x;

    for (uint i = thread.x; i < numExperts; i += BLOCK) {
        g_working[i] = B0.Load<float>((token * numExperts + i) * 4);
    }
    GroupMemoryBarrierWithGroupSync();

    // Repeated argmax rather than a sort: k is 8 against 128 experts, so eight
    // cheap reductions win.
    for (uint round = 0; round < topK; ++round) {
        float bestValue = -1.#INF;
        uint bestIndex = 0;
        for (uint e = thread.x; e < numExperts; e += BLOCK) {
            if (g_working[e] > bestValue) {
                bestValue = g_working[e];
                bestIndex = e;
            }
        }
        g_bestValue[thread.x] = bestValue;
        g_bestIndex[thread.x] = bestIndex;
        GroupMemoryBarrierWithGroupSync();

        for (uint stride = BLOCK / 2; stride > 0; stride >>= 1) {
            if (thread.x < stride) {
                const uint other = thread.x + stride;
                // Ties resolve to the lower index, matching the reference.
                const bool takeOther = g_bestValue[other] > g_bestValue[thread.x] ||
                                       (g_bestValue[other] == g_bestValue[thread.x] &&
                                        g_bestIndex[other] < g_bestIndex[thread.x]);
                if (takeOther) {
                    g_bestValue[thread.x] = g_bestValue[other];
                    g_bestIndex[thread.x] = g_bestIndex[other];
                }
            }
            GroupMemoryBarrierWithGroupSync();
        }

        if (thread.x == 0) {
            g_selectedScore[round] = g_bestValue[0];
            g_selectedIndex[round] = g_bestIndex[0];
            // Remove the winner so the next round finds the following expert.
            g_working[g_bestIndex[0]] = -1.#INF;
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (thread.x == 0) {
        float maximum = -1.#INF;
        for (uint a = 0; a < topK; ++a) {
            maximum = max(maximum, g_selectedScore[a]);
        }
        float total = 0.0f;
        for (uint b = 0; b < topK; ++b) {
            g_selectedScore[b] = exp(g_selectedScore[b] - maximum);
            total += g_selectedScore[b];
        }
        for (uint c = 0; c < topK; ++c) {
            const uint expert = g_selectedIndex[c];
            B2.Store((token * topK + c) * 4, expert);
            B3.Store<float>((token * topK + c) * 4,
                            (g_selectedScore[c] / total) * loadBf16(B1, expert));
        }
    }
}

// ---------------------------------------------------------------------------
// MoE combine
// ---------------------------------------------------------------------------
//
// B0 expert outputs (float), B1 weights (float), B2 output (float).
// C0 = (topK, hidden, 0, 0).

[numthreads(BLOCK, 1, 1)]
void MoeCombine(uint3 id : SV_DispatchThreadID) {
    const uint topK = C0.x;
    const uint hidden = C0.y;
    if (id.x >= hidden) {
        return;
    }

    float total = 0.0f;
    for (uint k = 0; k < topK; ++k) {
        total += B1.Load<float>(k * 4) * B0.Load<float>((k * hidden + id.x) * 4);
    }
    B2.Store<float>(id.x * 4, total);
}

// ---------------------------------------------------------------------------
// Row gather and scatter
// ---------------------------------------------------------------------------
//
// B0 input (float), B1 rows (uint), B2 output (float), B3 scales (float).
// C0 = (count, width, 0, 0).
//
// Prefill runs one expert GEMM over all the tokens that selected that expert
// rather than one GEMV per (token, expert). These collect those tokens and put
// the results back.

[numthreads(BLOCK, 1, 1)]
void GatherRows(uint3 group : SV_GroupID, uint3 thread : SV_GroupThreadID,
                uint3 id : SV_DispatchThreadID) {
    const uint width = C0.y;
    const uint row = group.y;
    const uint sourceRow = B1.Load(row * 4);

    for (uint i = id.x; i < width; i += BLOCK * 256) {
        B2.Store<float>((row * width + i) * 4, B0.Load<float>((sourceRow * width + i) * 4));
    }
}

/// output[rows[i]] += scales[i] * input[i]
///
/// A token selects a given expert at most once, so within one dispatch no two
/// source rows share a destination and no atomics are needed.
[numthreads(BLOCK, 1, 1)]
void ScatterAddRows(uint3 group : SV_GroupID, uint3 thread : SV_GroupThreadID,
                    uint3 id : SV_DispatchThreadID) {
    const uint width = C0.y;
    const uint row = group.y;
    const uint destinationRow = B1.Load(row * 4);
    const float scale = B3.Load<float>(row * 4);

    for (uint i = id.x; i < width; i += BLOCK * 256) {
        const uint address = (destinationRow * width + i) * 4;
        B2.Store<float>(address,
                        B2.Load<float>(address) + scale * B0.Load<float>((row * width + i) * 4));
    }
}
