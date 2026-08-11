#include "tf/reference/Ops.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace tf::reference {
namespace {

/// Extracts one quantized value from a packed word array. Values are stored
/// low-order first: at 4 bits, element 0 is the low nibble of word 0.
[[nodiscard]] u32 unpackValue(std::span<const u32> packed, u64 index, u32 bits) {
    const u32 perWord = 32 / bits;
    const u64 word = index / perWord;
    const u32 slot = static_cast<u32>(index % perWord);
    const u32 shift = slot * bits;
    const u32 mask = (bits == 32) ? 0xFFFFFFFFu : ((1u << bits) - 1u);
    return (packed[word] >> shift) & mask;
}

}  // namespace

// ---------------------------------------------------------------------------
// Quantization
// ---------------------------------------------------------------------------

float dequantizeElement(std::span<const u32> packed, std::span<const bf16> scales,
                        std::span<const bf16> biases, u64 row, u64 column, u64 inFeatures,
                        QuantSpec spec) {
    const u64 perWord = 32 / spec.bits;
    const u64 wordsPerRow = inFeatures / perWord;
    const u64 groupsPerRow = inFeatures / spec.groupSize;

    const u64 flatIndex = row * wordsPerRow * perWord + column;
    const u32 quantized = unpackValue(packed, flatIndex, spec.bits);

    const u64 group = row * groupsPerRow + column / spec.groupSize;
    const float scale = toFloat(scales[group]);
    const float bias = toFloat(biases[group]);

    return static_cast<float>(quantized) * scale + bias;
}

std::vector<float> dequantizeMatrix(std::span<const u32> packed, std::span<const bf16> scales,
                                    std::span<const bf16> biases,
                                    const QuantizedLinearLayout& layout) {
    std::vector<float> out(static_cast<usize>(layout.outFeatures * layout.inFeatures));
    for (u64 row = 0; row < layout.outFeatures; ++row) {
        for (u64 column = 0; column < layout.inFeatures; ++column) {
            out[static_cast<usize>(row * layout.inFeatures + column)] = dequantizeElement(
                    packed, scales, biases, row, column, layout.inFeatures, layout.spec);
        }
    }
    return out;
}

QuantizedMatrix quantizeMatrix(std::span<const float> values,
                               const QuantizedLinearLayout& layout) {
    const u64 perWord = layout.valuesPerWord();
    const u64 wordsPerRow = layout.packedWordsPerRow();
    const u64 groupsPerRow = layout.groupsPerRow();
    const u32 levels = (1u << layout.spec.bits) - 1u;

    QuantizedMatrix out;
    out.packed.assign(static_cast<usize>(layout.outFeatures * wordsPerRow), 0);
    out.scales.assign(static_cast<usize>(layout.outFeatures * groupsPerRow), bf16{});
    out.biases.assign(static_cast<usize>(layout.outFeatures * groupsPerRow), bf16{});

    for (u64 row = 0; row < layout.outFeatures; ++row) {
        for (u64 group = 0; group < groupsPerRow; ++group) {
            const u64 begin = group * layout.spec.groupSize;
            const u64 end = begin + layout.spec.groupSize;

            float minimum = values[static_cast<usize>(row * layout.inFeatures + begin)];
            float maximum = minimum;
            for (u64 i = begin; i < end; ++i) {
                const float value = values[static_cast<usize>(row * layout.inFeatures + i)];
                minimum = std::min(minimum, value);
                maximum = std::max(maximum, value);
            }

            // Affine: bias carries the minimum, scale spans the range. Rounding
            // both through bf16 first means the reference dequantization sees
            // exactly the stored values.
            const bf16 scaleBf = toBf16((maximum - minimum) / static_cast<float>(levels));
            const bf16 biasBf = toBf16(minimum);
            const float scale = toFloat(scaleBf);
            const float bias = toFloat(biasBf);

            out.scales[static_cast<usize>(row * groupsPerRow + group)] = scaleBf;
            out.biases[static_cast<usize>(row * groupsPerRow + group)] = biasBf;

            for (u64 i = begin; i < end; ++i) {
                const float value = values[static_cast<usize>(row * layout.inFeatures + i)];
                u32 quantized = 0;
                if (scale > 0.0f) {
                    const float level = std::round((value - bias) / scale);
                    quantized = static_cast<u32>(
                            std::clamp(level, 0.0f, static_cast<float>(levels)));
                }

                const u64 flatIndex = row * wordsPerRow * perWord + i;
                const u64 word = flatIndex / perWord;
                const u32 shift = static_cast<u32>((flatIndex % perWord) * layout.spec.bits);
                out.packed[static_cast<usize>(word)] |= quantized << shift;
            }
        }
    }
    return out;
}

std::vector<float> dequantGemv(std::span<const u32> packed, std::span<const bf16> scales,
                               std::span<const bf16> biases, std::span<const float> x,
                               const QuantizedLinearLayout& layout) {
    std::vector<float> out(static_cast<usize>(layout.outFeatures), 0.0f);

    for (u64 row = 0; row < layout.outFeatures; ++row) {
        // Accumulate in double so the reference is not itself a source of
        // rounding when compared against an fp32-accumulating kernel.
        double sum = 0.0;
        for (u64 column = 0; column < layout.inFeatures; ++column) {
            const float weight = dequantizeElement(packed, scales, biases, row, column,
                                                   layout.inFeatures, layout.spec);
            sum += static_cast<double>(weight) * static_cast<double>(x[static_cast<usize>(column)]);
        }
        out[static_cast<usize>(row)] = static_cast<float>(sum);
    }
    return out;
}

std::vector<float> embedLookup(std::span<const u32> packed, std::span<const bf16> scales,
                               std::span<const bf16> biases, u32 tokenId,
                               const QuantizedLinearLayout& layout) {
    std::vector<float> out(static_cast<usize>(layout.inFeatures));
    for (u64 column = 0; column < layout.inFeatures; ++column) {
        out[static_cast<usize>(column)] = dequantizeElement(packed, scales, biases, tokenId,
                                                            column, layout.inFeatures,
                                                            layout.spec);
    }

    // Gemma scales embeddings by sqrt(hidden_size) on the way in.
    const auto embedScale = static_cast<float>(std::sqrt(static_cast<double>(layout.inFeatures)));
    for (auto& value : out) {
        value *= embedScale;
    }
    return out;
}

// ---------------------------------------------------------------------------
// Normalization
// ---------------------------------------------------------------------------
namespace {

[[nodiscard]] double meanSquare(std::span<const float> x) {
    double sum = 0.0;
    for (const float value : x) {
        sum += static_cast<double>(value) * static_cast<double>(value);
    }
    return sum / static_cast<double>(x.size());
}

}  // namespace

std::vector<float> rmsNorm(std::span<const float> x, std::span<const float> weight,
                           double eps) {
    const double inverseRms = 1.0 / std::sqrt(meanSquare(x) + eps);

    std::vector<float> out(x.size());
    for (usize i = 0; i < x.size(); ++i) {
        out[i] = static_cast<float>(static_cast<double>(x[i]) * inverseRms *
                                    static_cast<double>(weight[i]));
    }
    return out;
}

std::vector<float> rmsNormNoScale(std::span<const float> x, double eps) {
    const double inverseRms = 1.0 / std::sqrt(meanSquare(x) + eps);

    std::vector<float> out(x.size());
    for (usize i = 0; i < x.size(); ++i) {
        out[i] = static_cast<float>(static_cast<double>(x[i]) * inverseRms);
    }
    return out;
}

// ---------------------------------------------------------------------------
// Rotary embeddings
// ---------------------------------------------------------------------------

void applyRope(std::span<float> head, u64 position, const RopeParams& params) {
    const u64 rotatedPairs = params.rotatedPairs();
    if (rotatedPairs == 0) {
        return;
    }
    // Pairing spans the whole head regardless of how many pairs rotate, and the
    // exponent divides by headDim rather than by the rotated extent.
    const u64 half = params.headDim / 2;

    for (u64 i = 0; i < rotatedPairs; ++i) {
        const double exponent =
                static_cast<double>(2 * i) / static_cast<double>(params.headDim);
        const double frequency = 1.0 / std::pow(params.theta, exponent);
        const double angle = static_cast<double>(position) * frequency;
        const double cosine = std::cos(angle);
        const double sine = std::sin(angle);

        const double lower = head[static_cast<usize>(i)];
        const double upper = head[static_cast<usize>(i + half)];

        head[static_cast<usize>(i)] = static_cast<float>(lower * cosine - upper * sine);
        head[static_cast<usize>(i + half)] = static_cast<float>(lower * sine + upper * cosine);
    }
}

// ---------------------------------------------------------------------------
// Activations
// ---------------------------------------------------------------------------

float geluApprox(float x) {
    // gelu_pytorch_tanh
    constexpr double kSqrt2OverPi = 0.7978845608028654;
    constexpr double kCoefficient = 0.044715;

    const double v = x;
    const double inner = kSqrt2OverPi * (v + kCoefficient * v * v * v);
    return static_cast<float>(0.5 * v * (1.0 + std::tanh(inner)));
}

std::vector<float> geglu(std::span<const float> gate, std::span<const float> up) {
    std::vector<float> out(gate.size());
    for (usize i = 0; i < gate.size(); ++i) {
        out[i] = geluApprox(gate[i]) * up[i];
    }
    return out;
}

// ---------------------------------------------------------------------------
// Attention
// ---------------------------------------------------------------------------

std::vector<float> softmax(std::span<const float> logits) {
    std::vector<float> out(logits.size());
    if (logits.empty()) {
        return out;
    }

    const float maximum = *std::ranges::max_element(logits);

    double total = 0.0;
    for (usize i = 0; i < logits.size(); ++i) {
        const double value = std::exp(static_cast<double>(logits[i] - maximum));
        out[i] = static_cast<float>(value);
        total += value;
    }
    for (auto& value : out) {
        value = static_cast<float>(static_cast<double>(value) / total);
    }
    return out;
}

std::vector<float> decodeAttention(std::span<const float> queries, std::span<const float> keys,
                                   std::span<const float> values, u64 cachedLength,
                                   u64 queryPosition, const AttentionParams& params) {
    const u64 headDim = params.headDim;
    const u64 groupSize = params.numHeads / params.numKVHeads;
    const u64 rows = params.rowsPerHead(cachedLength);

    // A sliding window attends to the most recent `slidingWindow` positions
    // inclusive of the query, so anything older is masked out entirely.
    u64 firstVisible = 0;
    if (params.slidingWindow > 0 && queryPosition + 1 > params.slidingWindow) {
        firstVisible = queryPosition + 1 - params.slidingWindow;
    }
    // A ring also loses anything older than its own capacity, regardless of the
    // window: those rows have already been overwritten.
    if (params.circular && cachedLength > rows) {
        firstVisible = std::max(firstVisible, cachedLength - rows);
    }

    std::vector<float> out(static_cast<usize>(params.numHeads * headDim), 0.0f);

    for (u64 head = 0; head < params.numHeads; ++head) {
        const u64 kvHead = head / groupSize;
        const auto* query = queries.data() + head * headDim;

        std::vector<float> scores;
        std::vector<u64> slots;
        scores.reserve(static_cast<usize>(cachedLength - firstVisible));
        slots.reserve(static_cast<usize>(cachedLength - firstVisible));

        for (u64 position = firstVisible; position < cachedLength; ++position) {
            const u64 slot = params.slotFor(position, cachedLength);
            const auto* key = keys.data() + (kvHead * rows + slot) * headDim;

            double dot = 0.0;
            for (u64 d = 0; d < headDim; ++d) {
                dot += static_cast<double>(query[d]) * static_cast<double>(key[d]);
            }
            scores.push_back(static_cast<float>(dot * static_cast<double>(params.scale)));
            slots.push_back(slot);
        }

        if (scores.empty()) {
            continue;
        }

        const std::vector<float> weights = softmax(scores);

        auto* destination = out.data() + head * headDim;
        for (usize i = 0; i < weights.size(); ++i) {
            const auto* value = values.data() + (kvHead * rows + slots[i]) * headDim;
            const double weight = weights[i];
            for (u64 d = 0; d < headDim; ++d) {
                destination[d] = static_cast<float>(static_cast<double>(destination[d]) +
                                                    weight * static_cast<double>(value[d]));
            }
        }
    }
    return out;
}

std::vector<float> moeCombine(std::span<const float> expertOutputs,
                              std::span<const float> weights, u64 topK, u64 hidden) {
    std::vector<float> out(static_cast<usize>(hidden), 0.0f);
    for (u64 k = 0; k < topK; ++k) {
        const double weight = weights[static_cast<usize>(k)];
        const auto* row = expertOutputs.data() + k * hidden;
        for (u64 d = 0; d < hidden; ++d) {
            out[static_cast<usize>(d)] = static_cast<float>(
                    static_cast<double>(out[static_cast<usize>(d)]) +
                    weight * static_cast<double>(row[static_cast<usize>(d)]));
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// Mixture of experts
// ---------------------------------------------------------------------------

RouterResult routerTopK(std::span<const float> scores, std::span<const float> perExpertScale,
                        u64 topK) {
    const auto count = static_cast<u64>(scores.size());
    const u64 k = std::min(topK, count);

    std::vector<u32> order(static_cast<usize>(count));
    std::iota(order.begin(), order.end(), 0u);

    // Partial sort by descending score. Ties break toward the lower index so
    // selection is deterministic; MLX uses argpartition, which does not
    // guarantee an order, so exact tie behaviour may differ there.
    std::partial_sort(order.begin(), order.begin() + static_cast<isize>(k), order.end(),
                      [&](u32 a, u32 b) {
                          if (scores[a] != scores[b]) {
                              return scores[a] > scores[b];
                          }
                          return a < b;
                      });

    RouterResult result;
    result.indices.assign(order.begin(), order.begin() + static_cast<isize>(k));

    // Softmax over the selected scores only, never over all 128.
    std::vector<float> selected(static_cast<usize>(k));
    for (u64 i = 0; i < k; ++i) {
        selected[static_cast<usize>(i)] = scores[result.indices[static_cast<usize>(i)]];
    }
    result.weights = softmax(selected);

    for (u64 i = 0; i < k; ++i) {
        result.weights[static_cast<usize>(i)] *=
                perExpertScale[result.indices[static_cast<usize>(i)]];
    }
    return result;
}

std::vector<float> foldRouterScale(std::span<const float> routerScale, u64 hiddenSize) {
    const auto rootSize =
            static_cast<float>(1.0 / std::sqrt(static_cast<double>(hiddenSize)));

    std::vector<float> out(routerScale.size());
    for (usize i = 0; i < routerScale.size(); ++i) {
        out[i] = routerScale[i] * rootSize;
    }
    return out;
}

// ---------------------------------------------------------------------------
// Output head
// ---------------------------------------------------------------------------

std::vector<float> logitSoftcap(std::span<const float> logits, float cap) {
    std::vector<float> out(logits.size());
    for (usize i = 0; i < logits.size(); ++i) {
        out[i] = static_cast<float>(std::tanh(static_cast<double>(logits[i]) /
                                              static_cast<double>(cap)) *
                                    static_cast<double>(cap));
    }
    return out;
}

u32 argmax(std::span<const float> values) {
    u32 best = 0;
    for (u32 i = 1; i < values.size(); ++i) {
        if (values[i] > values[best]) {
            best = i;
        }
    }
    return best;
}

}  // namespace tf::reference
