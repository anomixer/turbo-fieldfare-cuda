#pragma once

#include <algorithm>
#include <cmath>
#include <format>
#include <span>
#include <string>

#include "tf/core/base/Types.h"

namespace tf::reference {

/// Summary of how far a candidate buffer strays from a reference.
struct Deviation {
    double maxAbsolute = 0.0;
    double maxRelative = 0.0;
    /// Index where maxRelative occurred, for a message that points somewhere.
    usize worstIndex = 0;
    double referenceAtWorst = 0.0;
    double candidateAtWorst = 0.0;
    /// Set when either buffer holds a NaN or infinity the other does not.
    bool hasNonFinite = false;

    [[nodiscard]] std::string describe() const {
        if (hasNonFinite) {
            return std::format("non-finite value at index {} (reference {}, candidate {})",
                               worstIndex, referenceAtWorst, candidateAtWorst);
        }
        return std::format(
                "max |abs| {:.3e}, max |rel| {:.3e} at index {} (reference {:.6g}, "
                "candidate {:.6g})",
                maxAbsolute, maxRelative, worstIndex, referenceAtWorst, candidateAtWorst);
    }
};

/// Relative error uses `|a - b| / max(|reference|, floor)` rather than dividing
/// by the reference directly: near-zero reference values would otherwise report
/// enormous relative error for a numerically irrelevant absolute difference.
[[nodiscard]] inline Deviation compare(std::span<const float> reference,
                                       std::span<const float> candidate,
                                       double relativeFloor = 1e-3) {
    Deviation deviation;
    const usize count = std::min(reference.size(), candidate.size());

    for (usize i = 0; i < count; ++i) {
        const double a = reference[i];
        const double b = candidate[i];

        if (!std::isfinite(a) || !std::isfinite(b)) {
            if (!(std::isnan(a) && std::isnan(b)) && a != b) {
                deviation.hasNonFinite = true;
                deviation.worstIndex = i;
                deviation.referenceAtWorst = a;
                deviation.candidateAtWorst = b;
                return deviation;
            }
            continue;
        }

        const double absolute = std::abs(a - b);
        deviation.maxAbsolute = std::max(deviation.maxAbsolute, absolute);

        const double relative = absolute / std::max(std::abs(a), relativeFloor);
        if (relative > deviation.maxRelative) {
            deviation.maxRelative = relative;
            deviation.worstIndex = i;
            deviation.referenceAtWorst = a;
            deviation.candidateAtWorst = b;
        }
    }
    return deviation;
}

/// Per-operation error budgets.
///
/// These are not arbitrary: each reflects the accumulation the operation
/// actually performs. A GEMV over 2816 inputs in fp16 accumulates far more
/// rounding than an elementwise norm, and 4-bit dequantization has a hard
/// quantization floor no amount of care removes.
namespace tolerance {

/// Elementwise fp16 arithmetic: one rounding step.
inline constexpr double kElementwise = 2e-3;

/// RMSNorm and RoPE: a reduction over the hidden dimension, then a scale.
inline constexpr double kNorm = 4e-3;

/// Dequantized GEMV. Dominated by 4-bit affine quantization error rather than
/// by the accumulation.
inline constexpr double kGemv = 1e-2;

/// Attention: two reductions plus a softmax, over up to 4K positions.
inline constexpr double kAttention = 1e-2;

/// Softmax and other normalized outputs, where values are bounded to [0, 1] and
/// absolute error is the meaningful measure.
inline constexpr double kSoftmax = 2e-3;

/// A full expert or MLP: GEMV, GELU, then another GEMV.
inline constexpr double kFeedForward = 2e-2;

}  // namespace tolerance

}  // namespace tf::reference
