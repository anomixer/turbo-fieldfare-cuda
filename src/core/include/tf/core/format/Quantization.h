#pragma once

#include "tf/core/base/Error.h"
#include "tf/core/base/Types.h"
#include "tf/core/format/DType.h"

namespace tf {

/// MLX affine quantization parameters. The pinned checkpoint uses 4-bit at
/// group 64 throughout, except every `layers.N.router.proj`, which is 8-bit at
/// the same group size.
struct QuantSpec {
    u32 bits = 4;
    u32 groupSize = 64;

    friend constexpr bool operator==(const QuantSpec&, const QuantSpec&) = default;
};

inline constexpr QuantSpec kWeightQuant{.bits = 4, .groupSize = 64};
inline constexpr QuantSpec kRouterQuant{.bits = 8, .groupSize = 64};

/// Byte layout of one MLX affine quantized linear of shape
/// [outFeatures, inFeatures].
///
/// MLX stores three tensors per quantized linear:
///
///   weight  [out, in / (32/bits)]  U32   packed low-nibble-first
///   scales  [out, in / groupSize]  BF16
///   biases  [out, in / groupSize]  BF16
///
/// Dequantization is `value = packed * scale + bias`, per group of `groupSize`
/// contiguous input elements. Note this is an affine (scale+bias) scheme, not
/// symmetric: the bias is an additive term, not a zero-point to subtract.
struct QuantizedLinearLayout {
    u64 outFeatures = 0;
    u64 inFeatures = 0;
    QuantSpec spec;

    /// Quantized values packed into each 32-bit word: 8 at 4-bit, 4 at 8-bit.
    [[nodiscard]] constexpr u64 valuesPerWord() const noexcept {
        return 32 / spec.bits;
    }

    /// Second dimension of the packed weight tensor.
    [[nodiscard]] constexpr u64 packedWordsPerRow() const noexcept {
        return inFeatures / valuesPerWord();
    }

    /// Second dimension of the scales and biases tensors.
    [[nodiscard]] constexpr u64 groupsPerRow() const noexcept {
        return inFeatures / spec.groupSize;
    }

    [[nodiscard]] constexpr u64 weightBytes() const noexcept {
        return outFeatures * packedWordsPerRow() * byteWidth(DType::U32);
    }

    [[nodiscard]] constexpr u64 scaleBytes() const noexcept {
        return outFeatures * groupsPerRow() * byteWidth(DType::BF16);
    }

    /// Biases match scales in shape and dtype.
    [[nodiscard]] constexpr u64 biasBytes() const noexcept { return scaleBytes(); }

    [[nodiscard]] constexpr u64 totalBytes() const noexcept {
        return weightBytes() + scaleBytes() + biasBytes();
    }

    /// Rejects dimensions that do not divide evenly, which would otherwise
    /// produce a layout that silently disagrees with the checkpoint.
    [[nodiscard]] Status validate() const {
        if (spec.bits != 4 && spec.bits != 8) {
            return makeError(ErrorCode::Unsupported,
                             "unsupported quantization width {} bits (expected 4 or 8)",
                             spec.bits);
        }
        if (spec.groupSize == 0) {
            return makeError(ErrorCode::InvalidArgument, "group size must be non-zero");
        }
        if (outFeatures == 0 || inFeatures == 0) {
            return makeError(ErrorCode::InvalidArgument,
                             "quantized linear has a zero dimension");
        }
        if (inFeatures % valuesPerWord() != 0) {
            return makeError(ErrorCode::MalformedData,
                             "inFeatures {} is not a multiple of {} values per 32-bit word",
                             inFeatures, valuesPerWord());
        }
        if (inFeatures % spec.groupSize != 0) {
            return makeError(ErrorCode::MalformedData,
                             "inFeatures {} is not a multiple of group size {}", inFeatures,
                             spec.groupSize);
        }
        return {};
    }
};

}  // namespace tf
