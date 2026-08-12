#pragma once

#include <bit>
#include <cmath>
#include <cstring>
#include <vector>

#include "tf/core/base/Types.h"

namespace tf {

/// Storage-only bfloat16: the upper 16 bits of an IEEE-754 float. All the
/// checkpoint's scales, biases and norm weights are stored this way.
struct bf16 {
    u16 bits = 0;

    friend constexpr bool operator==(const bf16&, const bf16&) = default;
};

/// Storage-only IEEE-754 half. The KV cache and activations use this.
struct fp16 {
    u16 bits = 0;

    friend constexpr bool operator==(const fp16&, const fp16&) = default;
};

// ---------------------------------------------------------------------------
// bfloat16
// ---------------------------------------------------------------------------

[[nodiscard]] inline float toFloat(bf16 value) noexcept {
    // bf16 shares the exponent layout of float, so widening is a shift.
    const u32 widened = static_cast<u32>(value.bits) << 16;
    return std::bit_cast<float>(widened);
}

/// Round-to-nearest-even, matching what PyTorch and MLX emit. Truncation would
/// introduce a consistent downward bias that shows up as drift over 30 layers.
[[nodiscard]] inline bf16 toBf16(float value) noexcept {
    const u32 bits = std::bit_cast<u32>(value);

    // Propagate NaN rather than letting rounding turn it into an infinity.
    if ((bits & 0x7F800000u) == 0x7F800000u && (bits & 0x007FFFFFu) != 0) {
        return bf16{static_cast<u16>((bits >> 16) | 0x0040u)};
    }

    // Add half an ulp, biased by the low bit of the retained mantissa so that
    // exact ties round to even.
    const u32 roundBit = (bits >> 16) & 1u;
    const u32 rounded = bits + 0x7FFFu + roundBit;
    return bf16{static_cast<u16>(rounded >> 16)};
}

// ---------------------------------------------------------------------------
// float16
// ---------------------------------------------------------------------------

[[nodiscard]] inline float toFloat(fp16 value) noexcept {
    const u32 sign = static_cast<u32>(value.bits & 0x8000u) << 16;
    u32 exponent = (value.bits >> 10) & 0x1Fu;
    u32 mantissa = value.bits & 0x03FFu;

    if (exponent == 0) {
        if (mantissa == 0) {
            return std::bit_cast<float>(sign);  // signed zero
        }
        // Subnormal: renormalize into float's wider exponent range.
        exponent = 1;
        while ((mantissa & 0x0400u) == 0) {
            mantissa <<= 1;
            --exponent;
        }
        mantissa &= 0x03FFu;
        const u32 bits = sign | ((exponent + (127 - 15)) << 23) | (mantissa << 13);
        return std::bit_cast<float>(bits);
    }
    if (exponent == 0x1F) {
        // Infinity or NaN.
        const u32 bits = sign | 0x7F800000u | (mantissa << 13);
        return std::bit_cast<float>(bits);
    }

    const u32 bits = sign | ((exponent + (127 - 15)) << 23) | (mantissa << 13);
    return std::bit_cast<float>(bits);
}

[[nodiscard]] inline fp16 toFp16(float value) noexcept {
    const u32 bits = std::bit_cast<u32>(value);
    const u32 sign = (bits >> 16) & 0x8000u;
    const i32 exponent = static_cast<i32>((bits >> 23) & 0xFFu) - 127 + 15;
    const u32 mantissa = bits & 0x007FFFFFu;

    if (((bits >> 23) & 0xFFu) == 0xFFu) {
        // Infinity, or a NaN whose payload must stay non-zero.
        const u32 payload = mantissa != 0 ? 0x0200u : 0u;
        return fp16{static_cast<u16>(sign | 0x7C00u | payload)};
    }
    if (exponent >= 0x1F) {
        return fp16{static_cast<u16>(sign | 0x7C00u)};  // overflow to infinity
    }
    if (exponent <= 0) {
        if (exponent < -10) {
            return fp16{static_cast<u16>(sign)};  // underflow to signed zero
        }
        // Subnormal: shift the implicit leading one back into the mantissa.
        const u32 withImplicit = mantissa | 0x00800000u;
        const u32 shift = static_cast<u32>(14 - exponent);
        const u32 rounded =
                (withImplicit + (1u << (shift - 1)) - 1 + ((withImplicit >> shift) & 1u)) >>
                shift;
        return fp16{static_cast<u16>(sign | rounded)};
    }

    // Round-to-nearest-even on the 13 discarded mantissa bits.
    const u32 roundBit = (mantissa >> 13) & 1u;
    const u32 rounded = mantissa + 0x0FFFu + roundBit;
    if ((rounded & 0x00800000u) != 0) {
        // Rounding carried into the exponent.
        return fp16{static_cast<u16>(sign | ((exponent + 1) << 10))};
    }
    return fp16{static_cast<u16>(sign | (static_cast<u32>(exponent) << 10) | (rounded >> 13))};
}

// ---------------------------------------------------------------------------
// Typed views over raw model bytes
// ---------------------------------------------------------------------------

/// Reinterprets a byte span as bf16 elements. The checkpoint stores scales and
/// biases little-endian, which matches x86 layout, so this is a cast rather
/// than a conversion.
[[nodiscard]] inline std::span<const bf16> asBf16(ByteSpan bytes) noexcept {
    return {reinterpret_cast<const bf16*>(bytes.data()), bytes.size() / sizeof(bf16)};
}

[[nodiscard]] inline std::span<const fp16> asFp16(ByteSpan bytes) noexcept {
    return {reinterpret_cast<const fp16*>(bytes.data()), bytes.size() / sizeof(fp16)};
}

[[nodiscard]] inline std::span<const u32> asU32(ByteSpan bytes) noexcept {
    return {reinterpret_cast<const u32*>(bytes.data()), bytes.size() / sizeof(u32)};
}

// These views borrow; they never own. Handing one an owning temporary - most
// easily `asU32(readSomething())` - yields a span into storage that dies at the
// end of the expression. Deleting the rvalue overloads turns that into a
// compile error instead of a use-after-free that reads plausible garbage.
std::span<const bf16> asBf16(const std::vector<u8>&&) = delete;
std::span<const fp16> asFp16(const std::vector<u8>&&) = delete;
std::span<const u32> asU32(const std::vector<u8>&&) = delete;

}  // namespace tf
