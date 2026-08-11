#pragma once

#include <cmath>
#include <numbers>
#include <span>
#include <vector>

#include "tf/core/base/Types.h"
#include "tf/core/math/Float.h"

namespace tf::reference {

/// SplitMix64, matching upstream's test generator bit for bit.
///
/// Ported faithfully rather than substituted with std::mt19937 so that a kernel
/// can be fed the same inputs upstream's Swift tests use, which keeps their
/// published tolerances meaningful.
class SplitMix64 {
public:
    explicit constexpr SplitMix64(u64 seed = 0x9E3779B97F4A7C15ull) : state_(seed) {}

    constexpr u64 next() noexcept {
        state_ += 0x9E3779B97F4A7C15ull;
        u64 z = state_;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }

    /// Uniform in [0, 1). Uses the top 53 bits, the usual double construction.
    double nextDouble() noexcept {
        return static_cast<double>(next() >> 11) * (1.0 / 9007199254740992.0);
    }

    /// Uniform in [low, high).
    float nextFloat(float low, float high) noexcept {
        return low + static_cast<float>(nextDouble()) * (high - low);
    }

    /// Standard normal via Box-Muller. Activations are better modelled by a
    /// normal than a uniform, and tolerance behaviour differs between them.
    float nextGaussian() noexcept {
        if (hasSpare_) {
            hasSpare_ = false;
            return spare_;
        }
        // Reject exact zero so the log is finite.
        double u1 = 0.0;
        while (u1 <= 1e-300) {
            u1 = nextDouble();
        }
        const double u2 = nextDouble();
        const double magnitude = std::sqrt(-2.0 * std::log(u1));
        const double angle = 2.0 * std::numbers::pi * u2;

        spare_ = static_cast<float>(magnitude * std::sin(angle));
        hasSpare_ = true;
        return static_cast<float>(magnitude * std::cos(angle));
    }

    u32 nextBelow(u32 bound) noexcept {
        return bound == 0 ? 0 : static_cast<u32>(next() % bound);
    }

private:
    u64 state_;
    float spare_ = 0.0f;
    bool hasSpare_ = false;
};

/// Derives independent streams from one root seed, so adding a tensor to a test
/// does not perturb the values every other tensor receives.
class SeedTree {
public:
    explicit constexpr SeedTree(u64 root) : root_(root) {}

    [[nodiscard]] SplitMix64 stream(u64 label) const {
        SplitMix64 mixer{root_ ^ (label * 0x9E3779B97F4A7C15ull)};
        return SplitMix64{mixer.next()};
    }

private:
    u64 root_;
};

// ---- Buffer fills ---------------------------------------------------------

[[nodiscard]] inline std::vector<float> randomFloats(SplitMix64& rng, usize count,
                                                     float low = -1.0f, float high = 1.0f) {
    std::vector<float> values(count);
    for (auto& value : values) {
        value = rng.nextFloat(low, high);
    }
    return values;
}

[[nodiscard]] inline std::vector<float> randomGaussians(SplitMix64& rng, usize count,
                                                        float scale = 1.0f) {
    std::vector<float> values(count);
    for (auto& value : values) {
        value = rng.nextGaussian() * scale;
    }
    return values;
}

/// Values quantized through bf16 and back, so a reference computation sees
/// exactly the values a kernel reading bf16 storage would see.
[[nodiscard]] inline std::vector<float> randomBf16Floats(SplitMix64& rng, usize count,
                                                         float low = -1.0f,
                                                         float high = 1.0f) {
    std::vector<float> values(count);
    for (auto& value : values) {
        value = toFloat(toBf16(rng.nextFloat(low, high)));
    }
    return values;
}

[[nodiscard]] inline std::vector<bf16> toBf16Buffer(std::span<const float> values) {
    std::vector<bf16> out(values.size());
    for (usize i = 0; i < values.size(); ++i) {
        out[i] = toBf16(values[i]);
    }
    return out;
}

[[nodiscard]] inline std::vector<fp16> toFp16Buffer(std::span<const float> values) {
    std::vector<fp16> out(values.size());
    for (usize i = 0; i < values.size(); ++i) {
        out[i] = toFp16(values[i]);
    }
    return out;
}

[[nodiscard]] inline std::vector<float> fromFp16Buffer(std::span<const fp16> values) {
    std::vector<float> out(values.size());
    for (usize i = 0; i < values.size(); ++i) {
        out[i] = toFloat(values[i]);
    }
    return out;
}

}  // namespace tf::reference
