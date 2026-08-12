#include <catch2/catch_test_macros.hpp>

#include <array>
#include <bit>
#include <cmath>
#include <limits>

#include "tf/core/math/Float.h"

using namespace tf;

TEST_CASE("bf16 round-trips exactly representable values", "[float]") {
    for (const float value : {0.0f, 1.0f, -1.0f, 2.0f, 0.5f, 30.0f, 1.0234375f, -0.0703125f}) {
        INFO("value: " << value);
        CHECK(toFloat(toBf16(value)) == value);
    }
}

TEST_CASE("bf16 widening is a shift of the upper half", "[float]") {
    // 1.0f is 0x3F800000, so bf16 is 0x3F80.
    CHECK(toBf16(1.0f).bits == 0x3F80);
    CHECK(toFloat(bf16{0x3F80}) == 1.0f);

    // The constant q_norm weight observed in the real checkpoint.
    CHECK(toFloat(bf16{0x3F83}) == 1.0234375f);
}

TEST_CASE("bf16 rounds to nearest even, not toward zero", "[float]") {
    // Truncation would bias every scale and bias downward, which compounds
    // across 30 layers. A value just above a representable point must round up.
    const float justAbove = std::bit_cast<float>(0x3F808000u | 0x1u);
    CHECK(toFloat(toBf16(justAbove)) > 1.0f);

    // An exact halfway case rounds to the even neighbour.
    const float halfway = std::bit_cast<float>(0x3F808000u);  // between 0x3F80 and 0x3F81
    CHECK(toBf16(halfway).bits == 0x3F80);

    const float halfwayUp = std::bit_cast<float>(0x3F818000u);  // between 0x3F81 and 0x3F82
    CHECK(toBf16(halfwayUp).bits == 0x3F82);
}

TEST_CASE("bf16 preserves sign, zero, infinity and NaN", "[float]") {
    CHECK(toFloat(toBf16(0.0f)) == 0.0f);
    CHECK(std::signbit(toFloat(toBf16(-0.0f))));
    CHECK(std::isinf(toFloat(toBf16(std::numeric_limits<float>::infinity()))));
    CHECK(std::isnan(toFloat(toBf16(std::numeric_limits<float>::quiet_NaN()))));
}

TEST_CASE("bf16 approximates arbitrary values within its precision", "[float]") {
    // 8 mantissa bits gives a relative step under 2^-8.
    for (const float value : {3.14159f, -2.71828f, 1e-4f, 1e6f, 0.0703f, 4.4846f}) {
        const float recovered = toFloat(toBf16(value));
        INFO("value: " << value << " recovered: " << recovered);
        CHECK(std::abs(recovered - value) <= std::abs(value) * 0.004f);
    }
}

TEST_CASE("fp16 round-trips exactly representable values", "[float]") {
    for (const float value : {0.0f, 1.0f, -1.0f, 2.0f, 0.5f, -0.25f, 1024.0f}) {
        INFO("value: " << value);
        CHECK(toFloat(toFp16(value)) == value);
    }
}

TEST_CASE("fp16 handles subnormals, overflow and underflow", "[float]") {
    // Smallest positive normal is 2^-14; subnormals go down to 2^-24.
    const float smallestNormal = std::ldexp(1.0f, -14);
    CHECK(toFloat(toFp16(smallestNormal)) == smallestNormal);

    const float subnormal = std::ldexp(1.0f, -20);
    CHECK(toFloat(toFp16(subnormal)) == subnormal);

    const float smallestSubnormal = std::ldexp(1.0f, -24);
    CHECK(toFloat(toFp16(smallestSubnormal)) == smallestSubnormal);

    // Below the subnormal range flushes to a correctly signed zero.
    CHECK(toFloat(toFp16(std::ldexp(1.0f, -30))) == 0.0f);
    CHECK(std::signbit(toFloat(toFp16(-std::ldexp(1.0f, -30)))));

    // Above the fp16 maximum saturates to infinity.
    CHECK(std::isinf(toFloat(toFp16(1e6f))));
    CHECK(toFloat(toFp16(65504.0f)) == 65504.0f);  // the fp16 maximum
}

TEST_CASE("fp16 preserves infinity and NaN", "[float]") {
    CHECK(std::isinf(toFloat(toFp16(std::numeric_limits<float>::infinity()))));
    CHECK(toFloat(toFp16(-std::numeric_limits<float>::infinity())) < 0.0f);
    CHECK(std::isnan(toFloat(toFp16(std::numeric_limits<float>::quiet_NaN()))));
}

TEST_CASE("fp16 rounds to nearest across the whole normal range", "[float]") {
    // A dense sweep catches exponent-boundary mistakes that spot checks miss.
    for (int exponent = -14; exponent < 15; ++exponent) {
        for (int step = 0; step < 32; ++step) {
            const float value =
                    std::ldexp(1.0f + static_cast<float>(step) / 32.0f, exponent);
            const float recovered = toFloat(toFp16(value));
            INFO("exponent " << exponent << " step " << step << " value " << value);
            // fp16 has 10 mantissa bits, so relative error is under 2^-11.
            REQUIRE(std::abs(recovered - value) <= std::abs(value) * 0.0005f);
        }
    }
}

TEST_CASE("byte spans reinterpret as typed element spans", "[float]") {
    const std::array<u8, 8> bytes{0x80, 0x3F, 0x00, 0x40, 0x00, 0x00, 0x80, 0x3F};

    const auto asHalves = asBf16(ByteSpan{bytes.data(), bytes.size()});
    REQUIRE(asHalves.size() == 4);
    CHECK(toFloat(asHalves[0]) == 1.0f);  // 0x3F80
    CHECK(toFloat(asHalves[1]) == 2.0f);  // 0x4000

    const auto asWords = asU32(ByteSpan{bytes.data(), bytes.size()});
    REQUIRE(asWords.size() == 2);
    CHECK(asWords[0] == 0x40003F80u);
}
