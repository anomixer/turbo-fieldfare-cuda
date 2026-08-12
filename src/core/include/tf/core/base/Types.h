#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

namespace tf {

using u8 = std::uint8_t;
using u16 = std::uint16_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;

using i8 = std::int8_t;
using i16 = std::int16_t;
using i32 = std::int32_t;
using i64 = std::int64_t;

using usize = std::size_t;
using isize = std::ptrdiff_t;

using ByteSpan = std::span<const u8>;
using MutableByteSpan = std::span<u8>;

/// A half-open byte range inside a file. The installer plans transfers in these,
/// and the expert streamer reads them.
struct ByteRange {
    u64 offset = 0;
    u64 length = 0;

    [[nodiscard]] constexpr u64 end() const noexcept { return offset + length; }
    [[nodiscard]] constexpr bool empty() const noexcept { return length == 0; }

    /// True when `other` begins exactly where this range ends, so the two can be
    /// merged into one read. Range coalescing in the streamer depends on this.
    [[nodiscard]] constexpr bool adjacentTo(const ByteRange& other) const noexcept {
        return end() == other.offset;
    }

    friend constexpr bool operator==(const ByteRange&, const ByteRange&) = default;
};

/// Rounds `value` up to the next multiple of `alignment`, which must be a power
/// of two. Expert strides are aligned this way so unbuffered reads can DMA
/// straight into pinned host memory.
[[nodiscard]] constexpr u64 alignUp(u64 value, u64 alignment) noexcept {
    return (value + alignment - 1) & ~(alignment - 1);
}

[[nodiscard]] constexpr bool isAligned(u64 value, u64 alignment) noexcept {
    return (value & (alignment - 1)) == 0;
}

/// Alignment constants for the .gturbo packed expert layout.
namespace align {
/// Minimum stride. FILE_FLAG_NO_BUFFERING requires offsets and lengths to be
/// sector-aligned, and 4 KiB covers every NVMe sector size we expect.
inline constexpr u64 kSector = 4096;
/// Preferred stride. Matches upstream's 2 MiB slot alignment and keeps pinned
/// staging allocations on large-page boundaries.
inline constexpr u64 kLargePage = 2ull * 1024 * 1024;
}  // namespace align

}  // namespace tf
