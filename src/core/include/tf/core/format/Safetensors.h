#pragma once

#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

#include "tf/core/base/Error.h"
#include "tf/core/base/Types.h"
#include "tf/core/format/DType.h"

namespace tf {

/// One tensor described by a safetensors header.
struct TensorEntry {
    std::string name;
    DType dtype = DType::BF16;
    std::vector<u64> shape;
    /// Offsets are relative to the start of the data section, not the file.
    /// Add SafetensorsHeader::dataOffset() for an absolute file position.
    ByteRange dataRange;

    [[nodiscard]] u64 elementCount() const noexcept {
        u64 count = 1;
        for (const u64 dim : shape) {
            count *= dim;
        }
        return count;
    }

    /// Bytes implied by shape and dtype, which must equal dataRange.length.
    [[nodiscard]] u64 expectedBytes() const noexcept {
        return elementCount() * byteWidth(dtype);
    }

    [[nodiscard]] std::string shapeString() const;
};

/// Parsed safetensors header.
///
/// File layout is: 8-byte little-endian header length, then that many bytes of
/// UTF-8 JSON, then the tensor data. The header maps each tensor name to its
/// dtype, shape and `data_offsets` pair.
class SafetensorsHeader {
public:
    /// Number of leading bytes holding the header length.
    static constexpr u64 kLengthPrefixBytes = 8;

    /// Guards against a corrupt length prefix causing a huge allocation. The
    /// largest header in the pinned checkpoint is ~90 KB.
    static constexpr u64 kMaxHeaderBytes = 256ull * 1024 * 1024;

    /// Reads the header length from the first 8 bytes of a file.
    [[nodiscard]] static Result<u64> readHeaderLength(ByteSpan prefix);

    /// Parses the JSON header body, excluding the 8-byte length prefix.
    /// `headerLength` is needed to compute dataOffset().
    [[nodiscard]] static Result<SafetensorsHeader> parse(std::string_view headerJson,
                                                         u64 headerLength);

    /// Absolute file offset where the tensor data section begins.
    [[nodiscard]] u64 dataOffset() const noexcept {
        return kLengthPrefixBytes + headerLength_;
    }

    [[nodiscard]] const std::vector<TensorEntry>& tensors() const noexcept {
        return tensors_;
    }

    /// Returns nullptr when the tensor is absent.
    [[nodiscard]] const TensorEntry* find(std::string_view name) const;

    /// Like find() but reports absence as an error, for required tensors.
    [[nodiscard]] Result<const TensorEntry*> require(std::string_view name) const;

    /// Absolute file byte range of a tensor, i.e. its dataRange shifted by
    /// dataOffset().
    [[nodiscard]] Result<ByteRange> fileRange(std::string_view name) const;

    /// Free-form `__metadata__` entries, if the file carried any.
    [[nodiscard]] const std::vector<std::pair<std::string, std::string>>& metadata()
            const noexcept {
        return metadata_;
    }

private:
    u64 headerLength_ = 0;
    std::vector<TensorEntry> tensors_;
    std::vector<std::pair<std::string, std::string>> metadata_;
};

/// The `model.safetensors.index.json` sidecar: maps each tensor name to the
/// shard file that holds it.
class SafetensorsIndex {
public:
    [[nodiscard]] static Result<SafetensorsIndex> parse(std::string_view json);

    /// Returns nullptr when the tensor is not in the index.
    [[nodiscard]] const std::string* shardFor(std::string_view tensorName) const;

    [[nodiscard]] Result<std::string_view> requireShardFor(std::string_view tensorName) const;

    /// Distinct shard filenames, in first-appearance order.
    [[nodiscard]] const std::vector<std::string>& shardFiles() const noexcept {
        return shardFiles_;
    }

    [[nodiscard]] const std::vector<std::pair<std::string, std::string>>& weightMap()
            const noexcept {
        return weightMap_;
    }

    /// `metadata.total_size` if present, else 0.
    [[nodiscard]] u64 totalSize() const noexcept { return totalSize_; }

private:
    std::vector<std::pair<std::string, std::string>> weightMap_;
    std::vector<std::string> shardFiles_;
    u64 totalSize_ = 0;
};

/// Opens a .safetensors file and parses only its header, reading the length
/// prefix and JSON body without touching the tensor data. The pinned
/// checkpoint's headers are 64-90 KB against 5 GB shards, so this is cheap.
[[nodiscard]] Result<SafetensorsHeader> readSafetensorsHeader(
        const std::filesystem::path& path);

/// Reads and parses `model.safetensors.index.json` from a checkpoint directory.
[[nodiscard]] Result<SafetensorsIndex> readSafetensorsIndex(
        const std::filesystem::path& checkpointDir);

}  // namespace tf
