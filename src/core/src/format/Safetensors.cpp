#include "tf/core/format/Safetensors.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstring>
#include <ranges>

#include "tf/core/io/File.h"
#include "tf/core/json/Json.h"

namespace tf {

Result<DType> parseDType(std::string_view name) {
    static constexpr std::pair<std::string_view, DType> kTable[] = {
            {"BF16", DType::BF16}, {"F16", DType::F16},   {"F32", DType::F32},
            {"F64", DType::F64},   {"U8", DType::U8},     {"I8", DType::I8},
            {"U16", DType::U16},   {"I16", DType::I16},   {"U32", DType::U32},
            {"I32", DType::I32},   {"U64", DType::U64},   {"I64", DType::I64},
            {"BOOL", DType::Bool},
    };
    for (const auto& [text, type] : kTable) {
        if (text == name) {
            return type;
        }
    }
    return makeError(ErrorCode::Unsupported, "unknown safetensors dtype '{}'", name);
}

std::string TensorEntry::shapeString() const {
    std::string out = "[";
    for (usize i = 0; i < shape.size(); ++i) {
        if (i > 0) {
            out += ',';
        }
        out += std::to_string(shape[i]);
    }
    out += ']';
    return out;
}

// ---------------------------------------------------------------------------
// SafetensorsHeader
// ---------------------------------------------------------------------------

Result<u64> SafetensorsHeader::readHeaderLength(ByteSpan prefix) {
    if (prefix.size() < kLengthPrefixBytes) {
        return makeError(ErrorCode::MalformedData,
                         "safetensors file shorter than the {}-byte length prefix",
                         kLengthPrefixBytes);
    }

    u64 length = 0;
    std::memcpy(&length, prefix.data(), sizeof(length));
    if constexpr (std::endian::native == std::endian::big) {
        length = std::byteswap(length);
    }

    if (length == 0) {
        return makeError(ErrorCode::MalformedData, "safetensors header length is zero");
    }
    if (length > kMaxHeaderBytes) {
        return makeError(ErrorCode::MalformedData,
                         "safetensors header length {} exceeds the {} byte sanity limit",
                         length, kMaxHeaderBytes);
    }
    return length;
}

Result<SafetensorsHeader> SafetensorsHeader::parse(std::string_view headerJson,
                                                   u64 headerLength) {
    TF_TRY(const json::Value root, json::parse(headerJson));
    TF_TRY(const json::Object* object, root.asObject());

    SafetensorsHeader header;
    header.headerLength_ = headerLength;
    header.tensors_.reserve(object->size());

    for (const auto& [name, entryValue] : *object) {
        if (name == "__metadata__") {
            // Free-form and optional; a non-string value is skipped rather than
            // treated as corruption.
            if (const auto meta = entryValue.asObject(); meta.has_value()) {
                for (const auto& [key, value] : **meta) {
                    if (const auto text = value.asString(); text.has_value()) {
                        header.metadata_.emplace_back(key, std::string{*text});
                    }
                }
            }
            continue;
        }

        auto describe = [&name](const Error& error) { return error.wrap(name); };

        const auto dtypeValue = entryValue.at("dtype");
        if (!dtypeValue) {
            return std::unexpected(describe(dtypeValue.error()));
        }
        const auto dtypeName = (*dtypeValue)->asString();
        if (!dtypeName) {
            return std::unexpected(describe(dtypeName.error()));
        }
        const auto dtype = parseDType(*dtypeName);
        if (!dtype) {
            return std::unexpected(describe(dtype.error()));
        }

        const auto shapeValue = entryValue.at("shape");
        if (!shapeValue) {
            return std::unexpected(describe(shapeValue.error()));
        }
        const auto shapeArray = (*shapeValue)->asArray();
        if (!shapeArray) {
            return std::unexpected(describe(shapeArray.error()));
        }

        TensorEntry entry;
        entry.name = name;
        entry.dtype = *dtype;
        entry.shape.reserve((*shapeArray)->size());
        for (const auto& dim : **shapeArray) {
            const auto value = dim.asUInt();
            if (!value) {
                return std::unexpected(describe(value.error()));
            }
            entry.shape.push_back(*value);
        }

        const auto offsetsValue = entryValue.at("data_offsets");
        if (!offsetsValue) {
            return std::unexpected(describe(offsetsValue.error()));
        }
        const auto offsets = (*offsetsValue)->asArray();
        if (!offsets) {
            return std::unexpected(describe(offsets.error()));
        }
        if ((*offsets)->size() != 2) {
            return makeError(ErrorCode::MalformedData,
                             "{}: data_offsets must hold exactly 2 values, found {}", name,
                             (*offsets)->size());
        }

        const auto begin = (**offsets)[0].asUInt();
        if (!begin) {
            return std::unexpected(describe(begin.error()));
        }
        const auto end = (**offsets)[1].asUInt();
        if (!end) {
            return std::unexpected(describe(end.error()));
        }
        if (*end < *begin) {
            return makeError(ErrorCode::MalformedData,
                             "{}: data_offsets end {} precedes begin {}", name, *end, *begin);
        }

        entry.dataRange = ByteRange{.offset = *begin, .length = *end - *begin};

        if (entry.dataRange.length != entry.expectedBytes()) {
            return makeError(ErrorCode::MalformedData,
                             "{}: shape {} of {} implies {} bytes but data_offsets span {}",
                             name, entry.shapeString(), toString(entry.dtype),
                             entry.expectedBytes(), entry.dataRange.length);
        }

        header.tensors_.push_back(std::move(entry));
    }

    return header;
}

const TensorEntry* SafetensorsHeader::find(std::string_view name) const {
    for (const auto& entry : tensors_) {
        if (entry.name == name) {
            return &entry;
        }
    }
    return nullptr;
}

Result<const TensorEntry*> SafetensorsHeader::require(std::string_view name) const {
    if (const TensorEntry* entry = find(name)) {
        return entry;
    }
    return makeError(ErrorCode::NotFound, "tensor '{}' not present in safetensors header", name);
}

Result<ByteRange> SafetensorsHeader::fileRange(std::string_view name) const {
    TF_TRY(const TensorEntry* entry, require(name));
    return ByteRange{.offset = dataOffset() + entry->dataRange.offset,
                     .length = entry->dataRange.length};
}

// ---------------------------------------------------------------------------
// SafetensorsIndex
// ---------------------------------------------------------------------------

Result<SafetensorsIndex> SafetensorsIndex::parse(std::string_view text) {
    TF_TRY(const json::Value root, json::parse(text));

    SafetensorsIndex index;

    if (const json::Value* metadata = root.find("metadata")) {
        if (const json::Value* total = metadata->find("total_size")) {
            if (const auto value = total->asUInt(); value.has_value()) {
                index.totalSize_ = *value;
            }
        }
    }

    TF_TRY(const json::Value* weightMap, root.at("weight_map"));
    TF_TRY(const json::Object* entries, weightMap->asObject());

    index.weightMap_.reserve(entries->size());
    for (const auto& [tensorName, shardValue] : *entries) {
        const auto shard = shardValue.asString();
        if (!shard) {
            return std::unexpected(shard.error().wrap(tensorName));
        }
        index.weightMap_.emplace_back(tensorName, std::string{*shard});
    }

    // Distinct shards, first-appearance order, so callers can open each file
    // once instead of per tensor. Linear scan is fine: a checkpoint has a
    // handful of shards, not thousands.
    for (const auto& shard : index.weightMap_ | std::views::values) {
        if (std::ranges::find(index.shardFiles_, shard) == index.shardFiles_.end()) {
            index.shardFiles_.push_back(shard);
        }
    }

    return index;
}

const std::string* SafetensorsIndex::shardFor(std::string_view tensorName) const {
    for (const auto& [name, shard] : weightMap_) {
        if (name == tensorName) {
            return &shard;
        }
    }
    return nullptr;
}

Result<std::string_view> SafetensorsIndex::requireShardFor(std::string_view tensorName) const {
    if (const std::string* shard = shardFor(tensorName)) {
        return std::string_view{*shard};
    }
    return makeError(ErrorCode::NotFound, "tensor '{}' not present in the index", tensorName);
}

// ---------------------------------------------------------------------------
// File-backed entry points
// ---------------------------------------------------------------------------

Result<SafetensorsHeader> readSafetensorsHeader(const std::filesystem::path& path) {
    TF_TRY(io::File file, io::File::openRead(path, io::ReadMode::Sequential));

    std::array<u8, SafetensorsHeader::kLengthPrefixBytes> prefix{};
    TF_CHECK(file.readExactAt(0, prefix));

    auto headerLength = SafetensorsHeader::readHeaderLength(prefix);
    if (!headerLength) {
        return std::unexpected(headerLength.error().wrap(path.string()));
    }

    TF_TRY(const u64 fileSize, file.size());
    if (SafetensorsHeader::kLengthPrefixBytes + *headerLength > fileSize) {
        return makeError(ErrorCode::MalformedData,
                         "{}: header claims {} bytes but the file is only {} bytes",
                         path.string(), *headerLength, fileSize);
    }

    std::string body(static_cast<usize>(*headerLength), '\0');
    TF_CHECK(file.readExactAt(SafetensorsHeader::kLengthPrefixBytes,
                              MutableByteSpan{reinterpret_cast<u8*>(body.data()), body.size()}));

    auto header = SafetensorsHeader::parse(body, *headerLength);
    if (!header) {
        return std::unexpected(header.error().wrap(path.string()));
    }
    return header;
}

Result<SafetensorsIndex> readSafetensorsIndex(const std::filesystem::path& checkpointDir) {
    const std::filesystem::path path = checkpointDir / "model.safetensors.index.json";
    TF_TRY(const std::string text, io::readTextFile(path));

    auto index = SafetensorsIndex::parse(text);
    if (!index) {
        return std::unexpected(index.error().wrap(path.string()));
    }
    return index;
}

}  // namespace tf
