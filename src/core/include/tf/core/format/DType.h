#pragma once

#include <string_view>

#include "tf/core/base/Error.h"
#include "tf/core/base/Types.h"

namespace tf {

/// Element types that appear in safetensors headers. The pinned checkpoint only
/// uses BF16 (scales, biases, norms) and U32 (packed quantized weights), but the
/// full set is recognised so an unexpected dtype produces a clear error rather
/// than a silent misread.
enum class DType {
    BF16,
    F16,
    F32,
    F64,
    U8,
    I8,
    U16,
    I16,
    U32,
    I32,
    U64,
    I64,
    Bool,
};

[[nodiscard]] constexpr u64 byteWidth(DType type) noexcept {
    switch (type) {
        case DType::U8:
        case DType::I8:
        case DType::Bool:
            return 1;
        case DType::BF16:
        case DType::F16:
        case DType::U16:
        case DType::I16:
            return 2;
        case DType::F32:
        case DType::U32:
        case DType::I32:
            return 4;
        case DType::F64:
        case DType::U64:
        case DType::I64:
            return 8;
    }
    return 0;
}

[[nodiscard]] constexpr std::string_view toString(DType type) noexcept {
    switch (type) {
        case DType::BF16: return "BF16";
        case DType::F16:  return "F16";
        case DType::F32:  return "F32";
        case DType::F64:  return "F64";
        case DType::U8:   return "U8";
        case DType::I8:   return "I8";
        case DType::U16:  return "U16";
        case DType::I16:  return "I16";
        case DType::U32:  return "U32";
        case DType::I32:  return "I32";
        case DType::U64:  return "U64";
        case DType::I64:  return "I64";
        case DType::Bool: return "BOOL";
    }
    return "?";
}

[[nodiscard]] Result<DType> parseDType(std::string_view name);

}  // namespace tf
