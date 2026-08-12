// The compiled shaders, embedded.
//
// Each header is produced by dxc at build time (see cmake/CompileHLSL.cmake)
// and declares one byte array named after its entry point. The mapping below is
// written out rather than generated: it is the one place that guarantees the
// ShaderId enum and the compiled set agree, and a generated version would
// silently accept a mismatch.

#include "D3D12Common.h"

#include "shaders/Add.h"
#include "shaders/Argmax.h"
#include "shaders/Attention.h"
#include "shaders/DequantGemm4.h"
#include "shaders/DequantGemm8.h"
#include "shaders/DequantGemv4.h"
#include "shaders/DequantGemv8.h"
#include "shaders/EmbedLookup4.h"
#include "shaders/EmbedLookup8.h"
#include "shaders/FillZero.h"
#include "shaders/GatherRows.h"
#include "shaders/Geglu.h"
#include "shaders/KvWrite.h"
#include "shaders/LogitSoftcap.h"
#include "shaders/MoeCombine.h"
#include "shaders/RmsNorm.h"
#include "shaders/RmsNormNoWeight.h"
#include "shaders/Rope.h"
#include "shaders/RouterTopK.h"
#include "shaders/Scale.h"
#include "shaders/ScatterAddRows.h"

namespace tf::gpu::d3d12 {
namespace {

struct Entry {
    const u8* bytes;
    usize size;
    std::string_view name;
};

#define TF_SHADER(id) \
    Entry { g_##id, sizeof(g_##id), #id }

/// Indexed by ShaderId, in enum order. A reordering that breaks the
/// correspondence trips the static_assert below rather than dispatching the
/// wrong shader.
constexpr Entry kShaders[] = {
        TF_SHADER(RmsNorm),        TF_SHADER(RmsNormNoWeight), TF_SHADER(DequantGemv4),
        TF_SHADER(DequantGemv8),   TF_SHADER(DequantGemm4),    TF_SHADER(DequantGemm8),
        TF_SHADER(EmbedLookup4),   TF_SHADER(EmbedLookup8),    TF_SHADER(Rope),
        TF_SHADER(Geglu),          TF_SHADER(Add),             TF_SHADER(Scale),
        TF_SHADER(LogitSoftcap),   TF_SHADER(Argmax),          TF_SHADER(KvWrite),
        TF_SHADER(Attention),      TF_SHADER(RouterTopK),      TF_SHADER(GatherRows),
        TF_SHADER(ScatterAddRows), TF_SHADER(FillZero),        TF_SHADER(MoeCombine),
};

#undef TF_SHADER

static_assert(std::size(kShaders) == static_cast<usize>(ShaderId::Count),
              "every ShaderId needs a compiled shader, in the same order");

}  // namespace

ShaderBlob shaderBlob(ShaderId id) {
    const auto index = static_cast<usize>(id);
    if (index >= std::size(kShaders)) {
        return {};
    }
    return ShaderBlob{.bytes = kShaders[index].bytes, .size = kShaders[index].size};
}

std::string_view shaderName(ShaderId id) {
    const auto index = static_cast<usize>(id);
    return index < std::size(kShaders) ? kShaders[index].name : std::string_view{"?"};
}

}  // namespace tf::gpu::d3d12
