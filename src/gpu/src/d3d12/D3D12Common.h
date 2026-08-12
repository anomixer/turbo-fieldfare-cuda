#pragma once

#include <windows.h>

#include <d3d12.h>
#include <dxgi1_6.h>

#include <format>
#include <string>
#include <utility>

#include "tf/core/base/Error.h"
#include "tf/core/base/Types.h"

/// Shared plumbing for the D3D12 backend.
///
/// The second IGpuBackend implementation. It exists to reach AMD and Intel, and
/// to open the door to DirectStorage, which is the real Windows analogue of the
/// pread path the Mac original was built around.
///
/// Nothing above tf::gpu knows this file exists; the seam is what M4 was for.
namespace tf::gpu::d3d12 {

/// A COM pointer with just the operations this backend uses. ATL and WRL both
/// pull in more than a static library in this project should depend on.
template <class T>
class ComPtr {
public:
    ComPtr() = default;
    ~ComPtr() { reset(); }

    ComPtr(const ComPtr& other) : pointer_(other.pointer_) {
        if (pointer_ != nullptr) {
            pointer_->AddRef();
        }
    }
    ComPtr& operator=(const ComPtr& other) {
        if (this != &other) {
            if (other.pointer_ != nullptr) {
                other.pointer_->AddRef();
            }
            reset();
            pointer_ = other.pointer_;
        }
        return *this;
    }

    ComPtr(ComPtr&& other) noexcept : pointer_(std::exchange(other.pointer_, nullptr)) {}
    ComPtr& operator=(ComPtr&& other) noexcept {
        if (this != &other) {
            reset();
            pointer_ = std::exchange(other.pointer_, nullptr);
        }
        return *this;
    }

    void reset() {
        if (pointer_ != nullptr) {
            pointer_->Release();
            pointer_ = nullptr;
        }
    }

    [[nodiscard]] T* get() const noexcept { return pointer_; }
    [[nodiscard]] T* operator->() const noexcept { return pointer_; }
    [[nodiscard]] explicit operator bool() const noexcept { return pointer_ != nullptr; }

    /// For the out-parameter of a Create call. Releases whatever was held.
    [[nodiscard]] T** put() {
        reset();
        return &pointer_;
    }
    [[nodiscard]] void** putVoid() { return reinterpret_cast<void**>(put()); }

private:
    T* pointer_ = nullptr;
};

/// Turns an HRESULT into an Error carrying the system message.
[[nodiscard]] Error hresultError(HRESULT result, std::string_view what);

#define TF_D3D(expr, what)                                        \
    do {                                                          \
        const HRESULT tf_hr = (expr);                             \
        if (FAILED(tf_hr)) {                                      \
            return std::unexpected(hresultError(tf_hr, (what)));  \
        }                                                         \
    } while (false)

/// Which shader a dispatch wants. One entry per compute kernel; the order
/// matches the compiled blob table.
enum class ShaderId : u32 {
    RmsNorm,
    RmsNormNoWeight,
    DequantGemv4,
    DequantGemv8,
    DequantGemm4,
    DequantGemm8,
    EmbedLookup4,
    EmbedLookup8,
    Rope,
    Geglu,
    Add,
    Scale,
    LogitSoftcap,
    Argmax,
    KvWrite,
    Attention,
    RouterTopK,
    GatherRows,
    ScatterAddRows,
    FillZero,
    MoeCombine,
    Count,
};

/// The compiled DXIL for one shader, produced at build time by dxc and
/// embedded so there is no runtime compiler dependency and no shader cache to
/// go stale.
struct ShaderBlob {
    const u8* bytes = nullptr;
    usize size = 0;
};

[[nodiscard]] ShaderBlob shaderBlob(ShaderId id);
[[nodiscard]] std::string_view shaderName(ShaderId id);

}  // namespace tf::gpu::d3d12
