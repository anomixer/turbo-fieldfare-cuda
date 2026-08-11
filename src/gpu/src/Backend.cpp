#include "tf/gpu/Backend.h"

namespace tf::gpu {

#if TF_HAVE_CUDA
/// Defined in cuda/CudaBackend.cpp.
[[nodiscard]] Result<BackendPtr> createCudaBackend(u32 deviceIndex);
#endif

#if TF_HAVE_D3D12
namespace d3d12 {
/// Defined in d3d12/D3D12Backend.cpp.
[[nodiscard]] Result<BackendPtr> createD3D12Backend(u32 deviceIndex);
}  // namespace d3d12
#endif

std::string_view toString(BackendKind kind) noexcept {
    switch (kind) {
        case BackendKind::Cuda:  return "CUDA";
        case BackendKind::D3D12: return "D3D12";
    }
    return "?";
}

std::string_view toString(MemoryKind kind) noexcept {
    switch (kind) {
        case MemoryKind::Device:      return "device";
        case MemoryKind::PinnedHost:  return "pinned-host";
        case MemoryKind::HostVisible: return "host-visible";
    }
    return "?";
}

std::vector<BackendKind> compiledBackends() {
    // CUDA first when both are built. Callers that do not care take the front
    // of this list, and on NVIDIA hardware CUDA is the faster of the two.
    std::vector<BackendKind> kinds;
#if TF_HAVE_CUDA
    kinds.push_back(BackendKind::Cuda);
#endif
#if TF_HAVE_D3D12
    kinds.push_back(BackendKind::D3D12);
#endif
    return kinds;
}

Result<BackendPtr> createBackend(BackendKind kind, u32 deviceIndex) {
    switch (kind) {
        case BackendKind::Cuda:
#if TF_HAVE_CUDA
            return createCudaBackend(deviceIndex);
#else
            return makeError(ErrorCode::Unsupported,
                             "this build was configured without CUDA; reconfigure with "
                             "-DTF_ENABLE_CUDA=ON and a CUDA Toolkit installed");
#endif
        case BackendKind::D3D12:
#if TF_HAVE_D3D12
            return d3d12::createD3D12Backend(deviceIndex);
#else
            return makeError(ErrorCode::Unsupported,
                             "this build was configured without D3D12; reconfigure with "
                             "-DTF_HAVE_D3D12=ON and a Windows SDK that provides dxc");
#endif
    }
    return makeError(ErrorCode::InvalidArgument, "unknown backend kind");
}

}  // namespace tf::gpu
