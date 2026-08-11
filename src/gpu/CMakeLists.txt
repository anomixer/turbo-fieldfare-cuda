# tfgpu - the GPU backend seam plus its CUDA implementation.
#
# The interface in include/tf/gpu is CUDA-free by construction so the D3D12
# backend planned for M12 slots in beside this one. Nothing above this library
# includes a CUDA header.

add_library(tfgpu STATIC
        src/Backend.cpp
        src/Preflight.cpp)

add_library(tf::gpu ALIAS tfgpu)

target_include_directories(tfgpu PUBLIC include)

target_link_libraries(tfgpu
        PUBLIC tf::core
        PRIVATE tf::options)

if(TF_HAVE_CUDA)
    find_package(CUDAToolkit REQUIRED)

    target_sources(tfgpu PRIVATE
            src/cuda/CudaBackend.cpp
            src/cuda/CudaKernelDispatch.cpp
            src/cuda/CudaKernels.cu)

    set_target_properties(tfgpu PROPERTIES CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")

    target_link_libraries(tfgpu PRIVATE tf::cuda_options)

    # cudart_static rather than the shared runtime: it removes the cudart DLL
    # from the deployment entirely, so the only external dependency is the
    # NVIDIA driver. A machine without one then fails with a CUDA error we can
    # explain, instead of failing to load the executable at all.
    target_link_libraries(tfgpu PRIVATE CUDA::cudart_static)

    target_compile_definitions(tfgpu PUBLIC TF_HAVE_CUDA=1)
else()
    target_compile_definitions(tfgpu PUBLIC TF_HAVE_CUDA=0)
endif()

# ---------------------------------------------------------------------------
# D3D12
# ---------------------------------------------------------------------------
#
# The second backend. Unlike CUDA it needs no toolkit: d3d12 and dxgi are
# in-box, and dxc ships with the Windows SDK the project already requires.

option(TF_HAVE_D3D12 "Build the Direct3D 12 backend" ON)

if(TF_HAVE_D3D12)
    include(CompileHLSL)

    if(NOT TF_DXC)
        message(WARNING "dxc was not found; building without the D3D12 backend")
        set(TF_HAVE_D3D12 OFF)
    endif()
endif()

if(TF_HAVE_D3D12)
    set(TF_SHADER_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated/shaders")
    set(TF_SHADER_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/src/d3d12/Kernels.hlsl")
    set(TF_SHADER_HEADERS "")

    # Order is irrelevant here; D3D12Shaders.cpp is what pins the mapping.
    foreach(entry
            RmsNorm RmsNormNoWeight
            DequantGemv4 DequantGemv8 DequantGemm4 DequantGemm8
            EmbedLookup4 EmbedLookup8
            Rope Geglu Add Scale LogitSoftcap Argmax
            KvWrite Attention RouterTopK
            GatherRows ScatterAddRows FillZero MoeCombine)
        tf_compile_hlsl("${TF_SHADER_SOURCE}" ${entry} "${TF_SHADER_DIR}" TF_SHADER_HEADERS)
    endforeach()

    add_custom_target(tfgpu_shaders DEPENDS ${TF_SHADER_HEADERS})
    add_dependencies(tfgpu tfgpu_shaders)

    target_sources(tfgpu PRIVATE
            src/d3d12/D3D12Backend.cpp
            src/d3d12/D3D12Kernels.cpp
            src/d3d12/D3D12Shaders.cpp)

    # The generated headers are included as "shaders/Name.h".
    target_include_directories(tfgpu PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated")

    target_link_libraries(tfgpu PRIVATE d3d12 dxgi dxguid)
    target_compile_definitions(tfgpu PUBLIC TF_HAVE_D3D12=1)
else()
    target_compile_definitions(tfgpu PUBLIC TF_HAVE_D3D12=0)
endif()

set_target_properties(tfgpu PROPERTIES FOLDER "libs")
