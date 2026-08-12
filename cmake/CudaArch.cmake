# Selects CUDA target architectures.
#
# The development machine is an RTX 5060 Ti (Blackwell, compute capability 12.0,
# sm_120). We emit a real cubin for sm_120 so there is no JIT cost on the target,
# plus PTX for the newest arch so future cards still run.
#
# Override with -DTF_CUDA_ARCHITECTURES=... to build for a different card.

set(TF_CUDA_ARCHITECTURES "120-real;120-virtual" CACHE STRING
        "CUDA architectures to build (CMake CUDA_ARCHITECTURES syntax)")

set(CMAKE_CUDA_ARCHITECTURES "${TF_CUDA_ARCHITECTURES}")

# sm_120 requires CUDA 12.8 or newer. Fail loudly at configure time rather than
# with an opaque nvcc error deep in the kernel build.
if(CMAKE_CUDA_ARCHITECTURES MATCHES "12[0-9]" AND CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 12.8)
    message(FATAL_ERROR
            "Blackwell (sm_120) needs CUDA >= 12.8, found ${CMAKE_CUDA_COMPILER_VERSION}. "
            "Upgrade the toolkit or set -DTF_CUDA_ARCHITECTURES=89-real.")
endif()

add_library(tf_cuda_options INTERFACE)
add_library(tf::cuda_options ALIAS tf_cuda_options)

target_compile_options(tf_cuda_options INTERFACE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
        $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
        # Line info makes Nsight Compute attribute stalls to source lines, which
        # is how the M13 tuning pass will find the streaming bubbles.
        $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
        # Host-compiler settings for the C++ half of a .cu. /W4 is not used
        # here: the CUDA headers themselves do not survive it cleanly.
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W3>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/bigobj>)
