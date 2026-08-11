cmake_minimum_required(VERSION 3.28)

project(turbofieldfare-win
        VERSION 0.1.0
        DESCRIPTION "Windows port of TurboFieldfare: streaming Gemma 4 26B-A4B inference"
        LANGUAGES CXX)

if(NOT WIN32)
    message(FATAL_ERROR
            "This project targets Windows only. The portable inference core lives in "
            "src/core and src/runtime, but the I/O and backend layers are Win32/CUDA.")
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "" FORCE)
endif()

# Single output dir keeps tf-cli.exe, tf-repack.exe and their DLLs together so the
# binaries are runnable straight from the build tree.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

option(TF_ENABLE_CUDA  "Build the CUDA GPU backend (requires CUDA Toolkit)" ON)
option(TF_BUILD_TESTS  "Build the Catch2 test suite"                        ON)
option(TF_WARNINGS_AS_ERRORS "Treat compiler warnings as errors"            OFF)

# ---------------------------------------------------------------------------
# CUDA is optional at configure time.
#
# Everything up to and including M3 (format, installer, tokenizer, CPU reference
# implementations) builds without nvcc. Only the M4+ backend needs it, so a
# missing toolkit degrades to a warning rather than blocking the build.
# ---------------------------------------------------------------------------
set(TF_HAVE_CUDA OFF)
if(TF_ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 20)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
        include(CudaArch)
        set(TF_HAVE_CUDA ON)
    else()
        message(WARNING
                "CUDA Toolkit not found - the GPU backend will not be built.\n"
                "  Install with: winget install Nvidia.CUDA\n"
                "  Then reconfigure. Core, installer, tokenizer and CPU reference "
                "tests all build and run without it.")
    endif()
endif()

# ---------------------------------------------------------------------------
# Shared compile options, carried as an interface target so third-party code
# pulled in by FetchContent does not inherit our warning level.
# ---------------------------------------------------------------------------
add_library(tf_options INTERFACE)
add_library(tf::options ALIAS tf_options)

target_compile_definitions(tf_options INTERFACE
        WIN32_LEAN_AND_MEAN
        NOMINMAX
        UNICODE
        _UNICODE
        _CRT_SECURE_NO_WARNINGS)

if(MSVC)
    # Guarded on COMPILE_LANGUAGE:CXX. nvcc receives its command line
    # unprefixed and would read a bare /W4 as an input filename, so MSVC flags
    # must never reach it directly - anything needed by the host compiler
    # inside a .cu goes through -Xcompiler in tf::cuda_options instead.
    target_compile_options(tf_options INTERFACE
            $<$<COMPILE_LANGUAGE:CXX>:/W4>
            $<$<COMPILE_LANGUAGE:CXX>:/permissive->      # conformance mode
            $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
            $<$<COMPILE_LANGUAGE:CXX>:/Zc:__cplusplus>   # real __cplusplus value
            $<$<COMPILE_LANGUAGE:CXX>:/Zc:preprocessor>  # conforming preprocessor
            $<$<COMPILE_LANGUAGE:CXX>:/EHsc>
            $<$<COMPILE_LANGUAGE:CXX>:/bigobj>           # ported TUs are large
            $<$<COMPILE_LANGUAGE:CXX>:/MP>)
    if(TF_WARNINGS_AS_ERRORS)
        target_compile_options(tf_options INTERFACE $<$<COMPILE_LANGUAGE:CXX>:/WX>)
    endif()
endif()

add_subdirectory(src/core)
add_subdirectory(src/gpu)
add_subdirectory(src/reference)
add_subdirectory(src/repack)
add_subdirectory(src/runtime)
add_subdirectory(src/cli)
add_subdirectory(src/server)
add_subdirectory(src/svc)

add_subdirectory(tools/decode-client)
add_subdirectory(tools/generate)
add_subdirectory(tools/preflight)

if(TF_HAVE_CUDA)
    add_subdirectory(tools/cuda-probe)
endif()

if(TF_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

message(STATUS "")
message(STATUS "turbofieldfare-win ${PROJECT_VERSION}")
message(STATUS "  build type : ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++        : ${CMAKE_CXX_STANDARD} (${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION})")
if(TF_HAVE_CUDA)
    message(STATUS "  CUDA       : ${CMAKE_CUDA_COMPILER_VERSION}, arch ${CMAKE_CUDA_ARCHITECTURES}")
else()
    message(STATUS "  CUDA       : disabled (GPU backend not built)")
endif()
message(STATUS "  tests      : ${TF_BUILD_TESTS}")
message(STATUS "")
