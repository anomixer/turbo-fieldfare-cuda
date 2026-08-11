# Windows CUDA build

`build.bat` is the supported Windows x64 build bootstrap.

## Prerequisites

- Windows 11 x64.
- NVIDIA CUDA-capable GPU.
- Git for Windows, CMake, Ninja, and Visual Studio 2022 Build Tools or Community with Desktop development with C++ and MSVC x64 tools.
- CUDA Toolkit on `PATH` for modern cards, or CUDA 12.x installed for the legacy profile.

## Profiles

```bat
build.bat auto
build.bat legacy
build.bat modern
```

`auto` detects the first NVIDIA GPU compute capability. GPUs below `sm_75` select `legacy`; Turing and newer select `modern`.

## Legacy CUDA 12

Maxwell, Pascal, and Volta require a CUDA 12.x toolchain. Set `CUDA_LEGACY_HOME` when it is not installed at the script default:

```bat
set CUDA_LEGACY_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6
build.bat legacy
```

## Architecture override

The project uses `TF_CUDA_ARCHITECTURES`, not only `CMAKE_CUDA_ARCHITECTURES`.

For an Ada GPU such as NVIDIA L4 or RTX 4060:

```bat
set TF_CUDA_ARCHITECTURES=89-real
build.bat modern
```

The `nvcc` line must include `compute_89` and `sm_89`. It must not be a Blackwell-only `sm_120` build.

## Known limitations

- The initial validated target is Ada `sm_89`.
- Pre-Maxwell GPUs are out of scope.
- The CLI currently has a separate Unicode argument limitation for non-ASCII `--prompt` values. The OpenAI-compatible HTTP API accepts UTF-8 JSON correctly.
- The bootstrap applies the known MSVC C4244 `std::tolower` correction to `src/cli/main.cpp` only when it finds the exact expected source pattern.
