# Compatibility policy

## Validated

| Platform | GPU architecture | Compute capability | Status |
| --- | --- | ---: | --- |
| Windows x64 | NVIDIA Ada | `sm_89` | Validated on NVIDIA L4; intended for RTX 4060-class Ada GPUs and above |

## Planned, not yet validated

| GPU architecture | Compute capability | Release status |
| --- | ---: | --- |
| NVIDIA Turing | `sm_75` | Planned after functional and quality tests |
| NVIDIA Ampere | `sm_80`, `sm_86` | Planned after functional and quality tests |
| NVIDIA Hopper | `sm_90` | Planned after functional and quality tests |
| NVIDIA Blackwell | `sm_120` | Planned after functional and quality tests |

## CUDA architecture configuration

This project uses the project-specific CMake cache variable:

```text
TF_CUDA_ARCHITECTURES
```

For Ada / NVIDIA L4 builds, configure with:

```bat
cmake -S . -B build\relwithdebinfo -G Ninja ^
  -DCMAKE_BUILD_TYPE=RelWithDebInfo ^
  -DTF_CUDA_ARCHITECTURES=89-real
```

Do not rely only on `CMAKE_CUDA_ARCHITECTURES`. The source baseline may define `TF_CUDA_ARCHITECTURES` and use it to overwrite the CMake-wide value.

## VRAM policy

The runtime supports explicit `--vram-budget` values. A successful L4 budget simulation does not guarantee identical performance on a physical low-VRAM card: GPU compute throughput, PCIe topology, CPU, RAM, NVMe latency, cooling, and desktop VRAM reservations all matter.
