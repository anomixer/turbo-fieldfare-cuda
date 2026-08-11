# Low-VRAM target policy

## Product requirement

TurboFieldfare CUDA targets NVIDIA Maxwell and newer CUDA GPUs with at least 4 GB VRAM.

This is a compatibility target, not a promise of equal speed on every GPU. The runtime relies on SSD-streamed MoE experts when the full model cannot remain resident in VRAM.

## User guidance

| VRAM | Product status | Expected use |
| ---: | --- | --- |
| 4 GB | Minimum / extreme mode | Full streamed expert path; short context; fast NVMe and sufficient system RAM required |
| 6 GB | Minimum practical | Interactive text use on suitable GPUs |
| 8 GB | Recommended | Primary low-VRAM target |
| 10–12 GB | Preferred | More resident layers and expert cache; reduced streaming pressure |
| 16 GB+ | Optional | Compare with conventional full-resident runtimes such as Ollama, llama.cpp, and vLLM |

## Support vocabulary

- **Build-supported:** an artifact can be compiled for the SM target.
- **Smoke-tested:** the artifact loads the model and produces a correct short response on real hardware.
- **Validated:** benchmark, long generation, quality, context, API, and failure behavior have been checked.

Ada `sm_89` is the initial validated target. Other architectures are not called validated until hardware tests are recorded.

## Toolchain profiles

| Profile | Intended architectures | Toolkit policy |
| --- | --- | --- |
| `legacy` | Maxwell, Pascal, Volta, early Turing | CUDA 12.x required |
| `modern` | Turing through Blackwell | CUDA 12.8+ or CUDA 13.x |

Use `build.bat auto` to select a profile from the detected GPU, `build.bat legacy` for a CUDA 12.x legacy build, or `build.bat modern` for a modern build.

## Hardware reality

Low VRAM only removes the capacity barrier. Actual throughput also depends on GPU compute, PCIe topology, system RAM, SSD latency, thermal limits, and operating-system graphics reservations. Maxwell cards in particular may be compatible but substantially slower because they lack later Tensor Core features.
