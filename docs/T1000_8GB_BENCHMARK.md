# NVIDIA T1000 8GB benchmark

This is a TurboFieldfare CUDA low-VRAM benchmark on an NVIDIA T1000 8GB
(Turing, `sm_75`) using the validated legacy package. The test generated 512
tokens from a 91-token prompt for each configuration.

| Target | Resident / streamed layers | Generation | Expert data read |
| ---: | ---: | ---: | ---: |
| 4 GB | 0 / 30 | 6.4 tok/s | 92.87 GiB |
| 6 GB | 4 / 26 | **6.7 tok/s** | 74.81 GiB |
| 8 GB | 11 / 19 | 6.6 tok/s | 56.81 GiB |
| 10 GB | 18 / 12 | 6.0 tok/s | 39.18 GiB |
| 12 GB | 24 / 6 | 6.3 tok/s | 21.81 GiB |
| full* | 6 / 24 | **6.7 tok/s** | 70.55 GiB |

`full*` is not a full-resident run on this 8GB card. The runtime automatically
fit the model to available memory, resulting in 6 resident and 24 streamed
layers. The 10GB and 12GB rows are logical runtime budgets; they exceed the
physical VRAM and therefore do not imply that the model is resident in VRAM.

## Recommendation

Use a 6GB or 8GB budget on the T1000. Increasing the logical budget above the
physical 8GB did not improve throughput; 10GB was the slowest result because
of additional memory pressure and streaming.

The raw logs are under `benchlogs\quickbench_*.log` when the benchmark is run.

The companion llama.cpp run is documented in
[T1000_8GB_LLAMA_BENCHMARK.md](T1000_8GB_LLAMA_BENCHMARK.md).
