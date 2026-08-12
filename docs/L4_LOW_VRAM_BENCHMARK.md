# NVIDIA L4 low-VRAM benchmark

This benchmark measures TurboFieldfare CUDA on an NVIDIA L4 while constraining the runtime VRAM budget below the card's 24 GB physical memory. It exercises the SSD-streamed MoE expert path at 4–12 GB budgets.

## Test setup

- GPU: NVIDIA L4, Ada Lovelace (`sm_89`), 24 GB GDDR6
- NVIDIA driver: 610.88
- Build: Windows CUDA `RelWithDebInfo`
- Prompt: 91 tokens
- Generation: 512 tokens, stopped on token limit
- Sampling seeds: independently generated per run

## Results

| VRAM budget | Planner total | Resident / streamed layers | Expert slots | Prompt throughput | Generation throughput | Expert-cache hit rate | Expert data read |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 GB | 3.96 GiB | 0 / 30 | 25 | 53.8 tok/s | 15.2 tok/s | 75.9% | 93.85 GiB |
| 6 GB | 5.81 GiB | 4 / 26 | 32 | 61.8 tok/s | 16.8 tok/s | 77.5% | 75.90 GiB |
| 8 GB | 7.90 GiB | 11 / 19 | 32 | 80.6 tok/s | 18.3 tok/s | 76.5% | 58.05 GiB |
| 10 GB | 10.00 GiB | 18 / 12 | 32 | 106.5 tok/s | 20.4 tok/s | 74.7% | 39.53 GiB |
| 12 GB | 11.79 GiB | 24 / 6 | 32 | 146.1 tok/s | 23.0 tok/s | 72.5% | 21.51 GiB |
| Full resident | 13.59 GiB | 30 / 0 | 32 | 285.2 tok/s | 28.5 tok/s | -- | -- |

## Observations

- The 4 GB configuration is functional with all 30 layers streamed, but it reads 93.85 GiB of expert data for this 512-token generation.
- Increasing the budget from 4 GB to 12 GB raises generation throughput from 15.2 tok/s to 23.0 tok/s, a 51% improvement.
- At 12 GB, only 6 of 30 layers remain streamed and expert data read falls by 77% relative to 4 GB.
- The 10–12 GB range is the practical low-VRAM sweet spot: it keeps 18–24 layers resident while retaining a large 32-slot expert cache.
- Expert-cache hit rate stays near 72–78% across all budgets. The main source of the speedup is therefore fewer streamed layers and lower SSD traffic, not higher cache-hit percentage.

## Reproduction

After a successful Windows build and model download, run:

```bat
scripts\quickbench.bat .\models\gemma4.gturbo
```

The benchmark writes one log per budget to `benchlogs`:

```text
quickbench_4GB.log
quickbench_6GB.log
quickbench_8GB.log
quickbench_10GB.log
quickbench_12GB.log
quickbench_full.log
```
