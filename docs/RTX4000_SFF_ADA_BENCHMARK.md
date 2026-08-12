# NVIDIA RTX 4000 SFF Ada benchmark

This benchmark measures TurboFieldfare CUDA on an NVIDIA RTX 4000 SFF Ada
while constraining the runtime VRAM budget. It is a separate baseline from the
NVIDIA L4 results; the two GPUs share Ada architecture but have different
power, memory capacity, and performance characteristics.

## Test setup

- GPU: NVIDIA RTX 4000 SFF Ada, 20 GB
- Driver: 610.88
- CUDA UMD: 13.3
- Windows WDDM
- Prompt: 91 tokens
- Generation: 512 tokens, stopped on token limit
- Sampling seeds: independently generated per run

## Results

| VRAM budget | Planner total | Resident / streamed layers | Expert slots | Prompt throughput | Generation throughput | Expert data read |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 GB | 3.96 GiB | 0 / 30 | 25 | 43.1 tok/s | 13.4 tok/s | 91.69 GiB |
| 6 GB | 5.81 GiB | 4 / 26 | 32 | 49.8 tok/s | 14.4 tok/s | 76.21 GiB |
| 8 GB | 7.90 GiB | 11 / 19 | 32 | 66.6 tok/s | 15.9 tok/s | 55.65 GiB |
| 10 GB | 10.00 GiB | 18 / 12 | 32 | 87.2 tok/s | 18.8 tok/s | 37.49 GiB |
| 12 GB | 11.79 GiB | 24 / 6 | 32 | 124.7 tok/s | 18.6 tok/s | 21.45 GiB |
| Full resident | 13.59 GiB | 30 / 0 | -- | 235.1 tok/s | 26.4 tok/s | -- |

The full-resident run loaded 13.59 GiB of device memory, below the card's
20 GB physical capacity. Expert-cache hit rate was not included in the full
run summary and is therefore intentionally not reported here.

## Interpretation

- Generation throughput rises from 13.4 tok/s at 4 GB to 18.8 tok/s at 10 GB.
- The 12 GB result is effectively flat relative to 10 GB in this run; this is
  normal benchmark variance and should not be interpreted as a regression in
  the residency planner.
- Full-resident execution reaches 26.4 tok/s, but this is not directly
  comparable to the L4 result because the RTX 4000 SFF Ada has a different
  power and compute envelope.

## Reproduction

Stop all other CUDA processes, then run:

```bat
quickbench.bat
```

The logs are written to `benchlogs\quickbench_*.log`.
