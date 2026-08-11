# Measurements

RTX 5060 Ti 16 GB (Blackwell, sm_120) · Ryzen 9 7900X · 64 GB RAM ·
WD_BLACK SN850X · Gemma 4 26B-A4B, 4-bit, greedy decoding.

Every figure here came from `tf-cli --verbose` or `tf_tests "[bench]"` on this
machine. None is extrapolated.

## Headline

| | Prefill | Decode |
|---|---|---|
| All resident | **314 tok/s** | **38 tok/s** |
| `--vram-budget 6` (streaming) | 52 tok/s | **24 tok/s** |

The streamed row is the interesting one. It is an 8 GB card's worth of VRAM
running a model whose weights are 14.3 GiB, which is the whole premise of the
port.

## What tuning changed

| | Before M13 | After |
|---|---|---|
| Prefill, resident, 512-token chunks | 203 tok/s | 314 tok/s |
| Decode, 6 GiB budget | 13.9 tok/s | 24.1 tok/s |
| Expert cache hit rate, 6 GiB | 39% | 75% |

Two changes, both from measurement rather than intuition.

## The GEMM: register tiling

The batched GEMM landed in M8 at 3.5× the GEMV launches it replaces, and then
sat at roughly 2.8 TFLOP/s against a card that does ~24, and 4.3 GB/s against
~448. Neither compute- nor bandwidth-bound, which pointed at shared memory:
per packed word a lane issued `kTokenTile` activation reads and did
`kTokenTile × kValuesPerWord` multiply-adds - four FMAs per `LDS.128`.

Giving each lane several output rows reuses one activation read for that many
rows of arithmetic without changing the shared traffic at all. Measured at 128
tokens over a 2816×2816 projection, against the 128 GEMV launches it replaces:

| Rows per warp | Token tile | Time | Speedup |
|---|---|---|---|
| 1 | 16 | 0.996 ms | 3.4× |
| 2 | 16 | 0.612 ms | 5.4× |
| **4** | **16** | **0.317 ms** | **10.8×** |
| 8 | 16 | 0.306 ms | 10.9× |
| 6 | 16 | 0.393 ms | 8.4× |
| 4 | 8 | 0.403 ms | 8.3× |
| 4 | 32 | 0.445 ms | 7.4× |
| 2 | 32 | 0.771 ms | 4.2× |

Four rows is the setting. Eight buys three percent for twice the accumulator
registers and more waste on a narrow matrix like the 128-row router.

Widening the token tile to 32 *hurts* at every row count - 32 accumulators per
row exhausts the register file and the occupancy loss outweighs the extra reuse.

## Expert slots: the single most valuable streaming knob

Every slot is a cached expert that does not cross PCIe. Upstream uses 16.
Measured at a 6 GiB budget, 4096 context, 50 generated tokens:

| Slots | Decode | Expert bytes read |
|---|---|---|
| 8 | 9.8 tok/s | - |
| 16 | 13.9 tok/s | 10.6 GiB |
| 24 | 16.7 tok/s | 10.6 GiB |
| **32** | **24.1 tok/s** | 8.95 GiB |
| 40 | 26.0 tok/s | 7.85 GiB |
| 46 | (largest that fits) | |

The default is now 32 rather than 16 - the largest round number that fits a
6 GiB budget at a 4096 context, which is the tightest configuration the project
targets. It nearly doubles streamed decode.

A request that does not fit is now **reduced rather than refused**, and the plan
says so under `--verbose`:

```
  layers          0 resident, 30 streamed
  slots           46 per streamed layer (reduced to fit the budget)
```

Slots are a performance knob; failing to start because a tuning default was
optimistic is the wrong response. Below top-k there is nothing sensible left to
reduce to - the cache would evict experts the current token still needs - so
that case is still an error, and says which limit was hit.

### Slots beat resident layers when the budget is tight

At 6 GiB the planner has to split between whole layers held resident and slots
on the rest. More slots wins decisively: 32 slots with 4 resident layers beats
16 slots with more resident ones by 73%. With few slots the hit rate collapses
and every miss is a 3.19 MiB transfer, which costs more than the layer promotion
saved.

## Prefill chunk

1505-token prompt, 4096 context, all resident:

| Chunk | Prefill | Scratch | KV cache |
|---|---|---|---|
| 128 | 239 tok/s | 0.03 GiB | 0.30 GiB |
| 256 | 284 tok/s | 0.06 GiB | 0.32 GiB |
| 512 | 314 tok/s | 0.12 GiB | 0.37 GiB |

Still climbing at 512. The default stays 128 because it equals the
sliding-window ring's headroom floor, so the default configuration needs no
extra KV rows at all - see docs/PREFILL.md for why that bound exists and what
happens when a chunk exceeds it.

`--prefill-chunk 512` is worth it on this card for long prompts.

## Buffered versus unbuffered reads

`--unbuffered` uses `FILE_FLAG_NO_BUFFERING`, DMA-ing straight into
sector-aligned pinned memory with no page-cache copy.

| | Decode at 6 GiB |
|---|---|
| Buffered (default) | 14.5 tok/s |
| Unbuffered | 12.9 tok/s |

Buffered wins, as the plan predicted: with 64 GB of RAM against 12 GiB of
experts, the whole set stays in the page cache after warmup, and a repeat hit
is served at RAM speed. Unbuffered saves one memcpy and gives up the cache
entirely, which is a bad trade here and a good one on a machine whose RAM cannot
hold the expert set. The flag exists for that machine.

This is the point where the port diverges from upstream on purpose: on macOS the
bottleneck is the SSD, so upstream's read-ahead tuning is aimed at it. Here the
bottleneck is PCIe and the page cache, and read-ahead hinting would be
optimizing the wrong resource.

## Decode rate depends on context

| Context | Decode |
|---|---|
| ~30 tokens | 38 tok/s |
| ~120 tokens | 36 tok/s |
| ~1500 tokens | 14 tok/s |

Attention reads the whole visible history per token. The 25 sliding layers cap
at their 1024-token window, but the 5 full-attention layers grow without bound,
and at 1500 tokens they dominate. A quoted "tokens per second" is meaningless
without the context length it was measured at.

## Backends

The D3D12 backend produces byte-identical output to CUDA and is slower on
NVIDIA. Same card, all resident, greedy:

| | Prefill | Decode |
|---|---|---|
| CUDA | 164 tok/s | 40.6 tok/s |
| D3D12 | 12.6 tok/s | 23.7 tok/s |

Decode is within 1.7x. Prefill is 13x off because the HLSL GEMM reads scalars
from a ByteAddressBuffer where the CUDA one reads float4, and prefill is
dominated by that kernel. docs/D3D12.md has the detail. Neither figure is a
reason to use D3D12 on an NVIDIA card; both matter on hardware where it is the
only option.

## Reproducing

```
tf_tests "[bench]"                      # the GEMM microbenchmark
tf-cli --model <dir> --prompt "..." --verbose
tf-cli ... --vram-budget 6              # force the streaming path
tf-cli ... --expert-slots 40            # override the planner
tf-cli ... --unbuffered --read-threads 8
tf-cli ... --backend d3d12              # the portable backend
```

`--verbose` prints the residency plan, the timings and the expert cache hit
rate. The `[bench]` tests are hidden by default: a suite that fails because a
machine is busy teaches people to ignore failures.

## Not pursued

- **Grouped MoE for decode.** The grouping that makes prefill fast needs several
  tokens to group; at one token it is the same work with more bookkeeping.
- **Fused kernels.** The layer issues ~20 launches per token and at 38 tok/s
  that is roughly 2% of the budget. Not where the time is.
- **Double-buffered expert staging.** The streamer already overlaps reads with
  the shared-expert branch; the remaining serialization is the router readback,
  which is a hard synchronization by nature.
- **CUTLASS tensor cores.** The weights are 4-bit and dequantized inline; a
  tensor-core path would need them materialized in fp16 first, which is the
  bandwidth this design exists to avoid spending.
