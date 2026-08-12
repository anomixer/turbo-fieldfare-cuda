# Hardware baseline

Measured on the development machine at M0 by `tools/cuda-probe`. These are the
numbers the M6 residency planner and the M13 tuning pass are calibrated against.
Re-run `build/<preset>/bin/tf-cuda-probe.exe` after any driver or hardware change.

## Machine

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 5060 Ti, sm_120 (Blackwell), 36 SMs |
| VRAM | 15.93 GiB total, **14.82 GiB free** with a normal desktop session |
| VRAM bandwidth | 128-bit GDDR7 @ 14.00 GHz - 448 GB/s peak |
| Shared memory | 48 KiB/block default (opt-in raises it; Apple's threadgroup limit is 32 KiB) |
| Async copy engines | **1** |
| CPU | Ryzen 9 7900X, 12 cores |
| System RAM | 63.1 GiB |
| Model volume | `C:` - WD_BLACK SN850X NVMe, ~7 GB/s |
| Toolchain | CUDA 13.3.73, MSVC 19.44, Windows SDK 10.0.26100 |

## Measured transfer bandwidth

| Path | Throughput |
|---|---|
| Pinned host → device | **26.49 GiB/s** |
| Pageable host → device | 17.18 GiB/s (1.5× slower) |

Pinned staging is worth the allocation cost, which settles the M6 design: reads
land in a pinned ring, never in pageable memory.

## What this implies

### The PCIe ceiling is not the constraint we expected

A fully-streamed token moves 30 layers × 8 experts × 3.36 MB ≈ 806 MB. At
26.49 GiB/s that is **33.6 tok/s** even if every single expert misses cache -
already at the top of upstream's M5 range (31-35 tok/s). With 63 GiB of RAM
holding all 12 GiB of experts in the page cache after warmup, the SSD drops out
of the steady-state path entirely.

### All experts fit in VRAM on this machine

```
resident core      1.35 GiB
KV cache @ 4K      0.30 GiB
all routed experts 12.00 GiB
                  ----------
                   13.65 GiB   vs 14.82 GiB free
```

So the M6 residency planner will, left to its own devices, mark **every layer
resident and never stream at all** on this box. That is the right answer for a
16 GB card, but it means the streaming path - the entire point of the port -
gets zero coverage here by default.

**Consequence for M6:** `--vram-budget` must be able to constrain the planner
*below* what the hardware offers, so the streamed path can be exercised and
tested on this machine. Treat an artificially low budget (e.g. `--vram-budget
6GiB`, reproducing an 8 GB card) as a first-class test configuration, not a
debug hack. Kernel correctness and streaming correctness must both be provable
here without a second GPU.

### One copy engine

`asyncEngineCount = 1` means H2D transfers overlap with compute, but H2D and D2H
cannot overlap with each other. The decode choreography only needs H2D against
compute, so the three-stream design holds - but M13 should not expect to hide a
D2H router readback behind an expert fetch.

### Confirmed kernel primitives

Both primitives every ported kernel depends on work correctly at sm_120:

- Warp shuffle reduction (`__shfl_down_sync`) - the CUDA form of Metal's `simd_sum`
- Tensor-core MMA (`wmma` 16×16×16 f16→f32) - stands in for Metal 4's `matmul2d`
  on the M8 prefill path

The M8 tensor-core risk is retired at the hardware level. What remains is whether
CUTLASS ships tuned sm_120 tile configurations, which only matters for speed, not
feasibility - the cuBLASLt fallback stays available.

## Verified model architecture

Read from `config.json` of the pinned checkpoint
(`mlx-community/gemma-4-26b-a4b-it-4bit` @ `0d77464e`). These supersede any
values inferred from upstream's docs.

| Field | Value |
|---|---|
| `hidden_size` | 2816 |
| `num_hidden_layers` | 30 |
| `num_attention_heads` | 16 |
| `num_key_value_heads` | 8 (sliding) / 2 (`num_global_key_value_heads`) |
| `head_dim` | 256 (sliding) / 512 (`global_head_dim`) |
| `intermediate_size` | 2112 (shared expert) |
| `moe_intermediate_size` | 704 (per routed expert) |
| `num_experts` / `top_k_experts` | 128 / 8 |
| `vocab_size` | 262144 |
| `sliding_window` | 1024 |
| `rms_norm_eps` | 1e-06 |
| `final_logit_softcapping` | 30.0 |
| `tie_word_embeddings` | true |
| `attention_k_eq_v` | true |
| `hidden_activation` | `gelu_pytorch_tanh` |
| `max_position_embeddings` | 262144 |
| `bos_token_id` / `eos_token_id` | 2 / `[1, 106, 50]` |

**Layer types** - 25 sliding, 5 full attention. Full attention lands on layers
**5, 11, 17, 23, 29** (every 6th, zero-indexed), which is the mask the KV cache
manager must reproduce.

**RoPE** - sliding layers use `theta = 10000.0`, `rope_type = "default"`; full
attention layers use `theta = 1000000.0`, `rope_type = "proportional"` with
`partial_rotary_factor = 0.25`.

**Quantization** - MLX affine, `group_size = 64`, 4-bit throughout, except every
`layers.N.router.proj` which is 8-bit at the same group size.

Tensors carry a `language_model.` prefix: the checkpoint is a
`Gemma4ForConditionalGeneration` multimodal wrapper. The port is text-only, so
the vision tower is skipped at repack time.
