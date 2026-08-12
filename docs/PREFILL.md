# Prefill

Prompt processing runs a chunk of tokens through the stack at once instead of
one at a time. On this machine that is the difference between 46 and 314 tokens
per second with everything resident, and between 12 and 52 when experts stream
from disk.

The reason is arithmetic intensity, not parallelism. Decode is bound by weight
bandwidth: a single token reads all 1.26 GiB of the resident set to produce one
output vector, and the GPU spends almost all of its time waiting on memory. A
chunk reads the same weights once and uses them for every token in it.

## What changes, and what deliberately does not

Every operation in the layer sequence takes a token count, and decode is the
one-token case of the same code. There is no second implementation of the layer
to drift out of step with the first. Concretely:

| Operation | How it batches |
|---|---|
| RMSNorm | already row-shaped; `rows` becomes `tokens x heads` where per-head |
| GeGLU, add, scale | elementwise; the count is simply longer |
| RoPE, KV write | one block per (head, token); token `t` sits at `base + t` |
| Attention | one block per (head, token), each masking to its own position |
| Router top-k | one block per token |
| Projections | `dequantGemv` becomes `dequantGemm` |
| Routed experts | grouped by expert rather than looped per token |

Three of those are worth explaining.

### The batched GEMM

`dequantGemmKernel` keeps the GEMV's warp-per-output-row structure but gives
each thread `kTokenTile` accumulators. A lane loads one packed 4-bit word,
dequantizes it once, and feeds those eight weights to sixteen tokens. The DRAM
traffic is unchanged; the work done per byte is sixteen times higher. Each lane
also carries four output rows, so one activation read feeds four rows of
arithmetic - which is what took this from 3.5x to 10.8x over the 128 GEMV
launches it replaces. See docs/BENCHMARKS.md for the tiling sweep.

The activation tile is staged in shared memory **token-major**, which is the
only non-obvious part. The features belonging to one lane's packed word are then
contiguous and can be read as `float4`. Feature-major would place those features
`kTokenTile` apart; the resulting bank stride is a multiple of eight, so all 32
lanes would collapse onto four banks and every read would serialize eight ways.

At one token the batched kernel is the wrong shape - fifteen sixteenths of every
tile is empty - so `IKernels::dequantGemm` dispatches to the GEMV when
`tokens == 1`. This is not a detail the runtime should have to know, so it lives
in the CUDA dispatch layer. It is worth 38.7 tok/s against 21.6 on decode, which
is what the first measurement after batching everything showed.

### Causality without a mask

Query `t` sits at position `basePosition + t` and computes its own visible range
inside the kernel. No mask tensor is built, materialized or uploaded, and a
decode step is `tokens == 1` of the same kernel rather than a separate one.

The chunk's own keys and values are written before attention runs, so a later
token in the chunk attends to an earlier one exactly as it would have if the two
had been decoded separately.

### Grouped routed experts

This is where streamed prefill wins most. Decode selects the top 8 of 128
experts for its one token. A chunk of 128 tokens makes 1024 selections that
between them touch most of the layer's experts, and the naive approach - loop
each token's eight experts - reads an expert's 3.19 MiB of weights once per
token that chose it.

Instead the routing is inverted on the host: for each expert, the list of tokens
that selected it. Those rows are gathered into a contiguous block, one GEMM runs
over all of them, and the results are scattered back with the router weight
folded in. Each expert's weights are read **once per chunk**.

A token selects a given expert at most once, so within one scatter no two source
rows share a destination and no atomics are needed.

For a streamed layer the experts are fetched a slotful at a time, since a chunk
routinely wants more experts than the slots hold. Each expert still crosses PCIe
at most once for the whole chunk - which is why streamed prefill runs at 52
tok/s while streamed decode runs at 24.

## The sliding-window ring must be sized for the chunk

This one is a trap, and it fails quietly.

The 25 sliding layers keep a ring of `window + headroom` rows. A prefill chunk
writes *all* of its keys and values before *any* of its queries read them. So
writing token `i` of a chunk starting at position `p` overwrites the row holding
position `p + i - window - headroom`, while the chunk's own first query still
needs everything back to `p - window + 1`. That survives only while
`i <= headroom`.

A chunk wider than the headroom therefore destroys history its early tokens are
about to read. The damage is confined to the oldest visible positions, so the
model produces fluent, subtly wrong output rather than anything that looks like
a failure.

`kvRowsForLayer` takes the chunk width and sizes the headroom to at least cover
it, and `prefillChunk` refuses a chunk wider than the rings it was given. Both
halves matter: the first makes large chunks work, the second makes a
misconfiguration an error instead of a corruption.

The cost is small - 512-token chunks at a 4096 context raise the KV cache from
0.30 to 0.37 GiB - which is exactly why it is easy to get wrong and never
notice.

## Chunk size

Larger chunks amortize the weight read over more tokens, at the cost of scratch
and KV rows. Measured on a 1505-token prompt, 4096 context, all resident:

| Chunk | Prefill | Scratch | KV cache |
|---|---|---|---|
| 128 | 239 tok/s | 0.03 GiB | 0.30 GiB |
| 256 | 284 tok/s | 0.06 GiB | 0.32 GiB |
| 512 | 314 tok/s | 0.12 GiB | 0.37 GiB |

The default is 128. It equals the ring's headroom floor, so the default
configuration needs no extra KV rows at all, and the returns above it are
shallow. `--prefill-chunk` trades memory for speed, and 512 is worth setting on
this card for long prompts - see docs/BENCHMARKS.md.

Both figures feed the residency planner *before* it decides how many layers stay
resident, via `ResidencyBudget::maxPrefillTokens`. Discovering the cost after
the weights are already in VRAM is the wrong order for that to happen in.

## How this is known to be correct

`tests/runtime/PrefillTests.cpp` runs a chunk of N tokens and the same N tokens
through `decodeStep` one at a time, from separate caches, and compares the final
residual stream and the logits. The decode path is the one already checked
against the CPU reference stage by stage, so it is a meaningful reference - and
the two paths share no code below the layer sequence.

The comparison reports both the worst-element and the RMS deviation, because
only the pair distinguishes rounding from a real disagreement:

| Tokens | worst/rms | rms/rms |
|---|---|---|
| 1 | 0 | 0 |
| 2 | 0.0053 | 0.00018 |
| 16 | 0.0053 | 0.00032 |
| 17 | 0.0052 | 0.00039 |
| 40 | 0.0212 | 0.00146 |

The jump in the worst element at 40 tokens looks alarming and is not: the RMS
ratio grows smoothly across the whole range, so what moved is a single element
out of 2816, which is what accumulation rounding looks like. A mis-grouped
expert or an off-by-one in the mask moves the whole tensor and drives both
columns together. The argmax matches at every size, and a second test checks
that four tokens generated after a prefill are identical to four generated after
the equivalent decode.

Reporting only the worst element is how a NaN once hid behind a passing check -
`std::max` returns the other operand when one side is NaN. Both metrics treat a
non-finite value as infinite deviation.
