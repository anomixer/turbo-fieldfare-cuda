# Gemma 4 26B-A4B semantics

Authoritative reference for the port's numerics, derived from
[`mlx_lm/models/gemma4_text.py`](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/gemma4_text.py)
and `switch_layers.py` (Apache-2.0), cross-checked against the pinned
checkpoint's actual weights. The CPU reference implementations in
`src/reference` encode exactly what is written here, and every CUDA kernel is
tested against those.

**Gemma 4 is not Gemma 3.** Several conventions changed, and the ones below are
the kind that produce plausible-looking but wrong output rather than a crash.

## Activations do not fit in fp16

The checkpoint's `dtype` is **bfloat16**, and that matters for range, not just
precision. bf16 carries fp32's 8-bit exponent; fp16 has 5 bits and saturates at
65504.

Gemma 4's norm weights are large - `norm.weight` peaks at 588 - and RMSNorm
output is unit-RMS multiplied by the weight, so a norm with large weights emits
correspondingly large activations. Measured on layer 5 against real weights:

| | expert input RMS | peak | gate projection peak |
|---|---|---|---|
| layer 0 (sliding) | 0.42 | 2.9 | 3.6 |
| layer 5 (full attention) | **67.7** | **410** | **359** |

GeGLU then multiplies the gate and up projections together. At layer 5 that
product reaches roughly 130,000, which overflows fp16 to infinity and becomes
NaN through the down projection. Five of the eight routed experts on that layer
produce NaN.

So activations must be held in bf16 or fp32. fp16 is not merely lossy here, it
is out of range, and the failure is silent: the residual stream still looks
healthy because the damage is confined to the expert branch of some layers, and
the model keeps emitting fluent text that is simply wrong.

`tests/runtime/LayerReferenceTests.cpp` pins this by driving one real layer on
both the CPU reference and the GPU and diffing every intermediate.

## The chat template is not optional

This is an instruction-tuned checkpoint, and raw completion text is out of
distribution for it. Fed `The capital of France is`, it produces fluent
nonsense. Fed the same question through its template it answers correctly and
stops cleanly:

```
<bos><|turn>user\nWhat is the capital of France?<turn|>\n<|turn>model\n
->  <|channel>thought\n<channel|>The capital of France is **Paris**.<turn|>
```

Token ids read from the installed tokenizer: `<|turn>` 105, `<turn|>` 106,
newline 107, `user` 2364, `model` 4368. The model opens a `<|channel>thought`
block of its own accord and terminates on `<turn|>`, which is one of the three
end-of-sequence ids.

Worth stating plainly because it cost real debugging time: a wrong-looking
generation is not evidence of a wrong implementation until the prompt is in the
format the model was trained on.

## Traps

### RMSNorm uses plain `weight`, not `1 + weight`

Gemma 1/2/3 apply `x * rsqrt(mean(x²) + eps) * (1 + w)`. Gemma 4 uses MLX's
stock `nn.RMSNorm`, which is:

```
rmsnorm(x, w) = x * rsqrt(mean(x²) + eps) * w
```

Confirmed independently from the checkpoint: `layers.0.input_layernorm.weight`
has mean 4.48 and min 1.70, and `layers.0.self_attn.q_norm.weight` is a constant
1.0234. Under the `1 + w` convention those would centre near zero, and q_norm
would be amplifying Q by 2.02x rather than acting near-identity.

`eps = 1e-6`. The mean is over the last dimension.

### Attention scale is 1.0, not `1/sqrt(head_dim)`

```python
self.scale = 1.0
```

Q and K are already unit-normalised per head by `q_norm` / `k_norm`, so the
usual `1/sqrt(d)` factor is absent. Applying it would silently flatten every
attention distribution.

### V branches off the *raw* K projection

On full-attention layers `attention_k_eq_v` is set, so there is no `v_proj`
tensor at all (which is why v_proj appears 25 times, not 30). V is **not** equal
to the final K:

```
k_raw = k_proj(x)
K = rope(k_norm(k_raw))        # scaled per-head norm, then RoPE
V = v_norm(k_raw)              # no-scale norm, no RoPE
```

`v_norm` is `RMSNormNoScale` - an RMS normalisation with **no learnable weight
at all** (`rms_norm(x, None, eps)`), not a norm whose weight happens to be one.
There is no `v_norm.weight` tensor in the checkpoint.

This split applies on sliding layers too, where V comes from `v_proj(x)` instead
of `k_proj(x)` but still takes the no-scale, no-RoPE path.

### RoPE differs per layer type

| | sliding | full |
|---|---|---|
| theta | 10000.0 | 1000000.0 |
| partial rotary factor | 1.0 (full rotation) | 0.25 |
| head dim | 256 | 512 |
| KV heads | 8 | 2 |

Only the first `partial_rotary_factor * head_dim` dimensions are rotated on
full-attention layers; the rest pass through unchanged.

## Layer structure

Full attention is on layers **5, 11, 17, 23, 29**; the other 25 are sliding
window (1024 tokens).

```
DecoderLayer(x):
    residual = x
    h = input_layernorm(x)
    h = self_attn(h)                      # includes o_proj
    h = post_attention_layernorm(h)       # post-norm, applied to the branch
    h = residual + h

    residual = h

    # Dense (shared) expert branch
    h1 = pre_feedforward_layernorm(h)
    h1 = mlp(h1)
    h1 = post_feedforward_layernorm_1(h1)

    # Routed expert branch. Note the router reads h, NOT h2.
    idx, wts = router(h)
    h2 = pre_feedforward_layernorm_2(h)
    h2 = experts(h2, idx, wts)
    h2 = post_feedforward_layernorm_2(h2)

    h = h1 + h2
    h = post_feedforward_layernorm(h)
    h = residual + h

    h = h * layer_scalar
```

Both branches run from the same `h`, and the router's input is the
pre-`pre_feedforward_layernorm_2` value. Seven distinct `[hidden]` norms per
layer, plus `q_norm` and `k_norm` at `[head_dim]`.

`layer_scalar` is a single scalar multiplying the layer output *including* the
residual. Layer 0's value is 0.0703, so it is a genuine attenuation, not a
near-one trim.

## Attention

```
q = q_norm(q_proj(x).reshape(B, L, n_heads, head_dim))
q = rope(q, offset)

k_raw = k_proj(x).reshape(B, L, n_kv_heads, head_dim)
k = rope(k_norm(k_raw), offset)
v = v_norm(use_k_eq_v ? k_raw : v_proj(x).reshape(...))

out = softmax(q @ kᵀ * 1.0 + mask) @ v
out = o_proj(out.reshape(B, L, n_heads * head_dim))
```

Grouped-query: 16 Q heads over 8 KV heads (sliding) or 2 (full).

## Router

```
xn = rmsnorm(x, router.scale * hidden_size^-0.5, eps)
scores = router.proj(xn)                       # 8-bit affine, [num_experts]
idx = top_k_indices(scores, k=8)               # by score
w = softmax(scores[idx])                       # softmax over the top-k ONLY
w = w * router.per_expert_scale[idx]
```

Two details worth stating: the RMSNorm weight is `router.scale` pre-multiplied
by `hidden_size^-0.5` (foldable into a per-layer buffer at load), and the
softmax is over the eight selected scores, not over all 128 before selection.

MLX selects via `argpartition`, so ties may order differently from a sorting
top-k. Selection is by raw score, and softmax is applied afterwards.

## Feed-forward

Both the dense MLP and each routed expert are the same GeGLU shape:

```
ffn(x) = down_proj( gelu_approx(gate_proj(x)) * up_proj(x) )
```

`gelu_approx` is the tanh approximation (`gelu_pytorch_tanh`):

```
gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x³)))
```

The gate/up order is easy to invert. `SwitchGLU` computes `x_up` and `x_gate`
then calls `activation(x_up, x_gate)`, and `GeGLU.__call__(self, x, gate)`
returns `gelu_approx(gate) * x` - so the **gate** projection is the one passed
through GELU, matching the dense MLP.

Widths: dense `intermediate_size = 2112`, each routed expert
`moe_intermediate_size = 704`, top-8 of 128.

MoE output is the weighted sum over the selected experts:
`sum_k(w_k * expert_{idx_k}(x))`.

## Embedding and output

```
h = embed_tokens(ids) * sqrt(hidden_size)      # embed_scale = 2816^0.5
...
logits = norm(h) @ embed_tokensᵀ               # tied weights
logits = tanh(logits / 30.0) * 30.0            # final_logit_softcapping
```

`tie_word_embeddings` is true and there is no separate `lm_head` tensor.

## Quantization

MLX affine, group size 64, 4-bit everywhere except `router.proj`, which is
8-bit at the same group size.

```
weight [out, in / (32/bits)]  U32    packed low-order-first
scales [out, in / 64]         BF16
biases [out, in / 64]         BF16

value[o][i] = packed[o][i] * scales[o][i / 64] + biases[o][i / 64]
```

This is affine (scale and additive bias), **not** symmetric with a zero point to
subtract. The bias is added, not multiplied in after an offset.

## Deliberately unsupported

`hidden_size_per_layer_input` is 0 and `num_kv_shared_layers` is 0 for this
checkpoint, so the per-layer input gating path and the cross-layer KV sharing
path are both absent. The reference implementations omit them rather than
carrying dead code; `ArchInfo::validate()` rejects a checkpoint that would need
them.
