# tf-cli

```
tf-cli --model C:\models\gemma4.gturbo --prompt "What is the capital of France?"
```

The binary is thin. Argument parsing is `src/cli/Args.cpp` and everything from
the prompt onward is `tf::runtime::Generator`, which the server in M10 and the
GUI in M11 use unchanged. That split is deliberate: the interesting logic should
not be reachable only by launching a process.

## The output is not the token stream

Gemma 4 does not reply with plain text. An assistant turn looks like

```
<|channel>thought\n  ...thinking...  <channel|>  ...answer...  <turn|>
```

mirroring the way a turn itself is `<|turn>role\n ... <turn|>`. Even with
thinking disabled the model still opens and closes an empty thought channel, so
a front end that prints every decoded token shows

```
<|channel>thought
<channel|>Paris
```

where the user asked for `Paris`. `AssistantDecoder` splits the two by token id
- the markers are single tokens, so there is nothing to parse and no way for
ordinary text that happens to spell `<channel|>` to be mistaken for a marker.
The answer goes to stdout; `--show-thinking` sends the channel to stderr.

Everything diagnostic goes to stderr, so `tf-cli ... > answer.txt` captures the
answer and nothing else.

## Sampling

Defaults are temperature 0.2, top-k 64, top-p 0.95, no repetition penalty.
Order of operations:

1. Repetition penalty over the last `--repeat-window` generated tokens.
2. Temperature 0 takes the greedy path and stops here.
3. Top-k, which bounds everything after it - the softmax, the top-p scan and the
   sort all run over k entries rather than 262144.
4. Softmax, with the maximum subtracted first. Logits arrive softcapped to ±30,
   and at temperature 0.05 that is a division by 20 - straight into expf's
   overflow without the subtraction.
5. Top-p over the sorted survivors, keeping the prefix inclusive of the token
   that crosses the threshold, so a nucleus below the leading token's own
   probability still leaves one candidate rather than none.
6. Draw.

Two details worth stating because getting them wrong is invisible:

**The repetition penalty divides positive logits and multiplies negative ones.**
Dividing unconditionally would *raise* an already unlikely token, making a
repeated token more likely - which is the opposite of a penalty, and produces
text that looks fine until it loops.

**Greedy ties resolve to the lowest index**, matching the GPU argmax kernel.
Which of the two runs depends only on whether the logits had to be downloaded,
so they must not disagree.

### Where sampling runs

On the host. The logits have to cross PCIe for any non-greedy choice anyway, and
sorting 262144 candidates is about a millisecond against a 26 ms token - visible,
but M13 found larger wins elsewhere and left it alone. Greedy decoding downloads nothing: the argmax
kernel decides on the device, saving a 1 MiB transfer per token.

`--seed` fixes the stream. An unseeded run still reports the seed it resolved to,
so a good answer can be reproduced after the fact.

## Stop conditions

`--stop` takes text, not tokens, and the text may straddle token boundaries:
`\n\nUser:` can arrive as `\n\n`, `User` and `:` across three steps, and matching
each decoded piece in isolation misses it.

The converse matters just as much. Text that *might* still become a stop string
cannot be emitted yet, because a terminal or an HTTP stream cannot take it back.
`StopMatcher` holds back the longest suffix of what it has seen that is a proper
prefix of some stop string, and releases it as soon as the next piece proves it
was ordinary text - `done.\n\nUse` prints `done.` and waits, then prints
`\n\nUseful notes follow` when `ful` arrives.

Whatever is still held back when generation ends some other way is flushed: it
was part of the answer.

## Exit codes

| Code | Meaning |
|---|---|
| 0 | Generated normally |
| 1 | Preflight failed, or the model could not be loaded or run |
| 2 | The command line was wrong |

Argument errors print the message and the usage text. `Args` is a pure value
with no side effects so this is testable directly, which is why
`tests/cli/ArgsTests.cpp` can check that `--tokens 12abc` is refused rather than
silently read as 12, and that a typo'd flag is an error rather than ignored.

## Memory

`--vram-budget` plans residency below what the card has, which is how the
streamed path is exercised on hardware that would otherwise hold the whole
model. `--prefill-chunk` trades scratch and KV rows for prompt throughput; see
docs/PREFILL.md, including why a chunk wider than the sliding-window headroom is
refused rather than silently corrupting attention.

Preflight runs before anything is loaded, so a missing driver or toolkit
produces an explanation rather than a failure deep inside the backend.

## Measured

RTX 5060 Ti 16 GB, Gemma 4 26B-A4B, greedy:

| | Prefill | Decode |
|---|---|---|
| All resident | 314 tok/s | 38 tok/s |
| `--vram-budget 6` | 52 tok/s | 24 tok/s |

Decode falls to about 14 tok/s at a 1500-token context: the five full-attention
layers read the whole history per token. docs/BENCHMARKS.md has the breakdown
and the tuning knobs.
