# NVIDIA T1000 8GB llama.cpp benchmark

This is the companion llama.cpp run on the same NVIDIA T1000 8GB system. It
used the official Google Gemma 4 Q4_0 GGUF and the unified benchmark prompt;
llama.cpp counted that prompt as 83 tokens.

| Target | `--n-gpu-layers` | Prompt | Generation |
| ---: | ---: | ---: | ---: |
| 4GB | 4 | 84.6 tok/s | 17.0 tok/s |
| 6GB | 11 | 82.4 tok/s | 16.8 tok/s |
| 8GB | 18 | 82.4 tok/s | 16.7 tok/s |
| 10GB | 24 | 83.2 tok/s | 16.7 tok/s |
| 12GB | 30 | 81.5 tok/s | 16.6 tok/s |
| Full | 999 | 82.9 tok/s | 16.7 tok/s |

All six runs completed without an out-of-memory error. Generation throughput
was effectively flat at 16.6–17.0 tok/s, so this run does not demonstrate a
benefit from increasing the requested GPU-layer count.

## Comparison caveats

The same input text is 91 tokens in TurboFieldfare's tokenizer but 83 tokens
in llama.cpp's GGUF tokenizer. This is expected because the runtimes use
different model/tokenizer formats (`.gturbo` versus GGUF). The T1000
TurboFieldfare result is documented in
[T1000_8GB_BENCHMARK.md](T1000_8GB_BENCHMARK.md).

T1000 uses the project's legacy `sm_75` build. The benchmark must use the
legacy CUDA package, not the modern Ada package.
