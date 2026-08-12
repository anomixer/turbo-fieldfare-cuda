# NVIDIA L4 llama.cpp benchmark

This is the llama.cpp companion benchmark on an NVIDIA L4 (`sm_89`, 24 GB)
using the official Gemma 4 Q4_0 GGUF and the same benchmark prompt text as
TurboFieldfare. The llama.cpp tokenizer counted that prompt as 83 tokens; the
TurboFieldfare tokenizer reports 91 for the same input text.

| Target | `--n-gpu-layers` | Prompt | Generation |
| ---: | ---: | ---: | ---: |
| 4GB | 4 | 50.5 tok/s | 20.5 tok/s |
| 6GB | 11 | 66.3 tok/s | 25.1 tok/s |
| 8GB | 18 | 95.7 tok/s | 32.4 tok/s |
| 10GB | 24 | 150.6 tok/s | 44.5 tok/s |
| 12GB | 30 | 385.1 tok/s | 71.2 tok/s |
| Full | 999 | 538.1 tok/s | 77.2 tok/s |

All six runs completed successfully. Prompt-token counts differ because the
two runtimes use different model/tokenizer formats (`.gturbo` versus GGUF),
despite receiving the same input text. Treat this as a same-prompt-text
comparison, not an identical-tokenization comparison.
