# NVIDIA RTX 4000 SFF Ada llama.cpp benchmark

This is the llama.cpp companion run on an NVIDIA RTX 4000 SFF Ada (`sm_89`)
using the official Gemma 4 Q4_0 GGUF. It generated 512 tokens from the
unified benchmark prompt; llama.cpp counted the prompt as 83 tokens.

| Target | `--n-gpu-layers` | Prompt | Generation |
| ---: | ---: | ---: | ---: |
| 4GB | 4 | 33.9 tok/s | 16.8 tok/s |
| 6GB | 11 | 47.4 tok/s | 19.1 tok/s |
| 8GB | 18 | 63.7 tok/s | 27.7 tok/s |
| 10GB | 24 | 98.5 tok/s | 43.9 tok/s |
| 12GB | 30 | 222.6 tok/s | 69.5 tok/s |
| Full | 999 | 342.3 tok/s | 75.5 tok/s |

The same input text is 91 tokens in TurboFieldfare's tokenizer. The difference
is expected because the two runtimes use different model/tokenizer formats
(`.gturbo` versus GGUF). All six runs completed without an error.
