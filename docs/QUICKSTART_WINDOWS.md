# Windows quick start

## 1. Build

```bat
build
```

The build script detects the NVIDIA GPU, selects the appropriate CUDA toolchain, checks prerequisites, and builds the project.

## 2. Download the model

```bat
dlmodel
```

The Gemma 4 checkpoint is installed under:

```text
.\models\gemma4.gturbo
```

## 3. Benchmark low VRAM

```bat
quickbench
```

The benchmark detects the GPU VRAM with `nvidia-smi`, runs supported low-VRAM budgets, and writes logs to `benchlogs`.

On an NVIDIA L4, the recorded 4–12 GB results were:

| VRAM budget | Generation throughput | Expert data read |
| ---: | ---: | ---: |
| 4 GB | 15.7 tok/s | 90.53 GiB |
| 6 GB | 16.8 tok/s | 75.96 GiB |
| 8 GB | 18.4 tok/s | 57.95 GiB |
| 10 GB | 21.0 tok/s | 37.53 GiB |
| 12 GB | 23.2 tok/s | 22.07 GiB |

Read the complete results in [L4_LOW_VRAM_BENCHMARK.md](L4_LOW_VRAM_BENCHMARK.md).

## 4. Start the local server

```bat
server
```

`server` launches `tf-server.exe` with `.\models\gemma4.gturbo`. Pass additional arguments directly:

```bat
server --help
server --port 8080
```

## More documentation

- [Windows build details](WINDOWS_BUILD.md)
- [Low-VRAM target policy](LOW_VRAM_TARGETS.md)
- [Server usage](SERVER.md)
- [CLI reference](CLI.md)
