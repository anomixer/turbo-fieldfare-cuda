# TurboFieldfare CUDA

Windows-first CUDA distribution of a specialized Gemma 4 26B-A4B MoE runtime.

> Status: source build and local Windows package staging are available.

## Goals

- Prebuilt Windows x64 binaries for NVIDIA CUDA GPUs (package staging ready;
  GitHub Release upload is the remaining publishing step).
- Correct CUDA architecture selection, beginning with Ada `sm_89`.
- Low-VRAM streamed-expert execution for 4 GB, 6 GB, and 8 GB budgets.
- CLI and OpenAI-compatible local HTTP server.
- Local model installation into the `.gturbo` format.

## Validated baseline

On an NVIDIA L4 (`sm_89`, 24 GB), the CUDA runtime has been validated with Gemma 4 26B-A4B in full-resident and constrained-VRAM modes:

| VRAM budget | Resident / streamed layers | 512-token decode |
| ---: | ---: | ---: |
| 4 GB | 0 / 30 | 15.2 tok/s |
| 6 GB | 4 / 26 | 16.8 tok/s |
| 8 GB | 11 / 19 | 18.3 tok/s |
| 10 GB | 18 / 12 | 20.4 tok/s |
| 12 GB | 24 / 6 | 23.0 tok/s |
| Full resident | 30 / 0 | 28.5 tok/s |

These are L4 budget simulations. Performance on physical low-VRAM cards depends on GPU compute, PCIe, system RAM, NVMe throughput, and driver configuration.

## GPU support policy

`sm_89` is the initial validated target: NVIDIA L4 and Ada GeForce/workstation GPUs such as RTX 4060 and newer Ada cards.

This project must not claim support for every CUDA GPU until each architecture has been built and tested. Planned release targets are listed in [docs/COMPATIBILITY.md](docs/COMPATIBILITY.md).

## Current known issue

The Windows CLI currently corrupts non-ASCII text passed through `--prompt`. Use the UTF-8 OpenAI-compatible HTTP API for CJK text until the CLI is migrated from `char** argv` to Windows Unicode arguments.

## This fork compared with upstream

The upstream project is [jaimeburnap/turbo-fieldfare-win](https://github.com/jaimeburnap/turbo-fieldfare-win).
This repository tracks that Windows CUDA runtime while adding a distribution
and local-agent layer. The list below is based on a direct comparison with the
upstream `master` branch, not on assumptions about the original project.

### Fork-specific additions and changes

- **Build bootstrap:** `build.bat`, `scripts/bootstrap-windows.ps1`, and the
  related diagnostics/quickbench scripts detect the NVIDIA GPU, select the
  legacy or modern CUDA profile, validate CUDA/CMake/Ninja/MSVC, build the
  binaries, and run smoke checks. When a dependency is missing, it asks for
  permission before installing it through `winget`, then automatically
  rechecks the toolchain.
- **Windows launchers:** `server.bat` starts the existing `tf-server.exe`, and
  `dlmodel.bat` reuses either `.\models\gemma4.gturbo` or
  `C:\models\gemma4.gturbo`, downloading and repacking the pinned checkpoint
  only when both locations are absent.
- **Unified build:** `build.bat` now builds the CMake binaries and then the
  WinUI 3 desktop GUI (`build\gui\Release\TurboFieldfare.exe`). The GUI build
  uses MSBuild and automatically discovers Visual Studio Community or Build
  Tools installations.
- **WinUI GUI runtime:** the unpackaged GUI explicitly bootstraps Windows App
  Runtime 1.7, reports HRESULTs instead of showing the SDK's generic dialog,
  discovers `.\models\gemma4.gturbo` beside the repository first, and starts
  the matching `tf-decode.exe` beside the GUI.
- **Fork validation data:** the fork adds the L4 4–12 GB benchmark, low-VRAM
  target policy, Windows quick start, compatibility, roadmap, and Windows
  build documentation. These record the validated Ada `sm_89`/L4 target and
  the practical VRAM budget guidance used by this fork.
- **Browser compatibility:** all JSON and SSE responses now include CORS
  headers, and `OPTIONS` preflight returns `204 No Content`.
- **SSE completion fix:** after sending `data: [DONE]`, the server closes the
  streaming connection. Clients waiting for EOF can therefore start the next
  turn instead of remaining blocked in the keep-alive reader.
- **Agent-facing documentation:** this README and `AGENTS.md` document the
  model fallback, server commands, OpenAI client settings, CORS behavior, and
  low-VRAM agent recommendations in one place.

### Functionality inherited from upstream

The OpenAI-compatible `tf-server` entrypoint in `src/server/main.cpp`, its
`TF_GTURBO_DIR` model override, GPU preflight, server options, HTTP keep-alive,
concurrent connection handling, serialized inference, prompt cache, and the
core `/v1` API are upstream functionality retained by this fork. The fork's
changes above package, validate, document, and fix the behavior for the
Windows/browser agent workflow.

For the exact server wire contract, streaming behavior, prompt caching, and
status-code rules, see [`docs/SERVER.md`](docs/SERVER.md).

## Intended use

```text
Download release
  -> run diagnostics
  -> install or repack a licensed Gemma checkpoint
  -> run tf-server
  -> connect an OpenAI-compatible client to http://127.0.0.1:8080/v1
```

Model weights are not distributed in this repository. Users must obtain an eligible checkpoint and accept its applicable license terms.

## Attribution and licensing

This project will retain all required license and attribution notices for imported or derived code. It is based on work from `jaimeburnap/turbo-fieldfare-win` and is inspired by `drumih/turbo-fieldfare`.

## Development

See [docs/ROADMAP.md](docs/ROADMAP.md) and [docs/COMPATIBILITY.md](docs/COMPATIBILITY.md).

## Binary availability and prebuilt packaging

This fork does not bundle model weights. The `legacy` and `modern` labels
describe CUDA build profiles:

- `build.bat legacy` builds with a CUDA 12.x toolchain for Maxwell, Pascal,
  Volta, and other older supported targets.
- `build.bat modern` builds with the current CUDA toolchain for Turing and
  newer targets, including the validated Ada `sm_89`/L4 path.

The packaging script stages the tested executables, GUI, metadata, README,
and a SHA-256 checksum without including PDB/ILK debug files or model data:

    powershell -ExecutionPolicy Bypass -File scripts\package-windows.ps1 -Profile modern -BuildDirectory build\relwithdebinfo\bin

It writes a timestamped archive under `dist\` and records that Windows App
Runtime 1.7 must be installed separately. A release is not considered
published until the archive and checksum are uploaded to GitHub Releases and
validated on the target GPU class. Until then, build from source with
`build.bat`; the local build directories are development outputs.

### Files in a prebuilt archive

- `bin\tf-cli.exe`: command-line inference and one-shot benchmark client.
- `bin\tf-decode.exe`: GUI decode service; it owns the model and CUDA context.
- `bin\tf-server.exe`: OpenAI-compatible HTTP/SSE server.
- `bin\tf-repack.exe`: repacks an installed checkpoint into `.gturbo` format.
- `bin\tf-preflight.exe`: checks the GPU, driver, CUDA, and model prerequisites.
- `gui\TurboFieldfare.exe`: WinUI desktop chat client.
- `server.bat`: starts the packaged `tf-server.exe` and resolves the model.
- `gui.bat`: launches the WinUI desktop GUI from either a source build or a
  prebuilt archive.
- `dlmodel.bat`: downloads the model into the archive's `models` directory when
  it is not already installed.
- `quickbench.bat`: runs the 4/6/8/10/12 GB and full-resident VRAM benchmarks.
- `bench-llama.bat`: downloads the matching CUDA 12/13 llama.cpp Windows
  server, downloads Google's official Q4_0 GGUF when absent, and runs the
  layer-offload comparison benchmark.
- `scripts\fetch-checkpoint.ps1`: downloader used by `dlmodel.bat`.
- `build-info.json`: profile, version, timestamp, and dependency metadata.
- `<archive>.sha256`: checksum for verifying the ZIP before extraction.
