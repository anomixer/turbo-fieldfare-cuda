# TurboFieldfare CUDA

[English](README.md) | [&#x7E41;&#x9AD4;&#x4E2D;&#x6587;](README.zh-TW.md)

Windows 優先的 CUDA 發行版，提供專用的 Gemma 4 26B-A4B MoE runtime。

> 狀態：原始碼建置與本機 Windows package staging 均可使用。

## 目標

- 提供 NVIDIA CUDA GPU 的 Windows x64 prebuilt binaries（package staging 已完成；GitHub Release 上傳仍是發布流程的一部分）。
- 正確選擇 CUDA 架構，從 Ada `sm_89` 開始。
- 支援 4GB、6GB、8GB VRAM budget 的低 VRAM streamed-expert 執行。
- 提供 CLI 與 OpenAI-compatible 本機 HTTP server。
- 將本機模型安裝成 `.gturbo` 格式。

## 已驗證基準

在 NVIDIA L4（`sm_89`、24GB）上，CUDA runtime 已用 Gemma 4 26B-A4B 驗證完整駐留與受限 VRAM 模式：

| VRAM budget | Resident / streamed layers | 512-token decode |
| ---: | ---: | ---: |
| 4GB | 0 / 30 | 15.2 tok/s |
| 6GB | 4 / 26 | 16.8 tok/s |
| 8GB | 11 / 19 | 18.3 tok/s |
| 10GB | 18 / 12 | 20.4 tok/s |
| 12GB | 24 / 6 | 23.0 tok/s |
| Full resident | 30 / 0 | 28.5 tok/s |

以上是 L4 上的 VRAM budget 模擬。實體低 VRAM 顯示卡的效能還會受到 GPU 算力、PCIe、系統 RAM、NVMe throughput 與 driver 設定影響。

另外的 RTX 4000 SFF Ada 結果記錄在 [`docs/RTX4000_SFF_ADA_BENCHMARK.md`](docs/RTX4000_SFF_ADA_BENCHMARK.md)。它不是同一硬體，不應直接和 L4 表格比較。

## 效能現實檢查（選用本 runtime 前請先閱讀）

這個 fork 目前不是 CUDA 吞吐量冠軍。在相同 prompt 文字的 benchmark 中，llama.cpp 在每一張測試過的 NVIDIA 離散 GPU 上都比 TurboFieldfare 快：

| GPU | TurboFieldfare | llama.cpp | llama.cpp 優勢 |
| --- | ---: | ---: | ---: |
| NVIDIA L4 | 15.2–28.5 tok/s | 20.5–77.2 tok/s | 約 1.3–3.1 倍 |
| RTX 4000 SFF Ada | 13.4–26.4 tok/s | 16.8–75.5 tok/s | 約 1.3–3.7 倍 |
| T1000 8GB | 6.0–6.7 tok/s | 16.6–17.0 tok/s | 約 2.5–2.8 倍 |

兩個 runtime 收到相同 prompt 文字，但 tokenizer 不同（`.gturbo` 回報 91 tokens，GGUF 回報 83 tokens）。完整原始數據與方法請參考各硬體的 benchmark 文件。

TurboFieldfare 比較可能適合 UMA/shared-memory 系統，例如 Apple Silicon、AMD APU 或 Intel integrated GPU；CPU 與 GPU 共用記憶體時，SSD/expert streaming 的成本可能較低。這些平台目前不受本 Windows CUDA fork 支援：Mac 需要 Metal backend，AMD/Intel 則需要相容的 HIP、Vulkan 或 D3D12 backend。這是設計方向，不是相容性承諾。

如果你的優先目標是 NVIDIA CUDA raw tokens/s，請使用 llama.cpp。選擇本 fork 的理由是 `.gturbo` Gemma 4 支援、可控制的 VRAM budget、expert streaming，以及本機 server/GUI 整合。

## GPU 支援政策

`sm_89` 是目前最初驗證的目標：NVIDIA L4，以及 RTX 4060 和更新 Ada 卡等 Ada GeForce/workstation GPU。

在每個 CUDA 架構都完成建置與測試前，本專案不應宣稱支援所有 CUDA GPU。預計支援目標列在 [docs/COMPATIBILITY.md](docs/COMPATIBILITY.md)。

## 目前已知問題

Windows CLI 目前會破壞透過 `--prompt` 傳入的非 ASCII 文字。CJK 文字請先使用 UTF-8 OpenAI-compatible HTTP API，直到 CLI 從 `char** argv` 遷移到 Windows Unicode arguments。

## 本 fork 與 upstream 的比較

上游專案是 [jaimeburnap/turbo-fieldfare-win](https://github.com/jaimeburnap/turbo-fieldfare-win)。本 repository 追蹤該 Windows CUDA runtime，同時加入 distribution 與 local-agent layer。以下內容是直接比較 upstream `master` branch 的結果，不是對原專案的臆測。

### Fork 新增與修改

- **Build bootstrap：** `build.bat`、`scripts/bootstrap-windows.ps1` 與 diagnostics/quickbench scripts 會偵測 NVIDIA GPU、選擇 legacy 或 modern CUDA profile、驗證 CUDA/CMake/Ninja/MSVC、建置 binaries 並執行 smoke checks。缺少依賴時會先詢問，再透過 `winget` 安裝並自動重新檢查 toolchain。
- **Windows launchers：** `server.bat` 啟動既有的 `tf-server.exe`；`dlmodel.bat` 會重用 `.\models\gemma4.gturbo` 或 `C:\models\gemma4.gturbo`，只有兩處都不存在時才下載並 repack pinned checkpoint。
- **Unified build：** `build.bat` 先建置 CMake binaries，再建置 WinUI 3 desktop GUI（`build\gui\Release\TurboFieldfare.exe`）。GUI build 使用 MSBuild，並自動尋找 Visual Studio Community 或 Build Tools。
- **WinUI GUI runtime：** unpackaged GUI 會明確 bootstrap Windows App Runtime 1.7、顯示 HRESULT 而不是 SDK generic dialog、優先尋找旁邊的 `\.\models\gemma4.gturbo`，並啟動 GUI 旁邊相符的 `tf-decode.exe`。
- **Fork validation data：** 新增 L4 4–12GB benchmark、low-VRAM policy、Windows quick start、compatibility、roadmap 與 Windows build 文件，記錄已驗證的 Ada `sm_89`/L4 目標與實用 VRAM budget 建議。
- **Browser compatibility：** 所有 JSON 與 SSE response 都加入 CORS headers；`OPTIONS` preflight 回傳 `204 No Content`。
- **SSE completion fix：** 傳送 `data: [DONE]` 後 server 會關閉 streaming connection，等待 EOF 的 client 可以開始下一輪，不會卡在 keep-alive reader。
- **Agent-facing documentation：** 本 README 與 `AGENTS.md` 集中說明 model fallback、server commands、OpenAI client settings、CORS 行為與 low-VRAM agent 建議。

### 繼承自 upstream 的功能

`src/server/main.cpp` 的 OpenAI-compatible `tf-server` entrypoint、`TF_GTURBO_DIR` model override、GPU preflight、server options、HTTP keep-alive、concurrent connection handling、serialized inference、prompt cache 與核心 `/v1` API 都是本 fork 保留的 upstream functionality。上方 fork 修改則負責 Windows/browser agent workflow 的 packaging、validation、documentation 與行為修正。

完整 server wire contract、streaming 行為、prompt caching 與 status-code 規則，請見 [`docs/SERVER.md`](docs/SERVER.md)。

## 預期使用方式

```text
下載 release
  -> build（source checkout 才需要；prebuilt 使用者可跳過）
  -> dlmodel
  -> quickbench
  -> server
  -> 將 OpenAI-compatible client 連到 http://127.0.0.1:8080/v1
```

本 repository 不分發模型權重。使用者必須自行取得符合條件的 checkpoint，並遵守適用的 license terms。

## Attribution 與 licensing

本專案會保留所有 imported 或 derived code 所需的 license 與 attribution notices。本專案基於 [jaimeburnap/turbo-fieldfare-win](https://github.com/jaimeburnap/turbo-fieldfare-win)，並受到 [drumih/turbo-fieldfare](https://github.com/drumih/turbo-fieldfare) 啟發。

## 開發

請參考 [docs/ROADMAP.md](docs/ROADMAP.md) 與 [docs/COMPATIBILITY.md](docs/COMPATIBILITY.md)。

## Binary availability 與 prebuilt packaging

本 fork 不包含模型權重。`legacy` 與 `modern` 是 CUDA build profile：

- `build.bat legacy` 使用 CUDA 12.x toolchain，適用 Maxwell、Pascal、Volta 與其他較舊的支援目標。
- `build.bat modern` 使用目前 CUDA toolchain，適用 Turing 及更新目標，包括已驗證的 Ada `sm_89`/L4 path。

Packaging script 會 staging 已測試的 executables、GUI、metadata、README 與 SHA-256 checksum，不包含 PDB/ILK debug files 或 model data：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\package-windows.ps1 -Profile modern -BuildDirectory build\relwithdebinfo\bin
```

它會在 `dist\` 寫入帶 timestamp 的 archive，並記錄必須另外安裝 Windows App Runtime 1.7。archive 與 checksum 上傳 GitHub Releases 並在目標 GPU 類別驗證前，不應視為已發布。除此之外，請用 `build.bat` 從 source build；本機 build directories 是 development outputs。

### Prebuilt archive 內的檔案

- `bin\tf-cli.exe`：command-line inference 與 one-shot benchmark client。
- `bin\tf-decode.exe`：GUI decode service；負責 model 與 CUDA context。
- `bin\tf-server.exe`：OpenAI-compatible HTTP/SSE server。
- `bin\tf-repack.exe`：將已安裝 checkpoint repack 成 `.gturbo` format。
- `bin\tf-preflight.exe`：檢查 GPU、driver、CUDA 與 model prerequisites。
- `gui\TurboFieldfare.exe`：WinUI desktop chat client。
- `server.bat`：啟動 packaged `tf-server.exe` 並解析 model。
- `gui.bat`：從 source build 或 prebuilt archive 啟動 WinUI desktop GUI。
- `dlmodel.bat`：在 model 尚未安裝時，將 model 下載到 archive 的 `models` directory。
- `quickbench.bat`：執行 4/6/8/10/12GB 與 full-resident VRAM benchmarks。
- `bench-llama.bat`：下載相符 CUDA 12/13 的 llama.cpp Windows server；model 不存在時下載 Google 官方 Q4_0 GGUF，並執行 layer-offload comparison benchmark。
- `scripts\fetch-checkpoint.ps1`：`dlmodel.bat` 使用的 downloader。
- `build-info.json`：profile、version、timestamp 與 dependency metadata。
- `<archive>.sha256`：在解壓縮前驗證 ZIP 的 checksum。

Turing-class low-VRAM 結果記錄在 [`docs/T1000_8GB_BENCHMARK.md`](docs/T1000_8GB_BENCHMARK.md)。
對應的 llama.cpp 結果在 [`docs/T1000_8GB_LLAMA_BENCHMARK.md`](docs/T1000_8GB_LLAMA_BENCHMARK.md)。
RTX 4000 SFF Ada llama.cpp 基準在 [`docs/RTX4000_SFF_ADA_LLAMA_BENCHMARK.md`](docs/RTX4000_SFF_ADA_LLAMA_BENCHMARK.md)。
L4 unified prompt llama.cpp benchmark 在 [`docs/L4_LLAMA_BENCHMARK.md`](docs/L4_LLAMA_BENCHMARK.md)。

---

[English](README.md) | [&#x7E41;&#x9AD4;&#x4E2D;&#x6587;](README.zh-TW.md)
