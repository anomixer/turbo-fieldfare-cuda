[CmdletBinding()]
param(
    [ValidateSet('auto','legacy','modern')]
    [string]$Profile = 'auto',
    [string]$ModelPath = '',
    [string]$OutputDirectory = ''
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
if (-not $OutputDirectory) { $OutputDirectory = Join-Path $root 'benchlogs\llama' }
New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null

function Fail([string]$Message) { Write-Host "ERROR: $Message" -ForegroundColor Red; exit 1 }
function Download([string]$Uri, [string]$Destination, [hashtable]$Headers = @{}) {
    Write-Host "Downloading $Uri" -ForegroundColor Cyan
    Invoke-WebRequest -Uri $Uri -OutFile $Destination -Headers $Headers -UseBasicParsing
}

$smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if (-not $smi) { Fail 'nvidia-smi was not found.' }
$gpu = (& nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader | Select-Object -First 1).Trim()
if ($gpu -notmatch ',\s*(?<cap>\d+\.\d+)') { Fail "Could not read GPU compute capability: $gpu" }
$sm = [int](($Matches.cap -replace '\.',''))
if ($Profile -eq 'auto') { $Profile = if ($sm -lt 75) { 'legacy' } else { 'modern' } }
$cudaMajor = if ($Profile -eq 'legacy') { '12' } else { '13' }
Write-Host "GPU: $gpu | profile: $Profile | llama CUDA: $cudaMajor" -ForegroundColor Green

$tools = Join-Path $root 'third_party\llama'
New-Item -ItemType Directory -Path $tools -Force | Out-Null
$server = Join-Path $tools 'llama-server.exe'
if (-not (Test-Path $server)) {
    $release = Invoke-RestMethod 'https://api.github.com/repos/ggml-org/llama.cpp/releases/latest'
    $asset = $release.assets | Where-Object {
        $_.name -match "^llama-.*-bin-win-cuda-${cudaMajor}(?:\.\d+)?-x64\.zip$"
    } | Select-Object -First 1
    # Modern L4 builds may use a newer CUDA 13.x minor release than the
    # installed toolkit; the llama.cpp binary is still driver-compatible.
    if (-not $asset -and $cudaMajor -eq '13') {
        $asset = $release.assets | Where-Object {
            $_.name -match '^llama-.*-bin-win-cuda-13\.\d+-x64\.zip$'
        } | Select-Object -First 1
    }
    if (-not $asset) { Fail "No Windows CUDA $cudaMajor llama.cpp asset found in $($release.tag_name)." }
    $zip = Join-Path $env:TEMP $asset.name
    Download $asset.browser_download_url $zip
    Expand-Archive -LiteralPath $zip -DestinationPath $tools -Force
    Remove-Item $zip -Force
    $found = Get-ChildItem $tools -Recurse -Filter 'llama-server.exe' | Select-Object -First 1
    if (-not $found) { Fail 'Downloaded archive did not contain llama-server.exe.' }
    Copy-Item $found.FullName $server -Force
}

if (-not $ModelPath) { $ModelPath = Join-Path $root 'models\gemma-4-26B_q4_0-it.gguf' }
if (-not (Test-Path $ModelPath)) {
    $modelUri = 'https://huggingface.co/google/gemma-4-26B-A4B-it-qat-q4_0-gguf/resolve/main/gemma-4-26B_q4_0-it.gguf?download=true'
    Write-Host 'The official Google GGUF is about 15 GB.' -ForegroundColor Yellow
    $token = Read-Host 'Optional Hugging Face token (press Enter if public)' -AsSecureString
    $headers = @{}
    if ($token.Length -gt 0) {
        $ptr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($token)
        try { $plain = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($ptr); $headers.Authorization = "Bearer $plain" } finally { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($ptr) }
    }
    New-Item -ItemType Directory -Path (Split-Path $ModelPath) -Force | Out-Null
    Download $modelUri $ModelPath $headers
}

$prompt = 'Write a practical design review for running a 26B Mixture-of-Experts coding assistant. Compare VRAM budgets and give a final recommendation. Use about 350 words.'
$port = 11436
$runs = @(
    @{ Label='4GB'; Layers=4 }, @{ Label='6GB'; Layers=11 }, @{ Label='8GB'; Layers=18 },
    @{ Label='10GB'; Layers=24 }, @{ Label='12GB'; Layers=30 }, @{ Label='full'; Layers=999 }
)

foreach ($run in $runs) {
    $log = Join-Path $OutputDirectory "llama_$($run.Label).log"
    Write-Host "Running llama.cpp $($run.Label) target (`--n-gpu-layers $($run.Layers)`)" -ForegroundColor Cyan
    $before = (& nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv,noheader).Trim()
    $proc = Start-Process -FilePath $server -ArgumentList @('-m', $ModelPath, '--host','127.0.0.1','--port',$port,'--ctx-size','4096','--parallel','1','--n-gpu-layers', $run.Layers, '--jinja') -RedirectStandardOutput "$log.out" -RedirectStandardError "$log.err" -PassThru
    try {
        $ready = $false
        for ($i=0; $i -lt 120; $i++) { try { Invoke-WebRequest "http://127.0.0.1:$port/health" -UseBasicParsing -TimeoutSec 2 | Out-Null; $ready=$true; break } catch { Start-Sleep 1 } }
        if (-not $ready) { throw 'llama-server did not become ready' }
        $body = @{ model='gemma'; prompt=$prompt; n_predict=512; temperature=0.2; stream=$false } | ConvertTo-Json -Compress
        $result = Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:$port/completion" -ContentType 'application/json' -Body $body
        @("requested_label=$($run.Label)", "n_gpu_layers=$($run.Layers)", "before_vram=$before", "prompt_tokens=$($result.tokens_evaluated)", "generated_tokens=$($result.tokens_predicted)", "prompt_ms=$($result.timings.prompt_ms)", "generation_ms=$($result.timings.predicted_ms)", "prompt_tok_s=$($result.timings.prompt_per_second)", "generation_tok_s=$($result.timings.predicted_per_second)") | Set-Content $log
        Get-Content "$log.out","$log.err" | Add-Content $log
    } catch { "ERROR: $($_.Exception.Message)" | Set-Content $log }
    finally { if (-not $proc.HasExited) { Stop-Process $proc -Force }; Remove-Item "$log.out","$log.err" -Force -ErrorAction SilentlyContinue }
}
Write-Host "Completed. Logs: $OutputDirectory" -ForegroundColor Green
