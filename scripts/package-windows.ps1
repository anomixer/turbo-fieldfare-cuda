<#
.SYNOPSIS
  Stage and archive a Windows CUDA binary package.

This packages already-built artifacts. It never bundles model weights or the
Windows App Runtime installer; those remain separately licensed/installable
dependencies.
#>
[CmdletBinding()]
param(
    [ValidateSet('legacy', 'modern')]
    [string]$Profile = 'modern',
    [string]$BuildDirectory = '',
    [string]$OutputDirectory = ''
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
if (-not $BuildDirectory) { $BuildDirectory = Join-Path $root 'build\relwithdebinfo\bin' }
if (-not $OutputDirectory) { $OutputDirectory = Join-Path $root 'dist' }

if (-not (Test-Path $BuildDirectory)) {
    throw "Build directory not found: $BuildDirectory. Build the selected profile first."
}

# Do not let a profile label hide a binary built for another GPU family.
# CMake/Ninja records the real CUDA code-generation target in build.ninja.
$ninjaFile = Join-Path $BuildDirectory 'build.ninja'
if (-not (Test-Path $ninjaFile)) { $ninjaFile = Join-Path (Split-Path $BuildDirectory -Parent) 'build.ninja' }
$architecture = if (Test-Path $ninjaFile) {
    $match = Select-String -Path $ninjaFile -Pattern 'compute_(\d+)' | Select-Object -First 1
    if ($match -and $match.Matches[0].Groups[1].Value) { "sm_$($match.Matches[0].Groups[1].Value)" } else { 'unknown' }
} else { 'unknown' }
if ($Profile -eq 'legacy' -and $architecture -ne 'sm_75') {
    throw "Legacy package requires a sm_75 build, found $architecture. Configure with -DTF_CUDA_ARCHITECTURES=75-real."
}
if ($Profile -eq 'modern' -and $architecture -notin @('sm_89', 'sm_90', 'sm_100', 'sm_120')) {
    throw "Modern package requires a modern CUDA architecture, found $architecture."
}

$required = @('tf-cli.exe', 'tf-decode.exe', 'tf-server.exe', 'tf-repack.exe', 'tf-preflight.exe')
$missing = $required | Where-Object { -not (Test-Path (Join-Path $BuildDirectory $_)) }
if ($missing) { throw "Missing required binaries: $($missing -join ', ')" }

$version = (& (Join-Path $BuildDirectory 'tf-cli.exe') --version 2>$null).Trim()
if (-not $version) { throw 'tf-cli --version returned no version.' }

$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$name = "turbo-fieldfare-cuda-windows-$Profile-$stamp"
$stage = Join-Path $OutputDirectory $name
$zip = Join-Path $OutputDirectory "$name.zip"
if (Test-Path $stage) { Remove-Item $stage -Recurse -Force }
New-Item -ItemType Directory -Path $stage -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $stage 'bin') -Force | Out-Null

foreach ($file in $required) { Copy-Item (Join-Path $BuildDirectory $file) (Join-Path $stage 'bin') }
$optional = @('tf-cuda-probe.exe', 'tf-decode-client.exe', 'tf-generate.exe')
foreach ($file in $optional) {
    if (Test-Path (Join-Path $BuildDirectory $file)) { Copy-Item (Join-Path $BuildDirectory $file) (Join-Path $stage 'bin') }
}

$gui = Join-Path $root 'build\gui\Release'
if (Test-Path (Join-Path $gui 'TurboFieldfare.exe')) {
    New-Item -ItemType Directory -Path (Join-Path $stage 'gui') -Force | Out-Null
    foreach ($file in @('TurboFieldfare.exe', 'tf-decode.exe', 'App.xbf', 'resources.pri',
                        'Microsoft.WindowsAppRuntime.Bootstrap.dll',
                        'Microsoft.Web.WebView2.Core.dll', 'Microsoft.Web.WebView2.Core.winmd')) {
        if (Test-Path (Join-Path $gui $file)) { Copy-Item (Join-Path $gui $file) (Join-Path $stage 'gui') }
    }
}

Copy-Item (Join-Path $root 'README.md') $stage
Copy-Item (Join-Path $root 'README.zh-TW.md') $stage
Copy-Item (Join-Path $root 'LICENSE') $stage -ErrorAction SilentlyContinue
foreach ($launcher in @('server.bat', 'dlmodel.bat')) {
    Copy-Item (Join-Path $root $launcher) $stage
}
Copy-Item (Join-Path $root 'gui.bat') $stage
New-Item -ItemType Directory -Path (Join-Path $stage 'scripts') -Force | Out-Null
Copy-Item (Join-Path $root 'scripts\quickbench.bat') (Join-Path $stage 'quickbench.bat')
Copy-Item (Join-Path $root 'bench-llama.bat') (Join-Path $stage 'bench-llama.bat')
Copy-Item (Join-Path $root 'scripts\bench-llama.ps1') (Join-Path $stage 'scripts')
Copy-Item (Join-Path $root 'scripts\fetch-checkpoint.ps1') (Join-Path $stage 'scripts')
@{
    profile = $Profile
    cudaArchitecture = $architecture
    version = $version
    builtAtUtc = (Get-Date).ToUniversalTime().ToString('o')
    modelIncluded = $false
    windowsAppRuntime = '1.7 (install separately)'
} | ConvertTo-Json | Set-Content (Join-Path $stage 'build-info.json') -Encoding UTF8

if (Test-Path $zip) { Remove-Item $zip -Force }
Compress-Archive -Path (Join-Path $stage '*') -DestinationPath $zip
Get-FileHash $zip -Algorithm SHA256 | ForEach-Object {
    "$($_.Hash)  $([IO.Path]::GetFileName($zip))"
} | Set-Content "$zip.sha256" -Encoding ASCII
Write-Host "Package: $zip" -ForegroundColor Green
Write-Host "SHA256:  $zip.sha256" -ForegroundColor Green
