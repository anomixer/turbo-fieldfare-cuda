param(
    [ValidateSet('auto', 'legacy', 'modern')]
    [string]$Profile = 'auto'
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$cudaRoot = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA'

function Stop-Bootstrap([string]$Message) {
    Write-Host ''
    Write-Host "ERROR: $Message" -ForegroundColor Red
    exit 1
}

function Get-CommandPath([string]$Name) {
    $command = Get-Command $Name -ErrorAction SilentlyContinue
    if ($null -eq $command) { return $null }
    return $command.Source
}

function Confirm-Install([string]$Label, [string]$WingetArguments) {
    $answer = Read-Host "Missing $Label. Install it now? [Y/n]"
    if ($answer -match '^(|y|yes)$') {
        if (-not (Get-CommandPath 'winget')) {
            Stop-Bootstrap 'winget is required for automatic installation. Install App Installer from Microsoft Store, then run build again.'
        }
        Write-Host "Installing $Label. Windows may show an administrator prompt..." -ForegroundColor Yellow
        $installer = Start-Process -FilePath 'winget' -ArgumentList $WingetArguments -Verb RunAs -Wait -PassThru
        if ($installer.ExitCode -ne 0) {
            Stop-Bootstrap "The $Label installer returned exit code $($installer.ExitCode)."
        }

        Write-Host "Finished installing $Label. Rechecking the toolchain..." -ForegroundColor Green
        # Installers commonly update PATH only in newly-created processes.
        # Restart this script automatically so the user does not have to run
        # build.bat again after every missing dependency.
        $machinePath = [Environment]::GetEnvironmentVariable('Path', 'Machine')
        $userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
        $env:Path = "$machinePath;$userPath"
        $child = Start-Process -FilePath 'powershell.exe' -ArgumentList @(
                '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $PSCommandPath,
                '-Profile', $Profile) -Wait -PassThru
        exit $child.ExitCode
    }
    Stop-Bootstrap "$Label is required to build TurboFieldfare CUDA."
}

$nvidiaSmi = Get-CommandPath 'nvidia-smi'
if (-not $nvidiaSmi) {
    Stop-Bootstrap 'NVIDIA driver was not detected. Install a current NVIDIA driver, reboot if requested, then run build again.'
}

try {
    $gpuLine = (& $nvidiaSmi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | Select-Object -First 1).Trim()
} catch {
    Stop-Bootstrap 'nvidia-smi could not query the GPU. Update or reinstall the NVIDIA driver, then run build again.'
}

if ($gpuLine -notmatch '^(?<name>.*?),\s*(?<cap>\d+\.\d+),\s*(?<driver>.+)$') {
    Stop-Bootstrap "Could not parse nvidia-smi GPU data: $gpuLine"
}

$gpuName = $Matches.name.Trim()
$computeCapability = $Matches.cap
$driverVersion = $Matches.driver.Trim()
$sm = [int]($computeCapability -replace '\.', '')
$profile = if ($Profile -eq 'auto') {
    if ($sm -lt 75) { 'legacy' } else { 'modern' }
} else {
    $Profile
}
$requiredCudaMajor = if ($profile -eq 'legacy') { 12 } else { $null }

Write-Host ''
Write-Host '=== TurboFieldfare CUDA Windows setup ===' -ForegroundColor Cyan
Write-Host "GPU: $gpuName"
Write-Host "Compute capability: $computeCapability (sm_$sm)"
Write-Host "NVIDIA driver: $driverVersion"
Write-Host "Selected build profile: $profile"

$nvcc = Get-CommandPath 'nvcc'
$cudaHome = $null
if ($nvcc) {
    $candidate = Split-Path -Parent (Split-Path -Parent $nvcc)
    if (Test-Path (Join-Path $candidate 'bin\nvcc.exe')) { $cudaHome = $candidate }
}

if (-not $cudaHome -and (Test-Path $cudaRoot)) {
    $toolkits = Get-ChildItem $cudaRoot -Directory -ErrorAction SilentlyContinue | Sort-Object Name -Descending
    if ($profile -eq 'legacy') {
        $toolkits = $toolkits | Where-Object { $_.Name -match '^v12\.' }
    }
    $toolkit = $toolkits | Select-Object -First 1
    if ($toolkit) { $cudaHome = $toolkit.FullName }
}

if ($cudaHome) {
    $nvccVersion = (& (Join-Path $cudaHome 'bin\nvcc.exe') --version | Select-String 'release\s+(?<version>\d+\.\d+)' | Select-Object -First 1)
    if ($nvccVersion -and $nvccVersion.Matches[0].Groups['version'].Value -match '^(?<major>\d+)\.') {
        $cudaMajor = [int]$Matches.major
    }
}

if (-not $cudaHome -or ($requiredCudaMajor -and $cudaMajor -ne $requiredCudaMajor)) {
    $cudaVersion = if ($profile -eq 'legacy') { '12.6' } else { '12.8' }
    Confirm-Install "NVIDIA CUDA Toolkit $cudaVersion" "install --id Nvidia.CUDA --version $cudaVersion -e --accept-package-agreements --accept-source-agreements"
}

$vsNinja = Get-ChildItem (Join-Path ${env:ProgramFiles} 'Microsoft Visual Studio\2022') `
        -Filter 'ninja.exe' -Recurse -ErrorAction SilentlyContinue |
        Select-Object -First 1
if ($vsNinja) {
    $env:Path = "$(Split-Path -Parent $vsNinja.FullName);$env:Path"
}

if (-not (Get-CommandPath 'git')) { Confirm-Install 'Git for Windows' 'install --id Git.Git -e --accept-package-agreements --accept-source-agreements' }
if (-not (Get-CommandPath 'cmake')) { Confirm-Install 'CMake' 'install --id Kitware.CMake -e --accept-package-agreements --accept-source-agreements' }
if (-not (Get-CommandPath 'ninja')) { Confirm-Install 'Ninja' 'install --id Ninja-build.Ninja -e --accept-package-agreements --accept-source-agreements' }

$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
$vsDevCmd = $null
if (Test-Path $vswhere) {
    $vsInstall = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if ($vsInstall) { $vsDevCmd = Join-Path $vsInstall 'Common7\Tools\VsDevCmd.bat' }
}
if (-not $vsDevCmd -or -not (Test-Path $vsDevCmd)) {
    Confirm-Install 'Visual Studio 2022 Build Tools with C++ tools' 'install --id Microsoft.VisualStudio.2022.BuildTools -e --override "--wait --passive --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended" --accept-package-agreements --accept-source-agreements'
}

# The framework package alone is not enough for the unpackaged WinUI GUI.
# Bootstrap requires the matching Main package as well; winget's runtime
# installer installs the framework, Main, Singleton and DDLM packages.
$runtimeMain = Get-AppxPackage -Name 'MicrosoftCorporationII.WinAppRuntime.Main.1.7' `
        -ErrorAction SilentlyContinue | Where-Object Architecture -eq 'X64' | Select-Object -First 1
if (-not $runtimeMain) {
    Confirm-Install 'Windows App Runtime 1.7 (x64)' 'install --id Microsoft.WindowsAppRuntime.1.7 --source winget --accept-package-agreements --accept-source-agreements'
}

$buildDirectory = Join-Path $root 'build\relwithdebinfo'
$ninjaHome = Split-Path -Parent (Get-CommandPath 'ninja')
$generatedBatch = Join-Path $env:TEMP 'turbo-fieldfare-build.cmd'
@"
@echo off
call "$vsDevCmd" -no_logo -host_arch=x64 -arch=x64
if errorlevel 1 exit /b 1
set "PATH=$cudaHome\bin;$ninjaHome;%PATH%"
cmake -S "$root" -B "$buildDirectory" -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_CUDA_COMPILER="$cudaHome\bin\nvcc.exe" -DTF_CUDA_ARCHITECTURES=$sm-real
if errorlevel 1 exit /b 1
cmake --build "$buildDirectory" --parallel
if errorlevel 1 exit /b 1
"$buildDirectory\bin\tf-cli.exe" --version >nul
if errorlevel 1 exit /b 1
echo tf-cli smoke test passed.
"@ | Set-Content -Path $generatedBatch -Encoding ascii

Write-Host ''
Write-Host "Building sm_$sm with CUDA at $cudaHome..." -ForegroundColor Cyan
& cmd.exe /d /c $generatedBatch
$buildExitCode = $LASTEXITCODE
Remove-Item $generatedBatch -Force -ErrorAction SilentlyContinue
if ($buildExitCode -ne 0) { Stop-Bootstrap 'Build failed. Read the compiler output above.' }

Write-Host ''
Write-Host 'SUCCESS: build completed.' -ForegroundColor Green
Write-Host "Binaries: $buildDirectory\bin"
