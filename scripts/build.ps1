<#
.SYNOPSIS
    Configure, build and test turbofieldfare-win from a plain PowerShell prompt.

.DESCRIPTION
    Neither cl.exe nor ninja.exe are on PATH on a stock machine: MSVC needs
    vcvars64.bat, and CMake/Ninja ship inside CLion. This script locates all
    three and drives the build so CI and a local shell behave identically.
    CLion users can ignore it and select a CMakePresets entry instead.

.PARAMETER Preset
    A configure preset from CMakePresets.json. Defaults to relwithdebinfo.

.PARAMETER Fresh
    Delete the preset's binary directory before configuring.

.PARAMETER SkipTests
    Build only; do not run ctest.

.EXAMPLE
    scripts\build.ps1
    scripts\build.ps1 -Preset debug -Fresh
#>
[CmdletBinding()]
param(
    [ValidateSet('debug', 'relwithdebinfo', 'release', 'nocuda')]
    [string]$Preset = 'relwithdebinfo',
    [switch]$Fresh,
    [switch]$SkipTests
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot

# A shell started before the CUDA Toolkit was installed carries a stale PATH and
# no CUDA_PATH, so check_language(CUDA) silently reports NOTFOUND. Re-read the
# environment from the registry so an install mid-session is picked up without
# needing a new terminal.
function Sync-MachineEnvironment {
    foreach ($name in 'CUDA_PATH', 'CUDA_PATH_V13_3') {
        $value = [Environment]::GetEnvironmentVariable($name, 'Machine')
        if ($value -and -not [Environment]::GetEnvironmentVariable($name)) {
            Set-Item -Path "Env:\$name" -Value $value
        }
    }

    $machinePath = [Environment]::GetEnvironmentVariable('Path', 'Machine')
    $userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
    $merged = @($machinePath, $userPath, $env:Path) -ne $null -join ';'

    $seen = [System.Collections.Generic.HashSet[string]]::new(
        [System.StringComparer]::OrdinalIgnoreCase)
    $ordered = foreach ($entry in ($merged -split ';')) {
        $trimmed = $entry.Trim()
        if ($trimmed -and $seen.Add($trimmed)) { $trimmed }
    }
    $env:Path = $ordered -join ';'
}

Sync-MachineEnvironment

function Find-First {
    param([string[]]$Candidates, [string]$What)
    foreach ($c in $Candidates) {
        if ($c -and (Test-Path $c)) { return $c }
    }
    throw "Could not locate $What. Looked in:`n  $($Candidates -join "`n  ")"
}

# --- MSVC ------------------------------------------------------------------
$vswhere = 'C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe'
if (-not (Test-Path $vswhere)) {
    throw 'vswhere.exe not found. Install Visual Studio 2022 Build Tools with the C++ workload.'
}
$vsRoot = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vsRoot) { throw 'No Visual Studio install with the MSVC C++ toolset was found.' }
$vcvars = Join-Path $vsRoot 'VC\Auxiliary\Build\vcvars64.bat'
if (-not (Test-Path $vcvars)) { throw "vcvars64.bat missing under $vsRoot." }

# --- CMake and Ninja -------------------------------------------------------
# Prefer anything already on PATH, then fall back to CLion's bundled copies.
$clion = Get-ChildItem "$env:LOCALAPPDATA\Programs" -Filter 'CLion*' -Directory -ErrorAction SilentlyContinue |
         Sort-Object Name -Descending | Select-Object -First 1

$cmake = Find-First @(
    (Get-Command cmake -ErrorAction SilentlyContinue).Source
    if ($clion) { Join-Path $clion.FullName 'bin\cmake\win\x64\bin\cmake.exe' }
) 'cmake.exe'

$ninja = Find-First @(
    (Get-Command ninja -ErrorAction SilentlyContinue).Source
    if ($clion) { Join-Path $clion.FullName 'bin\ninja\win\x64\ninja.exe' }
    (Get-ChildItem 'C:\Program Files\Microsoft Visual Studio\2022' -Filter 'ninja.exe' -Recurse -ErrorAction SilentlyContinue |
        Select-Object -First 1 -ExpandProperty FullName)
) 'ninja.exe'

$ctest = Join-Path (Split-Path -Parent $cmake) 'ctest.exe'

Write-Host "cmake  : $cmake"
Write-Host "ninja  : $ninja"
Write-Host "msvc   : $vsRoot"
Write-Host "preset : $Preset`n"

$binaryDir = Join-Path $repoRoot "build\$Preset"
if ($Fresh -and (Test-Path $binaryDir)) {
    Write-Host "Removing $binaryDir"
    Remove-Item $binaryDir -Recurse -Force
}

# vcvars64 only exports into a cmd session, so each phase runs through cmd with
# the environment freshly established.
function Invoke-InDevShell {
    param([string]$CommandLine, [string]$Phase)
    $full = "`"$vcvars`" >nul 2>&1 && $CommandLine"
    & cmd /c $full
    if ($LASTEXITCODE -ne 0) {
        throw "$Phase failed with exit code $LASTEXITCODE."
    }
}

Invoke-InDevShell "`"$cmake`" -S `"$repoRoot`" --preset $Preset -DCMAKE_MAKE_PROGRAM=`"$ninja`"" 'Configure'
Invoke-InDevShell "`"$cmake`" --build `"$binaryDir`"" 'Build'

if (-not $SkipTests) {
    Invoke-InDevShell "`"$ctest`" --test-dir `"$binaryDir`" --output-on-failure" 'Tests'
}

Write-Host "`nOK - $Preset" -ForegroundColor Green
