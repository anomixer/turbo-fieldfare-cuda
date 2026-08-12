# Builds the WinUI 3 front end.
#
# Separate from scripts/build.ps1 because the GUI is the one target MSBuild has
# to own: WinUI's package injects targets a CMake-generated project never runs.
# The libraries it links come from the CMake build, so that has to happen first.

[CmdletBinding()]
param(
    [ValidateSet('Release', 'Debug')]
    [string]$Configuration = 'Release',
    [switch]$SkipLibraries
)

$ErrorActionPreference = 'Stop'
$env:VSLANG = '1033'
$root = Split-Path -Parent $PSScriptRoot

$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
$msbuild = $null
if (Test-Path $vswhere) {
    $msbuild = & $vswhere -latest -products * -find 'MSBuild\**\Bin\MSBuild.exe' |
               Select-Object -First 1
}
if (-not $msbuild) {
    $candidates = @(
        (Join-Path ${env:ProgramFiles} 'Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\MSBuild.exe'),
        (Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\2022\BuildTools\MSBuild\Current\Bin\MSBuild.exe')
    )
    $msbuild = $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
}
if (-not (Test-Path $msbuild)) {
    throw 'MSBuild not found. The GUI needs Visual Studio 2022 with the C++ desktop workload.'
}

if (-not $SkipLibraries) {
    Write-Host 'Building the static libraries first...' -ForegroundColor Cyan
    & (Join-Path $PSScriptRoot 'build.ps1') -Preset release -SkipTests
    if ($LASTEXITCODE -ne 0) { throw 'The library build failed.' }
}

$project = Join-Path $root 'src\gui\TurboFieldfare.vcxproj'
$common = @(
    "-p:Configuration=$Configuration",
    '-p:Platform=x64',
    "-p:SolutionDir=$root\",
    '-p:PreferredUILang=en-US',
    '-nologo',
    '-v:m'
)

Write-Host 'Restoring NuGet packages...' -ForegroundColor Cyan
& $msbuild $project -t:Restore @common '-v:q'
if ($LASTEXITCODE -ne 0) { throw 'NuGet restore failed.' }

Write-Host 'Building the GUI...' -ForegroundColor Cyan
& $msbuild $project @common
if ($LASTEXITCODE -ne 0) { throw 'The GUI build failed.' }

# The GUI starts tf-decode.exe from its own directory, so they have to sit
# together for a launch from the build output to work.
$outDir = Join-Path $root "build\gui\$Configuration"
$serviceCandidates = @(
    (Join-Path $root 'build\relwithdebinfo\bin\tf-decode.exe'),
    (Join-Path $root 'build\release\bin\tf-decode.exe')
)
$service = $serviceCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if ($service) {
    Copy-Item $service $outDir -Force
} else {
    throw 'tf-decode.exe was not found in build\release\bin or build\relwithdebinfo\bin.'
}

Write-Host "OK - $outDir\TurboFieldfare.exe" -ForegroundColor Green
