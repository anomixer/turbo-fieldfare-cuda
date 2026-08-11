<#
.SYNOPSIS
    Download the pinned source checkpoint that the M1a repacker consumes.

.DESCRIPTION
    Fetches mlx-community/gemma-4-26b-a4b-it-4bit at the exact revision upstream
    pins (SupportedModelSource.swift), using git-lfs rather than the Hugging Face
    Python CLI so no Python install is required.

    The clone runs with GIT_LFS_SKIP_SMUDGE so the initial checkout pulls only LFS
    pointers - a few hundred KB. The pinned revision is checked out first, then
    `git lfs pull` fetches the ~15 GB of real weight data for that revision only.

    Safe to re-run: git-lfs resumes rather than restarting.

.PARAMETER Destination
    Where to place the checkout. Defaults to C:\Users\<you>\model-data\... which
    sits on the NVMe. Do not point this at a HDD - the repacker reads it linearly
    but the resulting .gturbo must live on NVMe for streaming to make sense.
#>
[CmdletBinding()]
param(
    [string]$Destination = (Join-Path $env:USERPROFILE 'model-data\gemma-4-26b-a4b-it-4bit')
)

$ErrorActionPreference = 'Stop'

$RepoUrl  = 'https://huggingface.co/mlx-community/gemma-4-26b-a4b-it-4bit'
$Revision = '0d77464eeb233a2da68ebf9d7dc4edaac7db956d'
$ApproxBytes = 16GB

if (-not (Get-Command git -ErrorAction SilentlyContinue)) { throw 'git not found on PATH.' }
& git lfs version *> $null
if ($LASTEXITCODE -ne 0) { throw 'git-lfs not installed. Install Git for Windows with the LFS component.' }

$parent = Split-Path -Parent $Destination
if (-not (Test-Path $parent)) { New-Item -ItemType Directory -Path $parent -Force | Out-Null }

$driveLetter = (Split-Path -Qualifier $Destination).TrimEnd(':')
$free = (Get-PSDrive -Name $driveLetter).Free
if ($free -lt $ApproxBytes) {
    throw ("Need ~{0:N0} GB free on {1}: but only {2:N1} GB available." -f ($ApproxBytes/1GB), $driveLetter, ($free/1GB))
}

Write-Host "Repo        : $RepoUrl"
Write-Host "Revision    : $Revision"
Write-Host "Destination : $Destination"
Write-Host ("Free on {0}: : {1:N1} GB`n" -f $driveLetter, ($free/1GB))

# Phase 1: pointer-only clone. Fast, and lets us pin the revision before any
# large object is transferred.
if (-not (Test-Path (Join-Path $Destination '.git'))) {
    Write-Host '[1/3] Cloning pointers (GIT_LFS_SKIP_SMUDGE=1)...'
    $env:GIT_LFS_SKIP_SMUDGE = '1'
    & git clone --no-checkout $RepoUrl $Destination
    if ($LASTEXITCODE -ne 0) { throw "git clone failed ($LASTEXITCODE)." }
} else {
    Write-Host '[1/3] Existing checkout found, reusing.'
    $env:GIT_LFS_SKIP_SMUDGE = '1'
}

Write-Host "[2/3] Checking out pinned revision $Revision..."
& git -C $Destination checkout --force $Revision
if ($LASTEXITCODE -ne 0) { throw "git checkout failed ($LASTEXITCODE) - revision missing from remote?" }

# Phase 2: the actual weight transfer. Resumable.
Write-Host '[3/3] Fetching LFS objects (~15 GB, resumable)...'
Remove-Item Env:\GIT_LFS_SKIP_SMUDGE -ErrorAction SilentlyContinue
& git -C $Destination lfs pull
if ($LASTEXITCODE -ne 0) { throw "git lfs pull failed ($LASTEXITCODE). Re-run to resume." }

$actual = (Get-ChildItem $Destination -Recurse -File -Force |
           Where-Object { $_.FullName -notlike '*\.git\*' } |
           Measure-Object -Property Length -Sum).Sum

Write-Host ("`nOK - {0:N1} GB in {1}" -f ($actual/1GB), $Destination) -ForegroundColor Green
Get-ChildItem $Destination -File |
    Sort-Object Length -Descending |
    Select-Object Name, @{n='SizeMB'; e={[math]::Round($_.Length/1MB, 1)}} |
    Format-Table -AutoSize
