@echo off
setlocal EnableExtensions

set "ROOT=%~dp0"
set "MODEL=%ROOT%models\gemma4.gturbo"
set "REPACK=%ROOT%build\relwithdebinfo\bin\tf-repack.exe"
if not exist "%REPACK%" set "REPACK=%ROOT%bin\tf-repack.exe"

where git >nul 2>&1
if errorlevel 1 (
  echo Git for Windows is required to download the checkpoint with Git LFS.
  choice /C YN /N /M "Install Git for Windows now? [Y/N] "
  if errorlevel 2 exit /b 1
  where winget >nul 2>&1
  if errorlevel 1 (
    echo ERROR: winget was not found. Install App Installer, then run dlmodel.bat again.
    exit /b 1
  )
  winget install --id Git.Git -e --accept-package-agreements --accept-source-agreements
  if errorlevel 1 exit /b 1
  if exist "%ProgramFiles%\Git\cmd\git.exe" set "PATH=%ProgramFiles%\Git\cmd;%PATH%"
  if exist "%ProgramFiles(x86)%\Git\cmd\git.exe" set "PATH=%ProgramFiles(x86)%\Git\cmd;%PATH%"
  echo Git installed. Restarting dlmodel.bat...
  call "%~f0" %*
  exit /b %ERRORLEVEL%
)

if exist "%MODEL%" (
  echo Model already exists at:
  echo   %MODEL%
  echo Skipping download.
  exit /b 0
)

set "LEGACY=C:\models\gemma4.gturbo"
if exist "%LEGACY%" (
  echo Model already exists at:
  echo   %LEGACY%
  echo Skipping download.
  exit /b 0
)

echo Model not found at:
echo   %MODEL%
echo   %LEGACY%
echo.
echo Downloading Gemma 4 26B-A4B checkpoint...
echo.

if not exist "%ROOT%models" mkdir "%ROOT%models"

powershell -NoProfile -ExecutionPolicy Bypass -File "%ROOT%scripts\fetch-checkpoint.ps1" -Destination "%ROOT%models"
if errorlevel 1 exit /b 1

if not exist "%REPACK%" (
  echo ERROR: tf-repack.exe was not found.
  echo Build the project or extract a complete prebuilt package first.
  exit /b 1
)

if not exist "%MODEL%\manifest.json" (
  echo.
  echo Repacking checkpoint into:
  echo   %MODEL%
  "%REPACK%" --checkpoint "%ROOT%models" --output "%MODEL%"
  if errorlevel 1 exit /b 1
)

if not exist "%MODEL%\manifest.json" (
  echo ERROR: Repack completed but model manifest was not found at:
  echo   %MODEL%
  exit /b 1
)

echo.
echo Model installed at:
echo   %MODEL%
exit /b 0
