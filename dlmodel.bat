@echo off
setlocal EnableExtensions

set "ROOT=%~dp0"
set "MODEL=%ROOT%models\gemma4.gturbo"
set "REPACK=%ROOT%build\relwithdebinfo\bin\tf-repack.exe"
if not exist "%REPACK%" set "REPACK=%ROOT%bin\tf-repack.exe"

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
