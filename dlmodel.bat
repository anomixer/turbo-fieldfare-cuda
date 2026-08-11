@echo off
setlocal EnableExtensions

set "ROOT=%~dp0"
set "MODEL=%ROOT%models\gemma4.gturbo"

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

powershell -NoProfile -ExecutionPolicy Bypass -File "%ROOT%scripts\fetch-checkpoint.ps1" -OutputDir "%ROOT%models"
if errorlevel 1 exit /b 1

if not exist "%MODEL%" (
  echo ERROR: Download completed but model not found at:
  echo   %MODEL%
  exit /b 1
)

echo.
echo Model installed at:
echo   %MODEL%
exit /b 0
