@echo off
setlocal EnableExtensions

set "ROOT=%~dp0"
set "EXE=%ROOT%build\gui\Release\TurboFieldfare.exe"
if not exist "%EXE%" set "EXE=%ROOT%gui\TurboFieldfare.exe"

if not exist "%EXE%" (
  echo ERROR: TurboFieldfare GUI was not found.
  echo Expected:
  echo   %ROOT%build\gui\Release\TurboFieldfare.exe
  echo   %ROOT%gui\TurboFieldfare.exe
  echo Build the GUI or extract a prebuilt package first.
  exit /b 1
)

echo Starting TurboFieldfare GUI...
start "TurboFieldfare" /D "%~dp0" "%EXE%" %*
exit /b 0
