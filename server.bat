@echo off
setlocal EnableExtensions

set "ROOT=%~dp0"
set "EXE=%ROOT%build\relwithdebinfo\bin\tf-server.exe"
if not exist "%EXE%" set "EXE=%ROOT%bin\tf-server.exe"

if not exist "%EXE%" (
  echo ERROR: tf-server.exe was not found.
  echo Run build first, then try server again.
  exit /b 1
)

set "MODEL=%ROOT%models\gemma4.gturbo"
if not exist "%MODEL%" set "MODEL=C:\models\gemma4.gturbo"

if not exist "%MODEL%" (
  echo ERROR: Model was not found:
  echo   %MODEL%
  echo Run dlmodel first, then try server again.
  exit /b 1
)

echo Starting TurboFieldfare server...
echo Model: %MODEL%
echo.
"%EXE%" --model "%MODEL%" %*
exit /b %ERRORLEVEL%
