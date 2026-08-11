@echo off
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\bootstrap-windows.ps1" %*
if errorlevel 1 exit /b %ERRORLEVEL%

echo.
echo Building TurboFieldfare GUI...
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\build-gui.ps1" -Configuration Release
exit /b %ERRORLEVEL%
