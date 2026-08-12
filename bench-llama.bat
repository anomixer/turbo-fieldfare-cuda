@echo off
setlocal EnableExtensions
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\bench-llama.ps1" %*
exit /b %ERRORLEVEL%
