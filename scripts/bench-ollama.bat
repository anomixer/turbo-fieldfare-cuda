@echo off
setlocal EnableExtensions DisableDelayedExpansion

rem Ollama baseline benchmark. This uses OLLAMA_GPU_OVERHEAD as a reservation;
rem Ollama does not expose TurboFieldfare's exact VRAM-budget planner.
set "MODEL=%~1"
if "%MODEL%"=="" set "MODEL=gemma-4:26b-a4b"
set "PORT=11435"
set "HOST=http://127.0.0.1:%PORT%"
set "ROOT=%~dp0.."
set "LOGDIR=%ROOT%\benchlogs\ollama"
if not exist "%LOGDIR%" mkdir "%LOGDIR%"

where ollama >nul 2>&1 || (echo ERROR: ollama.exe not found on PATH.& exit /b 1)
where curl >nul 2>&1 || (echo ERROR: curl.exe not found on PATH.& exit /b 1)

echo Pull the model first if needed: ollama pull %MODEL%
echo This benchmark starts a private Ollama server on port %PORT%.
echo Close any existing Ollama server before starting.
echo.

for %%B in (4 6 8 10 12) do call :run %%B
call :run full
echo.
echo Results are in %LOGDIR%
exit /b 0

:run
set "LABEL=%~1"
set "OVERHEAD=0"
if not "%LABEL%"=="full" set /a OVERHEAD=(24-%LABEL%)*1073741824
set "LOG=%LOGDIR%\ollama_%LABEL%GB.log"
if "%LABEL%"=="full" set "LOG=%LOGDIR%\ollama_full.log"
echo.
if "%LABEL%"=="full" (echo Running Ollama full-resident baseline) else (echo Running Ollama %LABEL%GB requested baseline)
echo requested_vram=%LABEL%GB overhead=%OVERHEAD% > "%LOG%"

set "OLLAMA_HOST=127.0.0.1:%PORT%"
set "OLLAMA_GPU_OVERHEAD=%OVERHEAD%"
set "OLLAMA_NUM_PARALLEL=1"
set "OLLAMA_CONTEXT_LENGTH=4096"
start "ollama-bench" /b ollama serve >nul 2>&1
for /l %%W in (1,1,30) do (
  curl -fsS "%HOST%/api/tags" >nul 2>&1 && goto :ready
  timeout /t 1 /nobreak >nul
)
echo ERROR: Ollama server did not become ready. >> "%LOG%"
goto :stop

:ready
echo model=%MODEL% >> "%LOG%"
echo device= >> "%LOG%"
nvidia-smi --query-gpu=name,memory.used,memory.free,memory.total --format=csv,noheader >> "%LOG%" 2>&1
set "JSON=%TEMP%\ollama-bench-%RANDOM%.json"
curl -fsS "%HOST%/api/generate" -H "Content-Type: application/json" -d "{\"model\":\"%MODEL%\",\"prompt\":\"Write a practical design review for running a 26B Mixture-of-Experts coding assistant. Compare VRAM budgets and give a recommendation.\",\"stream\":false,\"options\":{\"num_predict\":512,\"temperature\":0.2}}" > "%JSON%"
if errorlevel 1 (echo ERROR: generate request failed. >> "%LOG%") else (powershell -NoProfile -Command "$j=Get-Content -Raw '%JSON%'|ConvertFrom-Json; 'eval_count='+$j.eval_count; 'eval_duration_ns='+$j.eval_duration; 'prompt_eval_count='+$j.prompt_eval_count; 'prompt_eval_duration_ns='+$j.prompt_eval_duration" >> "%LOG%")
del "%JSON%" >nul 2>&1

:stop
for /f "tokens=5" %%P in ('netstat -ano ^| findstr ":%PORT% .*LISTENING"') do taskkill /PID %%P /F >nul 2>&1
type "%LOG%"
exit /b 0
