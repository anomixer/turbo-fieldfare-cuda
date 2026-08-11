@echo off
setlocal EnableExtensions DisableDelayedExpansion

set "ROOT=%~dp0"
if not exist "%ROOT%bin" set "ROOT=%ROOT%..\"
cd /d "%ROOT%"

set "MODEL=%~1"
if "%MODEL%"=="" set "MODEL=%ROOT%models\gemma4.gturbo"
if not exist "%MODEL%" set "MODEL=C:\models\gemma4.gturbo"
set "EXE=%ROOT%build\relwithdebinfo\bin\tf-cli.exe"
if not exist "%EXE%" set "EXE=%ROOT%bin\tf-cli.exe"
set "LOGDIR=%CD%\benchlogs"
set "PROMPT=Write a practical design review for running a 26B Mixture-of-Experts coding assistant on a laptop. Compare 4 GB, 6 GB, 8 GB, 10 GB, and 12 GB VRAM. Explain expert caching, SSD streaming, three failure modes, and give a final recommendation. Use Markdown headings and a table. Aim for about 350 words."

if not exist "%EXE%" (
    echo ERROR: tf-cli.exe not found: %EXE%
    exit /b 1
)
if not exist "%MODEL%" (
    echo ERROR: model not found: %MODEL%
    exit /b 1
)
set "GPU_APPS="
for /f "usebackq delims=" %%P in (`nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2^>nul`) do (
    set "GPU_APPS=1"
    echo WARNING: CUDA process already owns VRAM: %%P
)
if defined GPU_APPS (
    echo.
    echo ERROR: Stop the listed CUDA processes before running quickbench.
    echo Otherwise the 10GB, 12GB, and full-resident runs can fail with out-of-memory.
    exit /b 2
)
if not exist "%LOGDIR%" mkdir "%LOGDIR%"

for %%B in (4 6 8 10 12) do call :run %%B
call :run full

echo.
echo Completed. Summary lines from all runs:
for %%F in ("%LOGDIR%\quickbench_4GB.log" "%LOGDIR%\quickbench_6GB.log" "%LOGDIR%\quickbench_8GB.log" "%LOGDIR%\quickbench_10GB.log" "%LOGDIR%\quickbench_12GB.log" "%LOGDIR%\quickbench_full.log") do (
    echo.
    echo --- %%~nxF ---
    findstr /R /C:"^  total" /C:"^  layers" /C:"^  slots" /C:"^[0-9][0-9]* prompt tokens" /C:"^Expert cache:" "%%~fF"
)
exit /b 0

:run
set "LABEL=%~1"
set "LOG=%LOGDIR%\quickbench_%LABEL%GB.log"
if /I "%LABEL%"=="full" set "LOG=%LOGDIR%\quickbench_full.log"

echo.
if /I "%LABEL%"=="full" (
    echo Running full-resident VRAM benchmark
) else (
    echo Running %LABEL%GB VRAM benchmark
)
if /I "%LABEL%"=="full" (
    "%EXE%" --model "%MODEL%" --backend cuda --verbose --prompt "%PROMPT%" > "%LOG%" 2>&1
) else (
    "%EXE%" --model "%MODEL%" --backend cuda --vram-budget %LABEL% --verbose --prompt "%PROMPT%" > "%LOG%" 2>&1
)
type "%LOG%"
exit /b 0
