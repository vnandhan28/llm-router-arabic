@echo off
REM Run Phases 3-6 on the REAL routing table (after both model runs are finished),
REM then retrain the demo router. Outputs go to results\.
cd /d "%~dp0.."
set PRACTICE=
set PYTHONIOENCODING=utf-8
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
.venv\Scripts\python.exe -m src.query_models --table || goto :error
.venv\Scripts\python.exe -m src.labels || goto :error
.venv\Scripts\python.exe -m src.router || goto :error
.venv\Scripts\python.exe -m src.evaluate || goto :error
.venv\Scripts\python.exe -m src.failures || goto :error
.venv\Scripts\python.exe -m src.demo --retrain || goto :error
echo.
echo === analysis finished: see results\ ===
exit /b 0
:error
echo.
echo === a step FAILED (exit code %ERRORLEVEL%) - see the message above ===
exit /b 1
