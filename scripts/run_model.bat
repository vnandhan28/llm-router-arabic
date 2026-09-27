@echo off
REM Run one model over the sample in its own window (Windows stand-in for tmux).
REM Usage: scripts\run_model.bat weak   or   scripts\run_model.bat strong
REM Safe to rerun: finished questions are skipped.
cd /d "%~dp0.."
title %1 model run
set PYTHONIOENCODING=utf-8
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
.venv\Scripts\python.exe -u -m src.query_models --role %1 --log data\run_%1.log
echo.
echo === finished (exit code %ERRORLEVEL%) - you can close this window ===
pause
