@echo off
REM V15 launcher for Task Scheduler. Module form only -- see run_v15_wrapped.ps1 for why.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_v15_wrapped.ps1" -Stage all
