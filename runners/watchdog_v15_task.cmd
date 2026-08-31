@echo off
REM V15 watchdog for Task Scheduler. Module form only.
cd /d "%~dp0.."
".venv\Scripts\python.exe" -X faulthandler -m runners.watchdog_v15 --stage all
