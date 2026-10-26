@echo off
setlocal
REM HotelSight - verification, automated tests and launch check
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" ( echo Run setup_windows.bat first. & pause & exit /b 1 )
".venv\Scripts\python.exe" "scripts\verify_experiment.py"
".venv\Scripts\python.exe" -m pytest
".venv\Scripts\python.exe" "scripts\launch_check.py"
pause
