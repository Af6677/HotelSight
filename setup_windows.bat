@echo off
setlocal
REM HotelSight - one-time setup. Creates a separate Python 3.12 environment in .venv
cd /d "%~dp0"
echo.
echo [1/4] Creating Python 3.12 virtual environment in "%~dp0.venv"
py -3.12 -m venv ".venv"
if errorlevel 1 (
  echo Python 3.12 was not found. Install Python 3.12 from python.org ^(tick "py launcher"^), then run this again.
  pause & exit /b 1
)
echo [2/4] Upgrading pip
".venv\Scripts\python.exe" -m pip install --upgrade pip
echo [3/4] Installing the exact package versions (this can take several minutes)
".venv\Scripts\python.exe" -m pip install -r "requirements.txt"
if errorlevel 1 ( echo Package installation failed - see the messages above. & pause & exit /b 1 )
echo [4/4] Verifying the saved model against the frozen experiment
".venv\Scripts\python.exe" "scripts\verify_experiment.py"
if errorlevel 1 ( echo Verification failed - do not use the dashboard until this passes. & pause & exit /b 1 )
echo.
echo Setup complete. Double-click run_dashboard.bat to start HotelSight.
pause
