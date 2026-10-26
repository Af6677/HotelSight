@echo off
setlocal
REM HotelSight - start the dashboard and open it in the default browser
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" ( echo Run setup_windows.bat first. & pause & exit /b 1 )
echo Starting HotelSight at http://localhost:8501  (close this window to stop it)
start "" cmd /c "timeout /t 8 /nobreak >nul & start http://localhost:8501"
".venv\Scripts\python.exe" -m streamlit run "app\streamlit_app.py" --server.port 8501
pause
