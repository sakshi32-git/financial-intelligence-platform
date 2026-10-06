@echo off
title Eye Control Standalone Test
cd /d "%~dp0"
echo ====================================================
echo   Testing Eye Control Standalone
echo   (Press Ctrl+C in this window to exit)
echo ====================================================
echo.
call .venv\Scripts\activate
python run_eye_control.py
pause
