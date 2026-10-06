@echo off
title Financial Intelligence Platform Dashboard
cd /d "%~dp0"
echo ====================================================
echo   Starting Financial Intelligence Platform
echo   (Interactive Session - Eye Control Enabled)
echo ====================================================
echo.
call .venv\Scripts\activate
python src\dashboard\app.py
pause
