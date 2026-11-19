@echo off
title DeepStream VMS - all cameras live
cd /d "%~dp0"
where python >nul 2>&1
if errorlevel 1 (
  echo [ERROR] Python was not found. Install Python 3.10 or newer first.
  pause
  exit /b 1
)
if not exist ".env" (
  echo [ERROR] .env not found. Run start_services.bat first.
  pause
  exit /b 1
)
echo Starting DeepStream VMS: services, models, all cameras, live viewer...
echo.
python scripts\run_all.py --watch %*
echo.
echo Stopped watching. The cameras keep running in the background.
pause
