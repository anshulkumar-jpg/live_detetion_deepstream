@echo off
title DeepStream VMS - all cameras live
cd /d "%~dp0"
echo Starting DeepStream VMS: services, models, all cameras, live viewer...
echo.
python scripts\run_all.py --watch
echo.
echo Stopped watching. The cameras keep running in the background.
pause
