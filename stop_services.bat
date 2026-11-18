@echo off
title DeepStream VMS - stop all Docker services
cd /d "%~dp0"
echo Stopping all DeepStream VMS services - your data and models are kept...
docker compose --profile analytics --profile monitoring down
echo.
echo Done. Start again with start_services.bat
pause
