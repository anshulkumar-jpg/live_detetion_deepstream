@echo off
title DeepStream VMS - stop all Docker services
cd /d "%~dp0"
where docker >nul 2>&1
if errorlevel 1 (
  echo [ERROR] Docker was not found.
  pause
  exit /b 1
)
echo Stopping all DeepStream VMS services - the containers are kept, only stopped...
docker compose --profile analytics --profile monitoring stop
echo.
echo Done. The containers still exist, stopped. Start them again with start_services.bat
pause
