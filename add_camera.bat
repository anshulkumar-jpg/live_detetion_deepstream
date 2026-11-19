@echo off
title Add camera - DeepStream VMS
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
REM The script checks the gateway's TLS certificate. A self-signed certificate at deploy\envoy\certs\tls.crt is trusted
REM automatically; for another CA set VMS_CA_BUNDLE=path\to\ca.pem before running.
python scripts\add_camera.py %*
echo.
pause
