@echo off
setlocal EnableDelayedExpansion
title DeepStream VMS - start all Docker services
cd /d "%~dp0"

echo ================================================================
echo   DeepStream VMS - starting all Docker services
echo ================================================================
echo.

if not exist ".env" goto noenv

REM ---- 1. Docker Desktop must be running ----
docker info >nul 2>&1
if not errorlevel 1 goto dockerok
echo Docker is not running - starting Docker Desktop...
if not exist "%ProgramFiles%\Docker\Docker\Docker Desktop.exe" goto nodocker
start "" "%ProgramFiles%\Docker\Docker\Docker Desktop.exe"
set /a tries=0
:waitdocker
timeout /t 5 /nobreak >nul
docker info >nul 2>&1
if not errorlevel 1 goto dockerready
set /a tries+=1
if %tries% GEQ 60 goto dockertimeout
goto waitdocker
:dockerready
echo Docker is ready.
echo.
:dockerok

REM ---- 2. Optional heavy services (run "start_services.bat all" to start everything without questions) ----
set PROFILES=
if /i "%~1"=="all" goto allprofiles
echo Core services always start: database, Kafka, API, scheduler, video server, gateway, GPU agent.
echo.
set /p ANALYTICS=Also start ANALYTICS - MTMC, Milvus, Elasticsearch, Kibana - needs about 6 GB RAM? [y/N]:
if /i "!ANALYTICS!"=="y" set PROFILES=!PROFILES! --profile analytics
set /p MONITORING=Also start MONITORING - Prometheus and Grafana? [y/N]:
if /i "!MONITORING!"=="y" set PROFILES=!PROFILES! --profile monitoring
goto startnow
:allprofiles
set PROFILES=--profile analytics --profile monitoring

REM ---- 3. Start everything ----
:startnow
echo.
echo Starting services - the first start can take a few minutes...
docker compose %PROFILES% up -d
if errorlevel 1 goto composefail

echo.
echo ================================================================
echo   Service status
echo ================================================================
docker compose %PROFILES% ps --format "table {{.Service}}\t{{.Status}}\t{{.Ports}}"

echo.
echo ================================================================
echo   Addresses
echo ================================================================
echo   API / gateway : https://localhost:8443        API docs at /docs
echo   Live viewer   : run  run_all.bat  to start cameras and open http://localhost:8080
echo   Kibana        : http://localhost:5601         only if analytics was started
echo   Grafana       : http://localhost:3000         only if monitoring was started
echo   Prometheus    : http://localhost:9090         only if monitoring was started
echo.
echo   To stop everything: stop_services.bat
echo.
pause
exit /b 0

:noenv
echo [ERROR] .env file not found. Copy .env.example to .env and fill every CHANGE_ME first.
pause
exit /b 1

:nodocker
echo [ERROR] Docker Desktop not found. Install it from https://www.docker.com and run this again.
pause
exit /b 1

:dockertimeout
echo [ERROR] Docker did not start within 5 minutes. Start Docker Desktop manually and run this again.
pause
exit /b 1

:composefail
echo.
echo [ERROR] docker compose failed. See the message above, or run: docker compose logs
pause
exit /b 1
