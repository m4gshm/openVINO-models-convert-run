@echo off
setlocal EnableExtensions EnableDelayedExpansion

:: ============================================================
:: OpenVINO Model Server
:: ============================================================

set "SCRIPT_DIR=%~dp0"
set "OVMS_DIR=C:\alex\app\ovms"
set "MODEL_DIR=%SCRIPT_DIR%models\gpt-oss-20b-int4-sym-g64-r1"
set "LOG_DIR=%SCRIPT_DIR%logs"

::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:: Initialize OVMS environment
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
call "%OVMS_DIR%\setupvars.bat" >nul 2>&1

::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:: Check directories
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
if not exist "%OVMS_DIR%" (
    echo [ERROR] OVMS directory not found: %OVMS_DIR%
    pause
    exit /b 1
)
if not exist "%OVMS_DIR%\ovms.exe" (
    echo [ERROR] OVMS executable not found: %OVMS_DIR%\ovms.exe
    pause
    exit /b 1
)
if not exist "%MODEL_DIR%" (
    echo [ERROR] Model directory not found: %MODEL_DIR%
    pause
    exit /b 1
)
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%" >nul 2>&1

set "LOG_FILE=%LOG_DIR%\ovms_server.log"

::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:: Start OVMS (single-model mode — no config.json needed!)
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
echo [INFO] Starting OpenVINO Model Server...
echo [INFO] REST API will be available on port 8000
echo [INFO] Press Ctrl+C to stop the server
echo ============================================================
echo.

"%OVMS_DIR%\ovms.exe" ^
    --rest_port 8000 ^
    --model_name gpt-oss-20b ^
    --model_path "%MODEL_DIR%" ^
    --log_level DEBUG ^
    --log_path "%LOG_FILE%"

echo.
echo [INFO] Server stopped.
pause