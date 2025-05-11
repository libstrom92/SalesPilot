@echo on
REM Enable detailed logging for debugging

REM Load configuration from config file
set CONFIG_FILE=config\default_settings.json
if exist %CONFIG_FILE% (
    for /f "tokens=* delims=" %%A in ('type %CONFIG_FILE% ^| findstr "audio_test_enabled"') do set "AUDIO_TEST_ENABLED=%%A"
    set AUDIO_TEST_ENABLED=%AUDIO_TEST_ENABLED:audio_test_enabled=%%%
    set AUDIO_TEST_ENABLED=%AUDIO_TEST_ENABLED:,%%=%
) else (
    echo [ERROR] Configuration file not found: %CONFIG_FILE%
    pause
    exit /b 1
)

REM Check if audio test is enabled in the configuration
if /i "%AUDIO_TEST_ENABLED%"=="true" (
    echo [INFO] Ljudtest är aktiverat enligt konfigurationen. Hoppar över test.
) else (
    echo [INFO] Ljudtest är inaktiverat enligt konfigurationen. Hoppar över test.
)

REM Starta Python-miljö och backend-server i nytt fönster
start "Backend" cmd /k "cd /d %~dp0 && call myenv\Scripts\activate.bat && python main_server.py"
if errorlevel 1 (
    echo [ERROR] Failed to start the backend server. Check if myenv\Scripts\activate.bat and main_server.py exist.
    pause
    exit /b 1
)

REM Vänta tills websocket_port.txt finns (max 20 sekunder)
setlocal enabledelayedexpansion
set COUNT=0
:waitloop
if exist websocket_port.txt (
    echo [INFO] websocket_port.txt hittad.
) else (
    set /a COUNT+=1
    echo [INFO] Waiting for websocket_port.txt... Attempt !COUNT!
    if !COUNT! geq 20 (
        echo [ERROR] Timeout: websocket_port.txt hittades inte.
        goto aftercopy
    )
    timeout /t 1 >nul
    goto waitloop
)

REM Kopiera websocket_port.txt till frontend/public
if exist my-transcribe-app\public (
    copy /Y websocket_port.txt my-transcribe-app\public\websocket_port.txt >nul
    if errorlevel 1 (
        echo [ERROR] Failed to copy websocket_port.txt to my-transcribe-app\public.
        pause
        exit /b 1
    )
    echo [INFO] Kopierade websocket_port.txt till my-transcribe-app\public\
) else (
    echo [INFO] Skapar my-transcribe-app\public\
    mkdir my-transcribe-app\public
    if errorlevel 1 (
        echo [ERROR] Failed to create directory my-transcribe-app\public.
        pause
        exit /b 1
    )
    copy /Y websocket_port.txt my-transcribe-app\public\websocket_port.txt >nul
    if errorlevel 1 (
        echo [ERROR] Failed to copy websocket_port.txt to my-transcribe-app\public.
        pause
        exit /b 1
    )
)
:aftercopy

REM Starta frontend (Vite dev server) i nytt fönster
start "Frontend" cmd /k "cd /d %~dp0my-transcribe-app && npm run dev"
if errorlevel 1 (
    echo [ERROR] Failed to start the frontend server. Check if npm and my-transcribe-app exist.
    pause
    exit /b 1
)

echo [INFO] Allt startat! Backend och frontend körs i egna fönster.
pause
