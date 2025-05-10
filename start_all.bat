@echo off
REM Starta Python-miljö och backend-server i nytt fönster
start "Backend" cmd /k "cd /d %~dp0 && call myenv\Scripts\activate.bat && python main_server.py"

REM Vänta tills websocket_port.txt finns (max 20 sekunder)
setlocal enabledelayedexpansion
set COUNT=0
:waitloop
if exist websocket_port.txt (
    echo websocket_port.txt hittad.
) else (
    set /a COUNT+=1
    if !COUNT! geq 20 (
        echo Timeout: websocket_port.txt hittades inte.
        goto aftercopy
    )
    timeout /t 1 >nul
    goto waitloop
)

REM Kopiera websocket_port.txt till frontend/public
if exist my-transcribe-app\public (
    copy /Y websocket_port.txt my-transcribe-app\public\websocket_port.txt >nul
    echo Kopierade websocket_port.txt till my-transcribe-app\public\
) else (
    echo Skapar my-transcribe-app\public\
    mkdir my-transcribe-app\public
    copy /Y websocket_port.txt my-transcribe-app\public\websocket_port.txt >nul
)
:aftercopy

REM Starta frontend (Vite dev server) i nytt fönster
start "Frontend" cmd /k "cd /d %~dp0my-transcribe-app && npm run dev"

echo Allt startat! Backend och frontend körs i egna fönster.
pause
