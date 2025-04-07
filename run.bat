@echo off
setlocal

REM Change to script directory
cd /d "%~dp0"

echo === VOICE TRANSCRIPTION SYSTEM ===
echo.

REM Activate virtual environment
call myenv\Scripts\activate.bat

REM Check if .env exists
if not exist ".env" (
    echo ERROR: .env file is missing!
    echo Please run setup.bat first.
    pause
    exit /b 1
)

REM Show menu
:menu
cls
echo === MAIN MENU ===
echo 1. Start transcription system
echo 2. Run audio diagnostics
echo 3. Show audio devices
echo 4. Configure settings
echo 5. Exit
echo.

set /p val="Choose an option (1-5): "

if "%val%"=="1" (
    echo Starting transcription system...
    python transcribe_live.py
    pause
    goto menu
)

if "%val%"=="2" (
    python test_audio_setup.py
    pause
    goto menu
)

if "%val%"=="3" (
    python hitta_enhet.py
    pause
    goto menu
)

if "%val%"=="4" (
    notepad .env
    goto menu
)

if "%val%"=="5" (
    echo Exiting...
    exit /b 0
)

echo.
echo Invalid option, please try again.
timeout /t 2 >nul
goto menu
