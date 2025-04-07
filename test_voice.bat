@echo off
setlocal

REM Change to script directory
cd /d "%~dp0"

echo === VOICE RECORDING TEST ===
echo.

REM Activate virtual environment
if not exist "myenv\Scripts\activate.bat" (
    echo Creating virtual environment...
    python -m venv myenv
    call myenv\Scripts\activate.bat
    echo Installing dependencies...
    pip install -r requirements.txt
) else (
    call myenv\Scripts\activate.bat
)

REM Check if .env exists
if not exist ".env" (
    echo WARNING: .env file missing!
    echo Copying template from .env.template...
    copy .env.template .env
    echo Please configure your .env file with the correct settings.
    pause
    exit
)

REM Run the test
python test_voice_recording.py

echo.
pause
