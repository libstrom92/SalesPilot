@echo off
setlocal

REM Change to script directory
cd /d "%~dp0"

echo === VOICE TRANSCRIPTION SYSTEM SETUP ===
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Python not found! Please install Python 3.8 or later.
    pause
    exit /b 1
)

REM Create virtual environment if it doesn't exist
if not exist "myenv\Scripts\activate.bat" (
    echo Creating virtual environment...
    python -m venv myenv
    call myenv\Scripts\activate.bat
    echo Installing dependencies...
    pip install -r requirements.txt
) else (
    call myenv\Scripts\activate.bat
)

REM Check for .env file
if not exist ".env" (
    echo Creating .env file from template...
    copy .env.template .env
    echo Please configure your .env file with your Hugging Face token.
    notepad .env
)

REM Run environment tests
python test_environment.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Environment tests failed! Please fix the issues before proceeding.
    echo Run 'python hitta_enhet.py' to find your audio device ID.
    pause
    exit /b 1
)

echo.
echo Setup completed successfully!
echo You can now run 'run.bat' to start the transcription system.
pause
