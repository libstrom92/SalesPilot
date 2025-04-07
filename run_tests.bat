@echo off
setlocal

REM Change to script directory
cd /d "%~dp0"

echo === Running Test Suite ===
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

REM Run environment tests first
echo Running environment tests...
python test_environment.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Environment tests failed! Please fix the issues before running unit tests.
    pause
    exit /b 1
)

REM Run unit tests
echo.
echo Running unit tests...
python -m unittest test_audio_processor.py -v

echo.
echo All tests completed.
pause