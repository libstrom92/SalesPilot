@echo off
setlocal

REM Change to script directory
cd /d "%~dp0"

echo === STARTA TRANSKRIBERINGSSYSTEM ===
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Python hittades inte! Installera Python 3.8 eller senare.
    pause
    exit /b 1
)

REM Check and create virtual environment
if not exist "myenv\Scripts\activate.bat" (
    echo Skapar virtual environment...
    python -m venv myenv
    call myenv\Scripts\activate.bat
    echo Installerar beroenden...
    pip install -r requirements.txt
) else (
    call myenv\Scripts\activate.bat
)

REM Check for .env file
if not exist ".env" (
    echo .env-fil saknas! Skapar från mall...
    copy .env.template .env
    echo VIKTIGT: Konfigurera din .env-fil med rätt inställningar.
    notepad .env
    exit /b 1
)

REM Run environment tests
python test_environment.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Miljötest misslyckades! Kör 'test_audio.bat' för felsökning.
    pause
    exit /b 1
)

REM Show menu
:menu
cls
echo === HUVUDMENY ===
echo 1. Starta transkriberingssystem
echo 2. Kör ljuddiagnostik
echo 3. Visa ljudenheter
echo 4. Konfigurera .env
echo 5. Avsluta
echo.

set /p val="Välj ett alternativ (1-5): "

if "%val%"=="1" (
    python transcribe_live.py
    pause
    goto menu
)

if "%val%"=="2" (
    call test_audio.bat
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
    echo Avslutar...
    exit /b
)

echo.
echo Ogiltigt val, försök igen.
timeout /t 2 >nul
goto menu
