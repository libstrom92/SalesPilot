@echo off
setlocal EnableDelayedExpansion

REM Change to script directory
cd /d "%~dp0"

echo === LJUDDIAGNOSTIK ===
echo.

REM Activate virtual environment
if not exist "myenv\Scripts\activate.bat" (
    echo Skapar virtual environment...
    python -m venv myenv
    call myenv\Scripts\activate.bat
    echo Installerar beroenden...
    pip install -r requirements.txt
) else (
    call myenv\Scripts\activate.bat
)

:menu
cls
echo === LJUDTESTMENY ===
echo 1. Visa tillgängliga ljudenheter
echo 2. Visa realtids ljudnivåer
echo 3. Snabbtest av ljudingång
echo 4. Kör fullständigt systemtest
echo 5. Avsluta
echo.

set /p val="Välj ett alternativ (1-5): "

if "%val%"=="1" (
    python hitta_enhet.py
    pause
    goto menu
)

if "%val%"=="2" (
    python volym_visualiserare.py
    goto menu
)

if "%val%"=="3" (
    python test_audio_input.py
    pause
    goto menu
)

if "%val%"=="4" (
    python test_audio_setup.py
    pause
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