@echo off
setlocal

REM Change to script directory
cd /d "%~dp0"

echo === Startar transkriberingssystem ===
echo.

REM Check if virtual environment exists
if not exist "myenv\Scripts\activate.bat" (
    echo Skapar virtual environment...
    python -m venv myenv
    call myenv\Scripts\activate.bat
    echo Installerar beroenden...
    pip install -r requirements.txt
) else (
    call myenv\Scripts\activate.bat
)

REM Check if .env exists
if not exist ".env" (
    echo VARNING: .env-fil saknas!
    echo Kopierar mall från .env.template...
    copy .env.template .env
    echo Vänligen konfigurera .env-filen med dina inställningar.
    pause
    exit
)

REM Start the system
echo Starting transcription system...
python transcribe_live.py
if errorlevel 1 goto :error

REM Om allt gick bra, avsluta
exit /b 0

:error
echo.
echo FEL: Transkriberingssystemet startade inte korrekt.
echo.
echo Kontrollera följande:
echo 1. .env-filen finns och innehåller giltig HF_AUTH_TOKEN
echo 2. Alla beroenden är installerade (kör fix_dependencies.bat vid behov)
echo 3. Ljudenhet/mikrofon är korrekt vald och ansluten (kör hitta_enhet.py)
echo 4. Eventuella felmeddelanden ovan
echo.
pause
exit /b 1