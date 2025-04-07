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

REM If we get here, there was an error
echo.
echo Om det uppstod fel, kontrollera att:
echo 1. Du har konfigurerat .env med korrekt HF_AUTH_TOKEN
echo 2. Du har valt rätt ljudenhet (kör hitta_enhet.py för att se tillgängliga enheter)
echo 3. Din mikrofon är ansluten och fungerar
echo.
pause