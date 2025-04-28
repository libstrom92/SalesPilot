@echo off
REM Startar Python-backend (main_server.py) i nytt fönster
start "Backend" cmd /k "call activate_env.bat && python main_server.py"

REM Startar frontend (React/Vite) i nytt fönster
cd my-transcribe-app
start "Frontend" cmd /k "npm run dev"
cd ..

echo Allt startat! Backend och frontend körs i egna fönster.
pause
