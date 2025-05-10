@echo off
echo Starting voice transcription server...
call myenv\Scripts\activate.bat
python start_server.py start
REM Visa eventuella fel och håll fönstret öppet
pause