@echo off
echo Stopping voice transcription server...
call myenv\Scripts\activate.bat
python start_server.py stop