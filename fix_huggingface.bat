@echo off
echo === Installing huggingface_hub package ===
echo.

REM Activate the virtual environment
call myenv\Scripts\activate.bat

REM Install the missing package
echo Installing huggingface_hub...
pip install huggingface_hub

echo.
echo Installation complete!
echo You can now run verify_token.py

pause
