@echo off
setlocal

REM === Activate the correct environment (adjust path as needed) ===
call .\myenv\Scripts\activate

echo === Fixing dependencies ===
echo.

REM === Step 1: Fix the NumPy version ===
echo Step 1: Fixing NumPy version...
pip uninstall numpy -y
pip install "numpy<2"

REM === Step 2: Clean and reinstall ONNX Runtime ===
echo.
echo Step 2: Reinstalling ONNX Runtime...
pip uninstall onnxruntime -y
pip install onnxruntime

REM === Step 3: Ensure compatible pyannote.audio installation ===
echo.
echo Step 3: Reinstalling pyannote.audio...
pip uninstall pyannote.audio -y
pip install pyannote.audio

echo.
echo === Dependencies fixed! ===
echo You can now run the system with 'python transcribe_live.py'
echo or test with 'python test_voice_recording.py'
echo.

pause
