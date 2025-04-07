import os
import sys
import sounddevice as sd
import numpy as np
import torch
from dotenv import load_dotenv
from huggingface_hub import HfApi
from logging_config import setup_logging
from audio_monitor import AudioLevelMonitor, create_level_meter
import time

logger = setup_logging("Diagnostics")

def check_environment():
    """Check Python environment and dependencies"""
    print("\n=== CHECKING ENVIRONMENT ===")
    
    # Check Python version
    python_version = sys.version.split()[0]
    print(f"Python version: {python_version}")
    
    # Check virtual environment
    in_venv = sys.prefix != sys.base_prefix
    print(f"Virtual environment: {'Active' if in_venv else 'Not active'}")
    
    # Check key packages
    try:
        import faster_whisper
        print(f"faster-whisper: {faster_whisper.__version__}")
    except (ImportError, AttributeError):
        print("faster-whisper: Not installed")
    
    try:
        from pyannote.audio import Pipeline
        print("pyannote.audio: Installed")
    except ImportError:
        print("pyannote.audio: Not installed")
    
    # Check GPU
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU: Not available")
    
    return True

def check_token():
    """Check Hugging Face token"""
    print("\n=== CHECKING HUGGING FACE TOKEN ===")
    
    load_dotenv()
    token = os.getenv("HF_AUTH_TOKEN", "").strip()
    
    if not token:
        print("❌ No token found in .env file")
        return False
    
    if not token.startswith("hf_") or len(token) < 40:
        print("❌ Invalid token format")
        return False
    
    try:
        api = HfApi(token=token)
        user_info = api.whoami()
        print(f"✅ Token valid for user: {user_info['name']}")
        
        # Check model access
        models = [
            "pyannote/speaker-diarization-3.0",
            "openai/whisper-medium"
        ]
        
        all_access = True
        for model in models:
            try:
                api.model_info(model)
                print(f"✅ Access to {model}: Granted")
            except Exception as e:
                print(f"❌ Access to {model}: Denied ({e})")
                all_access = False
        
        return all_access
    except Exception as e:
        print(f"❌ Token verification failed: {e}")
        return False

def test_audio_device():
    """Test audio device"""
    print("\n=== TESTING AUDIO DEVICE ===")
    
    load_dotenv()
    try:
        device_id = int(os.getenv("AUDIO_DEVICE_ID", "2"))
        device_info = sd.query_devices(device_id, 'input')
        print(f"✅ Found audio device: {device_info['name']}")
        print(f"   Channels: {device_info['max_input_channels']}")
        print(f"   Sample rate: {device_info['default_samplerate']} Hz")
        
        # Test recording
        print("\nRecording 3 seconds of audio...")
        monitor = AudioLevelMonitor()
        
        def callback(indata, frames, time, status):
            if status:
                print(f"Status: {status}")
            monitor.add_level(indata)
            level = monitor.get_average_level()
            if level is not None:
                meter = create_level_meter(level)
                print(f"\rAudio level: {meter}", end="")
        
        with sd.InputStream(device=device_id,
                          channels=1,
                          callback=callback,
                          samplerate=16000):
            sd.sleep(3000)
        
        print("\n")
        level = monitor.get_average_level()
        if level is not None and level < 0.01:
            print("⚠️ Very low audio level detected")
        else:
            print("✅ Audio recording successful")
        
        return True
    except Exception as e:
        print(f"❌ Audio device test failed: {e}")
        return False

def main():
    """Run all diagnostic tests"""
    print("=== VOICE TRANSCRIPTION SYSTEM DIAGNOSTICS ===")
    
    results = {
        "Environment": check_environment(),
        "Token": check_token(),
        "Audio": test_audio_device()
    }
    
    print("\n=== DIAGNOSTIC SUMMARY ===")
    for test, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test}")
    
    if all(results.values()):
        print("\n✅ All diagnostics passed! The system is ready to use.")
    else:
        print("\n❌ Some diagnostics failed. Please fix the issues before using the system.")
        
        if not results["Environment"]:
            print("   - Run 'setup.bat' to set up the environment")
        if not results["Token"]:
            print("   - Run 'python token_manager.py set' to set your Hugging Face token")
        if not results["Audio"]:
            print("   - Run 'python hitta_enhet.py' to find your audio device ID")
            print("   - Update AUDIO_DEVICE_ID in .env file")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nDiagnostics interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"\nUnexpected error: {e}")
