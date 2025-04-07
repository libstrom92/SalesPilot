import os
import logging
from dotenv import load_dotenv
from huggingface_hub import HfApi
import sounddevice as sd
import torch
import numpy as np
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_env_variables():
    """Test environment variables are properly set"""
    load_dotenv()
    
    # Add check for virtual environment
    if not os.path.exists("myenv"):
        logger.warning("Virtual environment 'myenv' not found")
        return False
        
    required_vars = {
        "HF_AUTH_TOKEN": os.getenv("HF_AUTH_TOKEN"),
        "AUDIO_DEVICE_ID": os.getenv("AUDIO_DEVICE_ID"),
        "SAMPLE_RATE": os.getenv("SAMPLE_RATE"),
        "WHISPER_MODEL": os.getenv("WHISPER_MODEL"),
        "COMPUTE_TYPE": os.getenv("COMPUTE_TYPE"),
        "WHISPER_LANGUAGE": os.getenv("WHISPER_LANGUAGE", "sv")
    }
    
    missing = [k for k, v in required_vars.items() if not v]
    if missing:
        logger.error(f"❌ Missing environment variables: {', '.join(missing)}")
        return False
    
    logger.info("✅ All required environment variables are set")
    return True

def test_huggingface_token():
    """Test Hugging Face token validity"""
    token = os.getenv("HF_AUTH_TOKEN")
    try:
        api = HfApi(token=token)
        user_info = api.whoami()
        
        # Also verify access to required models
        models_to_check = [
            "openai/whisper-medium",
            "pyannote/speaker-diarization-3.0"
        ]
        for model in models_to_check:
            try:
                api.model_info(model)
                logger.info(f"✅ Access verified for model: {model}")
            except Exception as e:
                logger.error(f"❌ Cannot access model {model}: {e}")
                return False
                
        logger.info(f"✅ Hugging Face token valid for user: {user_info['name']}")
        return True
    except Exception as e:
        logger.error(f"❌ Hugging Face token validation failed: {e}")
        return False

def test_audio_device():
    """Test audio device configuration"""
    try:
        device_id = int(os.getenv("AUDIO_DEVICE_ID", "2"))
        
        # Test if device exists
        devices = sd.query_devices()
        if device_id >= len(devices):
            logger.error(f"❌ Device ID {device_id} does not exist")
            return False
            
        device_info = sd.query_devices(device_id, 'input')
        
        # Verify device capabilities
        sample_rate = int(os.getenv("SAMPLE_RATE", "16000"))
        if device_info['max_input_channels'] < 1:
            logger.error("❌ Selected device has no input channels")
            return False
            
        # Test device can be opened
        try:
            with sd.InputStream(device=device_id, channels=1, samplerate=sample_rate):
                logger.info(f"✅ Audio device working: {device_info['name']}")
        except sd.PortAudioError as e:
            logger.error(f"❌ Could not open audio device: {e}")
            return False
            
        return True
    except Exception as e:
        logger.error(f"❌ Audio device test failed: {e}")
        return False

def test_gpu_availability():
    """Test GPU availability for PyTorch"""
    if torch.cuda.is_available():
        # Test CUDA memory allocation
        try:
            test_tensor = torch.zeros(1).cuda()
            del test_tensor
            logger.info(f"✅ GPU working: {torch.cuda.get_device_name(0)}")
            return True
        except Exception as e:
            logger.error(f"❌ GPU found but not working: {e}")
            return False
    else:
        logger.warning("⚠️ No GPU found, will use CPU")
        return True

def test_directory_structure():
    """Test required directories exist and are writable"""
    required_dirs = {
        "conversation_logs": "Store conversation history",
        "myenv": "Python virtual environment"
    }
    
    success = True
    for dir_name, purpose in required_dirs.items():
        dir_path = Path(dir_name)
        if not dir_path.exists():
            try:
                dir_path.mkdir(exist_ok=True)
                logger.info(f"✅ Created {dir_name} directory for {purpose}")
            except Exception as e:
                logger.error(f"❌ Could not create {dir_name} directory: {e}")
                success = False
                continue
                
        # Test write permissions
        try:
            test_file = dir_path / ".test_write"
            test_file.touch()
            test_file.unlink()
            logger.info(f"✅ {dir_name} directory is writable")
        except Exception as e:
            logger.error(f"❌ {dir_name} directory is not writable: {e}")
            success = False
            
    return success

def run_all_tests():
    """Run all environment tests"""
    tests = [
        ("Directory Structure", test_directory_structure),
        ("Environment Variables", test_env_variables),
        ("Hugging Face Token", test_huggingface_token),
        ("GPU Availability", test_gpu_availability),
        ("Audio Device", test_audio_device)
    ]
    
    results = []
    print("\n=== Environment Test Suite ===")
    
    for name, test_func in tests:
        print(f"\n▶️ Testing {name}...")
        result = test_func()
        results.append(result)
        
    success_rate = sum(results) / len(results) * 100
    
    print("\n=== Test Summary ===")
    print(f"Success Rate: {success_rate:.1f}%")
    for (name, _), result in zip(tests, results):
        status = "✅ Passed" if result else "❌ Failed"
        print(f"{status} - {name}")
    
    return all(results)

if __name__ == "__main__":
    success = run_all_tests()
    if success:
        print("\n✅ All systems operational! You can proceed with using the system.")
    else:
        print("\n❌ Some checks failed. Please address the issues before proceeding.")
        print("   Run 'python hitta_enhet.py' to troubleshoot audio device issues.")
        print("   Check the README.md for setup instructions.")
