import os
from dotenv import load_dotenv
from huggingface_hub import HfApi
import sys

def check_model_access():
    """Check if the user has accepted the terms for required models"""
    print("=== CHECKING MODEL ACCESS ===")
    
    # Load environment variables
    load_dotenv()
    
    # Get token
    token = os.getenv("HF_AUTH_TOKEN", "").strip()
    
    if not token:
        print("ERROR: No HF_AUTH_TOKEN found in .env file")
        return False
    
    # Models to check
    models = [
        "pyannote/speaker-diarization",
        "pyannote/segmentation"
    ]
    
    # Initialize API
    api = HfApi(token=token)
    
    # Check each model
    all_access = True
    for model in models:
        try:
            print(f"Checking access to {model}...")
            api.model_info(model)
            print(f"✓ Access granted to {model}")
        except Exception as e:
            print(f"✗ No access to {model}: {e}")
            print(f"  Please visit https://huggingface.co/{model} and accept the terms")
            all_access = False
    
    if all_access:
        print("\nSUCCESS: You have access to all required models!")
    else:
        print("\nWARNING: You need to accept the terms for some models.")
        print("Please visit the model pages listed above and click 'Accept' on the license terms.")
        print("After accepting the terms, run this script again to verify access.")
    
    return all_access

if __name__ == "__main__":
    success = check_model_access()
    sys.exit(0 if success else 1)
