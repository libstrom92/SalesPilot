import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Get Hugging Face token from the correct variable
HF_AUTH_TOKEN = os.getenv("HF_AUTH_TOKEN", "").strip()

def load_config():
    """Fetch configuration and validate important environment variables."""
    
    required_vars = {
        "auth_token": HF_AUTH_TOKEN,
        "sample_rate": 16000,
        "chunk_size": 16000 * 10,
        "websocket_host": "0.0.0.0",
        "websocket_port": 9091
    }

    if not required_vars["auth_token"] or not required_vars["auth_token"].startswith("hf_"):
        raise ValueError("HF_AUTH_TOKEN is missing or invalid in the .env file")

    return required_vars
