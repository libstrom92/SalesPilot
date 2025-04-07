import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def validate_tokens():
    """Validate that the required tokens are set."""
    huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
    hf_auth_token = os.getenv("HF_AUTH_TOKEN")

    if huggingface_token:
        print("Hugging Face Token is correctly set.")
    else:
        print("Hugging Face Token is not set correctly.")

    if hf_auth_token:
        print("HF Auth Token is correctly set.")
    else:
        print("HF Auth Token is not set correctly.")

# Call the function only if debugging
if __name__ == "__main__":
    validate_tokens()