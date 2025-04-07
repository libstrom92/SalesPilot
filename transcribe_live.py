import os
import sys
import socket
import logging
import json
import time
from config import load_config
from audio_processor import AudioProcessor
from logging_config import setup_logging

try:
    from main_server import start_server
except ImportError:
    raise ImportError("The 'main_server.py' module could not be found. Ensure it exists in the same directory.")

logger = setup_logging("TranscribeLive")


def is_port_in_use(port):
    """Check if the specified port is already in use"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0


def check_environment():
    """Perform basic environment checks before starting"""
    if not os.path.exists(".env"):
        logger.error("No .env file found. Please run setup.bat first.")
        print("ERROR: No .env file found. Please run setup.bat first.")
        return False

    config = load_config()
    if not config.get("auth_token"):
        logger.error("No Hugging Face token found in .env file.")
        print("ERROR: No Hugging Face token found in .env file.")
        print("Please run 'python token_manager.py set' to set your token.")
        return False

    return True


async def send_test_message(websocket):
    """Send a test message to the client upon connection"""
    await websocket.send(json.dumps({
        "type": "transcription",
        "text": "Detta är ett test. Om du ser detta fungerar WebSocket-kommunikation.",
        "timestamp": time.time()
    }))


def main():
    """Main entry point for the transcription system"""
    logger.info("Starting transcription system")

    if not check_environment():
        return 1

    config = load_config()
    audio_processor = AudioProcessor(config)

    if is_port_in_use(9091):
        logger.warning("WebSocket-servern är redan igång. Ingen ny server startas.")
        print("WebSocket-servern är redan igång. Ingen ny server startas.")
        return 0

    start_server(audio_processor, config)
    return 0


if __name__ == "__main__":
    sys.exit(main())