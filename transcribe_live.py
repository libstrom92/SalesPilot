import os
import sys
import socket
import json
import time
import asyncio
from config import load_config
from audio_processor import AudioProcessor
from logging_config import setup_logging
from main_server import start_server

logger = setup_logging("TranscribeLive")

def is_port_in_use(port: int) -> bool:
    """Check if the specified port is already in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def check_environment() -> bool:
    """Perform environment checks before starting the system."""
    if not os.path.exists(".env"):
        logger.error("No .env file found. Please run setup.bat first.")
        print("ERROR: No .env file found. Please run setup.bat first.")
        return False

    config = load_config()
    if not config.get("auth_token"):
        logger.error("No Hugging Face token found in .env.")
        print("ERROR: No Hugging Face token found in .env.")
        print("Run 'python token_manager.py set' to set your token.")
        return False

    return True

async def main_async():
    logger.info("Starting transcription system...")

    if not check_environment():
        return 1

    config = load_config()
    audio_processor = AudioProcessor(config)

    port = config.get("websocket_port", 9091)
    if is_port_in_use(port):
        logger.warning(f"WebSocket server is already running on port {port}. Not starting a new server.")
        print(f"WebSocket server is already running on port {port}. Not starting a new server.")
        return 0

    # Start WebSocket server
    await start_server(audio_processor, config)

    return 0

def main():
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If an event loop is already running, create a new task
            return loop.create_task(main_async())
        else:
            return loop.run_until_complete(main_async())
    except RuntimeError:
        # If no event loop exists, create a new one
        return asyncio.run(main_async())

if __name__ == "__main__":
    sys.exit(main())
