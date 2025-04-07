import os
import sys
import socket
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


def is_port_in_use(port: int) -> bool:
    """Kontrollerar om den angivna porten redan används."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0


def check_environment() -> bool:
    """Utför miljökontroller innan systemet startas."""
    if not os.path.exists(".env"):
        logger.error("Ingen .env-fil hittades. Kör setup.bat först.")
        print("ERROR: Ingen .env-fil hittades. Kör setup.bat först.")
        return False

    config = load_config()
    if not config.get("auth_token"):
        logger.error("Ingen Hugging Face-token hittades i .env.")
        print("ERROR: Ingen Hugging Face-token hittades i .env.")
        print("Kör 'python token_manager.py set' för att sätta din token.")
        return False

    return True


async def send_test_message(websocket):
    """Skicka ett testmeddelande till klienten vid anslutning."""
    await websocket.send(json.dumps({
        "type": "transcription",
        "text": "Detta är ett test. Om du ser detta fungerar WebSocket-kommunikationen.",
        "timestamp": time.time()
    }))


def main() -> int:
    """Huvudfunktion för transkriptionssystemet."""
    logger.info("Startar transkriptionssystem...")

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
