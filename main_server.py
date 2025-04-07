import asyncio
from websockets.legacy.server import WebSocketServerProtocol  # Updated import for legacy protocol
from websockets.legacy.server import serve  # Use legacy server for compatibility
from streaming_processor import StreamingProcessor  # Import the new streaming processor
import logging
import json
import numpy as np
import sounddevice as sd
import numpy as np
import threading
import queue
import time
import torch  # Import torch for tensor conversion
import os
import concurrent.futures
from typing import Set, Optional, Dict, List
from pathlib import Path

from audio_processor import AudioProcessor, AudioProcessingError
from config import load_config
from logging_config import setup_logging
from audio_monitor import AudioLevelMonitor, create_level_meter

# === CONFIGURATION ===
logger = setup_logging("VoicemeeterServer", level=logging.DEBUG)

config = load_config()  # Load configuration from .env
PORT = int(os.getenv("WEBSOCKET_PORT", "9091"))
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "16000"))

try:
    DEVICE = int(os.getenv("AUDIO_DEVICE_ID", "2").strip())  # Strip spaces/comments and convert to int
except ValueError:
    logger.warning("Invalid value for AUDIO_DEVICE_ID. Using default value 2.")
    DEVICE = 2  # Default to 2 if invalid

logger.info(f"Using audio device ID: {DEVICE}")
CHANNELS = int(os.getenv("AUDIO_CHANNELS", "1"))
CHUNK_SIZE = 512  # Reduced from 1024 for more frequent updates
BUFFER_DURATION = 2  # Reduced from 5 to 2 seconds for lower latency

# Get transcription mode from environment
TRANSCRIPTION_MODE = os.getenv("TRANSCRIPTION_MODE", "balanced")

# Message queue for WebSocket communication
message_queue = queue.Queue()

# === FRAMEWORK MANAGEMENT ===
class FrameworkManager:
    def __init__(self, config_dir: str = "frameworks"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        self.active_framework = None
        self.frameworks: Dict[str, dict] = {}
        self.load_frameworks()

    def load_frameworks(self):
        """Load all saved frameworks"""
        for file in self.config_dir.glob("*.json"):
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    framework = json.load(f)
                    self.frameworks[framework['id']] = framework
            except Exception as e:
                logger.error(f"Failed to load framework {file}: {e}")

    def save_framework(self, framework: dict) -> bool:
        """Save a framework configuration"""
        try:
            framework_id = framework.get('id')
            if not framework_id:
                framework_id = str(int(time.time()))
                framework['id'] = framework_id

            filepath = self.config_dir / f"{framework_id}.json"
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(framework, f, indent=2, ensure_ascii=False)
            
            self.frameworks[framework_id] = framework
            return True
        except Exception as e:
            logger.error(f"Failed to save framework: {e}")
            return False

    def get_framework(self, framework_id: str) -> Optional[dict]:
        """Get a specific framework by ID"""
        return self.frameworks.get(framework_id)

    def list_frameworks(self) -> List[dict]:
        """List all available frameworks"""
        return list(self.frameworks.values())

    def activate_framework(self, framework_id: str) -> bool:
        """Set a framework as active"""
        if framework := self.frameworks.get(framework_id):
            self.active_framework = framework
            return True
        return False

    def get_active_framework(self) -> Optional[dict]:
        """Get the currently active framework"""
        return self.active_framework

# === AUDIO PROCESSING ===
class ProcessingThread(threading.Thread):
    def __init__(self, websocket=None):
        super().__init__()
        self.daemon = True
        self.stream = None
        self.running = False
        self.audio_queue = queue.Queue()
        self.processor = StreamingProcessor(config)
        self.websocket = websocket
        self.audio_frames_received = 0
        self.last_log_time = time.time()

    def process_audio(self, audio_data):
        """Process audio data through the streaming processor"""
        if self.processor:
            return self.processor.process_streaming(audio_data)
        return None

    def start_recording(self):
        """Start recording from the audio device."""
        self.running = True
        logger.info(f"Starting audio recording with device={DEVICE}, rate={SAMPLE_RATE}, channels={CHANNELS}")
        try:
            self.stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                device=DEVICE,
                callback=self.audio_callback
            )
            self.stream.start()
            logger.info("Audio recording started successfully.")
        except Exception as e:
            logger.error(f"Failed to start audio recording: {e}")
            self.running = False
            # Informera klienten om felet
            if self.websocket:
                message_queue.put({
                    "websocket": self.websocket,
                    "message": {
                        "error": f"Kunde inte starta ljudinspelning: {e}",
                        "timestamp": time.time()
                    }
                })

    def stop_recording(self):
        """Stop the audio recording."""
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
                logger.info(f"Audio recording stopped. Processed {self.audio_frames_received} frames.")
            except Exception as e:
                logger.error(f"Error stopping audio stream: {e}")
        self.running = False

    def audio_callback(self, indata, frames, time_info, status):
        """Callback to handle incoming audio data."""
        current_time = time.time()
        
        # Logga varje sekund
        if current_time - self.last_log_time >= 1.0:
            self.last_log_time = current_time
            # Lägg till kontroll av ljudnivå för att avgöra om ljud tas upp
            max_audio_level = np.abs(indata).max()
            logger.debug(f"Audio callback active. Queue size: {self.audio_queue.qsize()}, Frames: {self.audio_frames_received}, Level: {max_audio_level:.4f}")
            
        if status:
            logger.warning(f"Audio stream status: {status}")
            
        if self.running:
            # Kontrollera om ljudet är tyst
            if np.abs(indata).max() < 0.001:  # Mycket låg ljudnivå
                self.quiet_frames = self.quiet_frames + 1 if hasattr(self, 'quiet_frames') else 1
                if self.quiet_frames % 100 == 0:  # Logga var 100:e tyst frame
                    logger.warning(f"Receiving silent audio frames ({self.quiet_frames} in a row). Check your microphone settings.")
            else:
                if hasattr(self, 'quiet_frames') and self.quiet_frames > 100:
                    logger.info(f"Audio detected after {self.quiet_frames} silent frames.")
                self.quiet_frames = 0
            
            self.audio_queue.put(indata.copy())
            self.audio_frames_received += 1

    def run(self):
        """Process audio data from the queue."""
        logger.info("Audio processing thread started.")
        processing_count = 0
        last_process_time = time.time()
        last_test_message = time.time()
        
        while self.running:
            try:
                # Skicka ett periodiskt testmeddelande varje 5 sekunder för att verifiera WebSocket-flödet
                current_time = time.time()
                if current_time - last_test_message >= 5.0 and self.websocket:
                    message_queue.put({
                        "websocket": self.websocket,
                        "message": {
                            "type": "transcription", 
                            "text": f"Test meddelande {int(current_time)}. Ljud processas...",
                            "timestamp": current_time
                        }
                    })
                    last_test_message = current_time
                    logger.debug("Sent periodic test message")
                
                # Använd timeout för att inte blockera för länge
                try:
                    audio_data = self.audio_queue.get(timeout=1)
                except queue.Empty:
                    continue
                
                # Logga var 10:e sekund
                if current_time - last_process_time >= 10.0:
                    logger.debug(f"Still processing audio. Processed {processing_count} chunks so far.")
                    last_process_time = current_time
                
                # Process the audio data here (e.g., send to transcription processor)
                processing_start = time.time()
                text = self.process_audio(audio_data)
                processing_time = time.time() - processing_start
                processing_count += 1
                
                if processing_time > 0.5:  # Logga om det tar mer än 0.5 sekunder
                    logger.warning(f"Long processing time: {processing_time:.2f}s for audio chunk")
                
                if text and text.strip():
                    logger.info(f"Transcription: {text}")
                    
                    # Queue message for sending via WebSocket
                    if self.websocket:
                        message_queue.put({
                            "websocket": self.websocket,
                            "message": {
                                "type": "transcription", 
                                "text": text,
                                "timestamp": time.time()
                            }
                        })
                        logger.debug(f"Queued transcription message: {text}")
            except Exception as e:
                logger.error(f"Error processing audio data: {e}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
        
        logger.info("Audio processing thread stopped.")

# === WEBSOCKET HANDLER ===
active_connections: Set[WebSocketServerProtocol] = set()

# Process outgoing messages in the main asyncio loop
async def process_messages():
    """Process outgoing WebSocket messages from the queue"""
    logger.info("Started message processing task")
    while True:
        await asyncio.sleep(0.05)  # Snabbare kontroll för att minska latensen
        try:
            # Non-blocking check for messages
            try:
                item = message_queue.get_nowait()
                websocket = item["websocket"]
                message = item["message"]
                
                # Kontrollera om websocket fortfarande är öppen
                if not websocket.open:
                    logger.warning("WebSocket is closed, can't send message")
                    message_queue.task_done()
                    continue
                
                try:
                    # Konvertera meddelandet till JSON
                    message_json = json.dumps(message)
                    logger.debug(f"Sending message to client: {message_json[:100]}...")
                    
                    # Skicka meddelandet och vänta tills det är klart
                    await websocket.send(message_json)
                    logger.debug("Message sent successfully")
                except Exception as send_err:
                    logger.error(f"Error sending message: {send_err}")
                
                # Markera uppgiften som klar
                message_queue.task_done()
                
            except queue.Empty:
                pass  # No messages in queue
        except Exception as e:
            logger.error(f"Error in message processor: {e}")
            import traceback
            logger.error(f"Message processor error trace: {traceback.format_exc()}")
            await asyncio.sleep(1)  # Wait a bit longer on error

async def handle_websocket(websocket: WebSocketServerProtocol):
    """Handle WebSocket connection and commands for audio processing"""
    try:
        active_connections.add(websocket)
        recording_thread = ProcessingThread(websocket)

        async for message in websocket:
            data = json.loads(message)

            if data.get('command') == 'start':
                logger.info("Received 'start' command from client.")
                if not recording_thread.running:
                    recording_thread.start()
                    recording_thread.start_recording()
                    
                    # Skicka initial statusinformation
                    await websocket.send(json.dumps({
                        "status": "recording_started",
                        "timestamp": time.time()}))
                    
                    # Skicka ett testmeddelande för att verifiera WebSocket-kommunikation
                    logger.info("Sending test transcription message to client")
                    await websocket.send(json.dumps({
                        "type": "transcription", 
                        "text": "Server ansluten. Testar WebSocket-kommunikation.",
                        "timestamp": time.time()
                    }))

            elif data.get('command') == 'stop':
                logger.info("Received 'stop' command from client.")
                if recording_thread.running:
                    recording_thread.stop_recording()
                    recording_thread.running = False
                    await websocket.send(json.dumps({"status": "recording_stopped"}))

    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        active_connections.remove(websocket)

async def start_websocket_server():
    """Start the WebSocket server for handling client connections"""
    try:
        # Start message processing task
        asyncio.create_task(process_messages())
        
        # Start WebSocket server
        server = await serve(handle_websocket, "localhost", PORT)
        logger.info(f"WebSocket server started on port {PORT}")
        await server.wait_closed()
    except OSError as e:
        logger.error(f"Failed to start WebSocket server: {e}")
        if e.errno == 10048:  # Address already in use
            logger.error(f"Port {PORT} is already in use. Is another instance already running?")
        raise

def start_server(audio_processor=None, config=None):
    """Start the WebSocket server with the given audio processor and configuration.
    This function is imported by transcribe_live.py to start the server.
    
    Args:
        audio_processor: Optional AudioProcessor instance
        config: Optional configuration dictionary
    """
    # Use the provided config or the default one
    if config:
        global PORT, SAMPLE_RATE, DEVICE, CHANNELS, TRANSCRIPTION_MODE
        PORT = int(config.get("websocket_port", PORT))
        SAMPLE_RATE = int(config.get("sample_rate", SAMPLE_RATE))
        DEVICE = int(config.get("audio_device_id", DEVICE))
        CHANNELS = int(config.get("audio_channels", CHANNELS))
        TRANSCRIPTION_MODE = config.get("transcription_mode", TRANSCRIPTION_MODE)
    
    logger.info(f"Starting WebSocket server on port {PORT}")
    logger.info(f"Audio settings: device={DEVICE}, rate={SAMPLE_RATE}, channels={CHANNELS}")
    
    try:
        # Start the WebSocket server using asyncio
        asyncio.run(start_websocket_server())
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        raise

if __name__ == "__main__":
    # If this script is run directly, start the server
    start_server()
