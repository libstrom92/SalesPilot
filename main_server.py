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
from queue import Queue
import sys
import socket

from audio_processor import AudioProcessor, AudioProcessingError
from config import load_config
from logging_config import setup_logging
from audio_monitor import AudioLevelMonitor, create_level_meter

# === CONFIGURATION ===
logger = setup_logging("VoicemeeterServer", level=logging.DEBUG)

config = load_config()  # Load configuration from .env
PORT = int(os.getenv("WEBSOCKET_PORT", "9091"))  # Använder ursprungliga standardporten 9091
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "16000"))

def find_voicemeeter_device():
    """Find Voicemeeter B1 output device"""
    for device in sd.query_devices():
        if isinstance(device, dict) and "Voicemeeter Out B1" in device.get('name', ''):
            logger.info(f"Found Voicemeeter B1: {device['name']} (ID: {device['index']})")
            return device['index']
    raise RuntimeError("Voicemeeter B1 not found. Please ensure Voicemeeter is running.")

try:
    DEVICE = find_voicemeeter_device()
except ValueError:
    logger.warning("Invalid value for AUDIO_DEVICE_ID. Using default value 2.")
    DEVICE = 2  # Default to 2 if invalid

logger.info(f"Using audio device ID: {DEVICE}")
CHANNELS = int(os.getenv("AUDIO_CHANNELS", "1"))
CHUNK_SIZE = 2048  # Reduced from 4160
BUFFER_DURATION = 2  # Reduced from 5 to 2 seconds for lower latency

# Get transcription mode from environment
TRANSCRIPTION_MODE = os.getenv("TRANSCRIPTION_MODE", "balanced")

# Message queue for WebSocket communication
message_queue = queue.Queue(maxsize=20)

def find_free_port(start_port=9091, max_tries=10):
    port = start_port
    for _ in range(max_tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('localhost', port))
                return port
            except OSError:
                port += 1
    raise RuntimeError("No free port found in range.")

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
        self.device_id = find_voicemeeter_device()
        self.running = False
        self.audio_queue = Queue(maxsize=500)  # Increased size to 500 for throttling
        self.processor = StreamingProcessor(config)
        self.websocket = websocket
        self.last_log_time = time.time()
        self.silence_threshold = 0.01  # Define silence_threshold as a class attribute
        
        # Start listening to Voicemeeter immediately
        self.start_recording()

    def start_recording(self):
        """Start recording from Voicemeeter B1"""
        self.running = True
        logger.info(f"Starting Voicemeeter B1 monitoring on device {self.device_id}")
        try:
            self.stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                device=self.device_id,
                callback=self.audio_callback  # OBS: nu vanlig funktion
            )
            self.stream.start()
            logger.info("Voicemeeter B1 monitoring started successfully.")
        except Exception as e:
            logger.error(f"Failed to start Voicemeeter B1 monitoring: {e}")
            self.running = False
            # Inform the client about the error
            if self.websocket:
                message_queue.put({
                    "websocket": self.websocket,
                    "message": {
                        "error": f"Could not start Voicemeeter B1 monitoring: {e}",
                        "timestamp": time.time()
                    }
                })

    def stop_recording(self):
        """Stop the Voicemeeter B1 monitoring."""
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
                logger.info("Voicemeeter B1 monitoring stopped.")
            except Exception as e:
                logger.error(f"Error stopping Voicemeeter B1 stream: {e}")
        self.running = False

    def audio_callback(self, indata, frames, time_info, status):
        """Callback to handle incoming audio data from Voicemeeter B1 (SYNKRON)."""
        current_time = time.time()
        
        # Debug: Log audio capture details
        logger.debug(f"Audio callback received data with shape: {indata.shape}, dtype: {indata.dtype}")

        # Debug: Log audio level
        max_audio_level = np.abs(indata).max()
        logger.debug(f"Audio level (max): {max_audio_level:.4f}")

        # Log every second
        if current_time - self.last_log_time >= 1.0:
            self.last_log_time = current_time
            # Add audio level check to determine if audio is being captured
            max_audio_level = np.abs(indata).max()
            logger.debug(f"Audio callback active. Queue size: {self.audio_queue.qsize()}, Level: {max_audio_level:.4f}")
            
            # Log audio levels in dBFS in the callback
            rms = np.sqrt(np.mean(np.square(indata)))
            dbfs = 20 * np.log10(rms) if rms > 0 else -float('inf')
            logger.debug(f"Audio callback level (dBFS): {dbfs:.2f}")
            
        if status:
            logger.warning(f"Audio stream status: {status}")
            
        if self.running:
            # Check if the audio is silent
            if np.abs(indata).max() < 0.001:  # Very low audio level
                return  # Skip processing silent frames
            
            # Throttle in audio_callback
            if self.audio_queue.qsize() > 450:
                logger.warning("Audio queue near full, dropping frame")
                return
            
            try:
                if not self.audio_queue.full():
                    self.audio_queue.put(indata.copy())  # INGEN await här!
                else:
                    logger.warning("Audio queue full, dropping frame")
            except queue.Full:
                logger.warning("Audio queue full, dropping frame")

    def run(self):
        """Process audio data from the queue."""
        logger.info("Audio processing thread started.")
        accumulated_audio = []
        
        while self.running:
            try:
                # Use a timeout to avoid blocking for too long
                audio_data = self.audio_queue.get(timeout=0.1)
                
                # Accumulate audio until we have enough for processing
                accumulated_audio.append(audio_data)
                
                # Increase the amount of accumulated audio to ensure enough data for the model
                # Process ~1 second of audio instead of 0.5 seconds
                if len(accumulated_audio) >= 10:  # Increased from 5 to 10 chunks
                    audio_chunk = np.concatenate(accumulated_audio)
                    
                    # Log audio level and shape for debug purposes
                    audio_level = np.abs(audio_chunk).max()
                    logger.debug(f"Processing audio chunk, max level: {audio_level:.4f}, shape: {audio_chunk.shape}, threshold: {self.silence_threshold}")
                    
                    # Only process if audio is loud enough
                    if audio_level >= self.silence_threshold:
                        text = self.process_audio(audio_chunk)
                        
                        if text and text.strip():
                            logger.info(f"[WS -> Client] Transcribed text: {text}")
                            
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
                            else:
                                logger.warning("No websocket connection available to send transcription")
                        else:
                            logger.debug("No transcription result from audio chunk (empty text)")
                    else:
                        logger.debug(f"Audio level {audio_level:.4f} below threshold {self.silence_threshold}, skipping")
                    
                    # Keep a small overlap for continuous speech
                    accumulated_audio = accumulated_audio[-2:] if len(accumulated_audio) > 2 else []
            except queue.Empty:
                pass  # No audio data in queue
            except Exception as e:
                logger.error(f"Error processing audio data: {e}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
        
        logger.info("Audio processing thread stopped.")

    def process_audio(self, audio_data: np.ndarray) -> str:
        """Process the audio data and return the transcription."""
        try:
            # Skip processing if audio is too quiet (redundant check, already done in run())
            if np.abs(audio_data).max() < self.silence_threshold:
                logger.debug(f"Audio too quiet in process_audio, level: {np.abs(audio_data).max():.4f}")
                return ""
            
            # Use the StreamingProcessor's process_streaming method
            logger.debug(f"Calling process_streaming with audio shape: {audio_data.shape}")
            text = self.processor.process_streaming(audio_data)
            logger.info(f"Processed text: {text}")  # Add logging
            
            # Debug: Log transcription result
            logger.debug(f"Transcription result: {text}")
            
            if text:
                # Send the text through WebSocket
                if self.websocket:
                    logger.info(f"📤 Sending to WebSocket: '{text}'")
                    message_queue.put({
                        "websocket": self.websocket,
                        "message": {
                            "type": "transcription",
                            "text": text,
                            "timestamp": time.time()
                        }
                    })
                    logger.info("✅ Message queued for sending")
                    # Debug: Log when transcription is sent to WebSocket
                    logger.debug(f"[WebSocket] Sending transcription: {text}")
            else:
                logger.debug("process_streaming returned empty text")
                
            # Debug: Log transcription result before sending to WebSocket
            logger.debug(f"Transcription result ready to send: {text}")

            return text
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            import traceback
            logger.error(f"Process audio error trace: {traceback.format_exc()}")
            return ""

# === WEBSOCKET HANDLER ===
active_connections: Set[WebSocketServerProtocol] = set()

# === GLOBAL STATE ===
server_running = False
transcription_running = False
processing_thread: Optional[ProcessingThread] = None

# Process outgoing messages in the main asyncio loop
async def process_messages():
    """Process outgoing WebSocket messages from the queue"""
    logger.info("Started message processing task")
    while True:
        await asyncio.sleep(0.05)  # Faster check to reduce latency
        try:
            # Non-blocking check for messages
            try:
                item = message_queue.get_nowait()
                websocket = item["websocket"]
                message = item["message"]
                
                # Check if the WebSocket is still open
                if not websocket.open:
                    logger.warning("WebSocket is closed, can't send message")
                    message_queue.task_done()
                    continue
                
                try:
                    # Convert the message to JSON
                    message_json = json.dumps(message)
                    logger.debug(f"Sending transcription to client: {message_json[:100]}...")
                    
                    # Debug: Log when sending transcription to WebSocket client
                    logger.debug(f"Sending transcription to WebSocket client: {message.get('text', '')}")
                    
                    # Send the message and wait for it to complete
                    await websocket.send(message_json)
                    logger.debug("Message sent successfully")
                except Exception as send_err:
                    logger.error(f"Error sending message: {send_err}")
                
                # Mark the task as done
                message_queue.task_done()
                
                # Debug: Log WebSocket message details
                logger.debug(f"WebSocket message queued: {message}")
                
            except queue.Empty:
                pass  # No messages in queue
        except Exception as e:
            logger.error(f"Error in message processor: {e}")
            import traceback
            logger.error(f"Message processor error trace: {traceback.format_exc()}")
            await asyncio.sleep(1)  # Wait a bit longer on error

async def handle_websocket(websocket: WebSocketServerProtocol):
    global server_running, transcription_running, processing_thread
    try:
        active_connections.add(websocket)
        logger.info("Ny WebSocket-anslutning etablerad")
        await websocket.send(json.dumps({
            "type": "status",
            "status": "connected",
            "timestamp": time.time()
        }))

        async for message in websocket:
            try:
                data = json.loads(message)
            except json.JSONDecodeError:
                logger.error("Invalid JSON received")
                await websocket.send(json.dumps({"type": "error", "error": "Invalid JSON"}))
                continue

            cmd = data.get('command')
            if cmd == 'start_server':
                if not server_running:
                    server_running = True
                    await websocket.send(json.dumps({"type": "status", "status": "server_started", "timestamp": time.time()}))
                else:
                    await websocket.send(json.dumps({"type": "status", "status": "server_already_running", "timestamp": time.time()}))

            elif cmd == 'stop_server':
                if server_running:
                    server_running = False
                    transcription_running = False
                    if processing_thread and processing_thread.running:
                        processing_thread.stop_recording()
                        processing_thread.running = False
                    await websocket.send(json.dumps({"type": "status", "status": "server_stopped", "timestamp": time.time()}))
                else:
                    await websocket.send(json.dumps({"type": "status", "status": "server_already_stopped", "timestamp": time.time()}))

            elif cmd == 'start_transcription':
                if not transcription_running:
                    transcription_running = True
                    if not processing_thread or not processing_thread.running:
                        processing_thread = ProcessingThread(websocket)
                        processing_thread.start()
                    else:
                        processing_thread.websocket = websocket
                    await websocket.send(json.dumps({"type": "status", "status": "transcription_started", "timestamp": time.time()}))
                else:
                    await websocket.send(json.dumps({"type": "status", "status": "transcription_already_running", "timestamp": time.time()}))

            elif cmd == 'stop_transcription':
                if transcription_running:
                    transcription_running = False
                    if processing_thread and processing_thread.running:
                        processing_thread.stop_recording()
                        processing_thread.running = False
                    await websocket.send(json.dumps({"type": "status", "status": "transcription_stopped", "timestamp": time.time()}))
                else:
                    await websocket.send(json.dumps({"type": "status", "status": "transcription_already_stopped", "timestamp": time.time()}))

            else:
                await websocket.send(json.dumps({"type": "error", "error": f"Unknown command: {cmd}"}))

    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.send(json.dumps({"type": "error", "error": str(e)}))
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

async def start_server(audio_processor=None, config=None):
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
        if sys.platform == "win32" and sys.version_info >= (3, 8):
            import asyncio
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        await start_websocket_server()
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        raise

if __name__ == "__main__":
    try:
        # Försök använda porten i config, annars hitta en ledig port
        port_to_use = PORT
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('localhost', PORT))
        except OSError:
            logger.warning(f"Port {PORT} is in use, searching for a free port...")
            port_to_use = find_free_port(PORT + 1)
            logger.info(f"Using free port {port_to_use} instead.")
        PORT = port_to_use  # Sätt globalt
        # Skriv porten till fil för frontend
        with open("websocket_port.txt", "w") as f:
            f.write(str(PORT))
        asyncio.run(start_server())
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        print(f"Failed to start server: {e}")
