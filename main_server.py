import asyncio
from websockets.legacy.server import WebSocketServerProtocol  # Updated import for legacy protocol
from websockets.legacy.server import serve  # Use legacy server for compatibility
from streaming_processor import StreamingProcessor  # Import the new streaming processor
import logging
import json
import numpy as np  # Flyttad hit för att np alltid ska vara globalt tillgänglig
import sounddevice as sd
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
from gpt_analyzer import analyze_notes
from audio_normalizer import normalize_audio_to_target

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
# === ASYNKRONA ANALYSARBETARE ===
class DiarizationWorker(threading.Thread):
    def __init__(self, audio_queue, websocket=None):
        super().__init__(daemon=True)
        self.audio_queue = audio_queue
        self.websocket = websocket
        self.running = True
    def run(self):
        from audio_processor import AudioProcessor
        processor = AudioProcessor(config)
        while self.running:
            try:
                audio_block = self.audio_queue.get(timeout=1)
                diarization_result = processor.retroactive_diarization(audio_block)
                if self.websocket:
                    message_queue.put({
                        "websocket": self.websocket,
                        "message": {
                            "type": "diarization",
                            "result": diarization_result,
                            "timestamp": time.time()
                        }
                    })
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"DiarizationWorker error: {e}")

class ContextWorker(threading.Thread):
    def __init__(self, text_queue, websocket=None):
        super().__init__(daemon=True)
        self.text_queue = text_queue
        self.websocket = websocket
        self.running = True
    def run(self):
        from gpt_analyzer import analyze_context
        buffer = []
        BLOCK_SIZE = 5
        while self.running:
            try:
                text = self.text_queue.get(timeout=1)
                buffer.append(text)
                if len(buffer) >= BLOCK_SIZE:
                    block_text = " ".join(buffer)
                    context_result = analyze_context(block_text)
                    if self.websocket:
                        message_queue.put({
                            "websocket": self.websocket,
                            "message": {
                                "type": "context_analysis",
                                "context": block_text,
                                "context_result": context_result,
                                "timestamp": time.time()
                            }
                        })
                    buffer.clear()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"ContextWorker error: {e}")

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
        self.last_block_time = time.time()  # Gör till instansvariabel
        self.transcription_start_time = None  # För startfördröjning
        self.silent_start_delay = 4  # sekunder
        self.audio_analysis_queue = queue.Queue()
        self.text_analysis_queue = queue.Queue()
        self.diarization_worker = DiarizationWorker(self.audio_analysis_queue, websocket)
        self.context_worker = ContextWorker(self.text_analysis_queue, websocket)
        self.diarization_worker.start()
        self.context_worker.start()
        
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
        import time
        logger.info("Audio processing thread started.")
        accumulated_audio = []
        block_buffer = []
        BLOCK_SIZE = 1  # 1 block = 0.5 sek ljud
        BLOCK_SAMPLES = 8000  # 0.5 sek vid 16 kHz
        VAD_THRESHOLD = 0.01  # Enkel VAD: block under denna nivå ignoreras
        context_buffer = []
        CONTEXT_BLOCKS = 3
        self.transcription_start_time = None
        self.silent_start_delay = 4
        while self.running:
            try:
                queue_start = time.time()
                audio_data = self.audio_queue.get(timeout=0.1)
                logger.info(f"[TIMER] Got audio from queue at {queue_start:.3f}")
                accumulated_audio.append(audio_data)
                # Samla tills vi har 1 sek ljud (AssemblyAI-style)
                total_samples = sum(len(chunk) for chunk in accumulated_audio)
                if total_samples >= BLOCK_SAMPLES:
                    audio_chunk = np.concatenate(accumulated_audio)[:BLOCK_SAMPLES].flatten()
                    accumulated_audio = []  # Töm bufferten, ingen överlapp
                    max_level = np.abs(audio_chunk).max()
                    logger.info(f"[DEBUG] NYTT BLOCK: max_level={max_level:.4f}, shape={audio_chunk.shape}")
                    # VAD-tröskel AVAKTIVERAD
                    # if (max_level < VAD_THRESHOLD):
                    #     logger.info(f"[VAD] Block ignorerad (max_level={max_level:.4f} < {VAD_THRESHOLD})")
                    #     continue
                    # Normalisera
                    audio_chunk = normalize_audio_to_target(audio_chunk, target_peak=0.7)
                    process_start = time.time()
                    text = self.process_audio(audio_chunk)
                    logger.info(f"[TIMER] process_audio returned at {time.time():.3f} (elapsed: {time.time()-process_start:.3f}s)")
                    logger.info(f"[DEBUG] Transcription result: '{text}'")
                    # --- Silent start delay AVAKTIVERAD ---
                    # if self.transcription_start_time is None:
                    #     self.transcription_start_time = time.time()
                    # if time.time() - self.transcription_start_time < self.silent_start_delay:
                    #     logger.debug("Silent start delay aktiv, hoppar över transkribering.")
                    #     accumulated_audio = accumulated_audio[-2:] if len(accumulated_audio) > 2 else []
                    #     continue
                    # --- VAD-plats (för framtida pyannote/webrtcvad) ---
                    # TODO: Lägg in VAD här om du vill använda pyannote/webrtcvad

                    # --- FÖRBÄTTRAD FILTRERING OCH DUBBELKOLL ---
                    logger.debug(f"[DEBUG] Audio chunk shape: {audio_chunk.shape}, max level: {max_level:.4f}")
                    process_start = time.time()
                    # Normalisera ljudet innan transkribering
                    audio_chunk = normalize_audio_to_target(audio_chunk, target_peak=0.6)
                    text = self.process_audio(audio_chunk)
                    logger.info(f"[TIMER] process_audio returned at {time.time():.3f} (elapsed: {time.time()-process_start:.3f}s)")
                    logger.debug(f"[DEBUG] Transcription result: '{text}'")
                    # ---
                    if text and text.strip():
                        logger.info(f"[WS -> Client] Transcribed text: {text}")
                        ai_feedback = analyze_notes(text)
                        # --- Blockanalys ---
                        block_buffer.append(text)
                        BLOCK_INTERVAL = 5  # sekunder mellan blockanalys
                        if len(block_buffer) >= BLOCK_SIZE or (time.time() - self.last_block_time) > BLOCK_INTERVAL:
                            block_text = " ".join(block_buffer)
                            from gpt_analyzer import analyze_block
                            block_result = analyze_block(block_text)
                            if self.websocket:
                                message_queue.put({
                                    "websocket": self.websocket,
                                    "message": {
                                        "type": "block_analysis",
                                        "block": block_text,
                                        "block_result": block_result,
                                        "timestamp": time.time()
                                    }
                                })
                            block_buffer.clear()
                            self.last_block_time = time.time()
                        # ---
                        # --- Kontextanalys ---
                        context_buffer.append(text)
                        if len(context_buffer) >= BLOCK_SIZE * CONTEXT_BLOCKS:
                            full_text = " ".join(context_buffer)
                            from gpt_analyzer import analyze_context
                            context_result = analyze_context(full_text)
                            if self.websocket:
                                message_queue.put({
                                    "websocket": self.websocket,
                                    "message": {
                                        "type": "context_analysis",
                                        "context": full_text,
                                        "context_result": context_result,
                                        "timestamp": time.time()
                                    }
                                })
                            context_buffer.clear()
                        # ---
                        # Queue message for sending via WebSocket
                        if self.websocket:
                            send_time = time.time()
                            total_elapsed = None
                            if hasattr(self, '_last_transcribe_time'):
                                total_elapsed = send_time - self._last_transcribe_time
                                logger.info(f"[TIMER] Transkribering: tid från start till skickad till frontend: {total_elapsed:.3f}s (själva transkriberingen: {getattr(self, '_last_transcribe_elapsed', 'N/A'):.3f}s)")
                            message_queue.put({
                                "websocket": self.websocket,
                                "message": {
                                    "type": "transcription",
                                    "text": text,
                                    "timestamp": send_time,
                                    "ai_feedback": ai_feedback
                                }
                            })
                            logger.debug(f"Queued transcription message: {text}")
                        else:
                            logger.warning("No websocket connection available to send transcription")
                        # Skicka ljud och text till analysarbetare parallellt
                        self.audio_analysis_queue.put(audio_chunk)
                        self.text_analysis_queue.put(text)
                    else:
                        logger.debug("No transcription result from audio chunk (empty text)")
                    
                    # Playback för felsökning: spela upp ljudet som skickas till transkribering
                    try:
                        print("[DEBUG] Spelar upp ljudfil som skickas till transkribering...")
                        TEST_TONE = False  # Sätt till True för syntetisk ton, False för fil
                        if not TEST_TONE:
                            import wave
                            import numpy as np
                            wf = wave.open("test_audio_output_trimmed.wav", 'rb')
                            samplerate = wf.getframerate()
                            frames = wf.readframes(wf.getnframes())
                            audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                            sd.play(audio, samplerate)
                            sd.wait()
                            wf.close()
                        else:
                            # ...syntetisk tonkod här om du vill växla tillbaka...
                            pass
                    except Exception as e:
                        logger.warning(f"Playback-felsökning misslyckades: {e}")
                        
            except queue.Empty:
                pass  # No audio data in queue
            except Exception as e:
                logger.error(f"Error processing audio data: {e}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
        
        logger.info("Audio processing thread stopped.")

    def process_audio(self, audio_data: np.ndarray) -> str:
        import time
        start_time = time.time()
        logger.info(f"[TIMER] process_audio START {start_time:.3f}")
        try:
            # Skip processing if audio is too quiet (redundant check, already done in run())
            if np.abs(audio_data).max() < self.silence_threshold:
                logger.debug(f"Audio too quiet in process_audio, level: {np.abs(audio_data).max():.4f}")
                return ""
            
            # Use the StreamingProcessor's process_streaming method
            logger.debug(f"Calling process_streaming with audio shape: {audio_data.shape}")
            text = self.processor.process_streaming(audio_data)
            end_time = time.time()
            elapsed = end_time - start_time
            logger.info(f"[TIMER] process_audio END {end_time:.3f} (elapsed: {elapsed:.3f}s)")
            # Spara tiden för vidare loggning när text skickas till frontend
            self._last_transcribe_time = start_time
            self._last_transcribe_elapsed = elapsed
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

def make_json_serializable(obj):
    """Helper för att konvertera objekt till JSON-serialiserbara former."""
    if hasattr(obj, 'to_dict'):
        return obj.to_dict()
    if hasattr(obj, '__dict__'):
        return obj.__dict__
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (set,)):
        return list(obj)
    if isinstance(obj, bytes):
        return obj.decode(errors='replace')
    return str(obj)

def convert_dict_keys_to_str(obj):
    """Rekursivt konvertera alla dict-nycklar till str (för JSON-serialisering)."""
    if isinstance(obj, dict):
        return {str(k): convert_dict_keys_to_str(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_dict_keys_to_str(i) for i in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_dict_keys_to_str(i) for i in obj)
    else:
        return obj

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
                # Konvertera alla dict-nycklar till str innan serialisering
                message_safe = convert_dict_keys_to_str(message)
                logger.info(f"[FRONTEND-DEBUG] Skickar till frontend: {json.dumps(message_safe, ensure_ascii=False, default=make_json_serializable)[:500]}")
                # Check if the WebSocket is still open
                if not websocket.open:
                    logger.warning("WebSocket is closed, can't send message")
                    message_queue.task_done()
                    continue
                
                try:
                    # Convert the message to JSON
                    message_json = json.dumps(message_safe, default=make_json_serializable)
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
            if cmd == 'ping':
                await websocket.send(json.dumps({"type": "pong", "timestamp": time.time()}))
                continue
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
