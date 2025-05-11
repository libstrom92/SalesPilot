import numpy as np
import os
from faster_whisper import WhisperModel
import torch
from logging_config import setup_logging
from typing import Dict, Any, Optional, List, Tuple
import time
import queue
import threading
import sounddevice as sd  # Import library for audio playback

logger = setup_logging("StreamingProcessor")

class StreamingProcessor:
    """Optimized processor for real-time streaming transcription with lower latency"""
    
    def __init__(self, config: Dict[str, Any]):
        self.logger = logger
        self.config = config
        # Sätt sample_rate och channels till samma som i test_audio_input.py
        self.sample_rate = 16000  # Matchar test_audio_input.py och main_server.py
        self.channels = 1
        self.device = config.get("audio_device_id", 2)  # Default till 2 om inget anges
        self.whisper_model = (
            config.get("whisper_model") or
            config.get("model", {}).get("size") or
            "large-v2"
        )
        self.compute_type = (
            config.get("compute_type") or
            config.get("model", {}).get("compute_type") or
            "float16"
        )
        
        # Svenska språkinställningar
        self.language = os.getenv("WHISPER_LANGUAGE", "sv")
        self.logger.info(f"Använder språk: {self.language}")
        
        # Domänspecifik kontext för transkription
        self.initial_prompt = os.getenv("WHISPER_PROMPT", 
            "Detta är en transkription av ett samtal inom försäljning, teknik och affärer på svenska. "
            "Tekniktermer, säljtermer och branschspråk bör transkriberas korrekt.")
        self.logger.info("Domänspecifik prompt aktiverad för optimerad transkription")
        
        self.buffer_size = 16384  # Increased buffer size for better handling of audio chunks
        self.min_audio_length = 0.25  # Reduced from 0.5 to 0.25 seconds
        self.silence_threshold = 0.002  # Lowered from 0.005 to ensure valid audio is not skipped
        self.max_queue_size = 10  # Limit the processing queue size to prevent overflows

        # Debug: Log initialization details
        self.logger.debug(f"Sample rate: {self.sample_rate}, Channels: {self.channels}, Device: {self.device}, Buffer size: {self.buffer_size}, Max queue size: {self.max_queue_size}")
        self.logger.debug(f"Whisper model: {self.whisper_model}, Compute type: {self.compute_type}")

        # Debug: Log language and prompt settings
        self.logger.debug(f"Language: {self.language}, Initial prompt: {self.initial_prompt}")

        # Initialize a semaphore to control queue size
        self.queue_semaphore = threading.Semaphore(self.max_queue_size)
        
        # Initialize model with optimized settings
        try:
            self.logger.info(f"Initializing streaming model: {self.whisper_model}")
            self.model = WhisperModel(
                self.whisper_model,
                device=config.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
                compute_type="int8",  # Force int8 for faster processing
                download_root=os.path.join(os.path.dirname(__file__), "models")
            )
            self.logger.info("✅ Streaming model initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize streaming model: {e}")
            raise
            
        # Buffer for accumulating audio
        self.audio_buffer = []
        self.buffer_lock = threading.Lock()
        self.max_buffer_size = int(0.5 * self.sample_rate)  # 0.5 seconds for best practice low latency
        
        # Processing queue and thread
        self.processing_queue = queue.Queue()
        self.result_callback = None
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        # Debug: Print model settings
        self.logger.info(f"Model settings: language={self.language}, model={self.whisper_model}, min_audio={self.min_audio_length}s")
        self.logger.info(f"VAD disabled for debugging")
        
        # Implement fallback mechanism for dropped buffers
        self.dropped_buffers = []  # Store dropped buffers for retry
        
    def add_audio_chunk(self, chunk: np.ndarray) -> None:
        """Add a chunk of audio to the buffer and play it back for verification"""
        with self.buffer_lock:
            self.audio_buffer.append(chunk.copy())

            # Debug: Log audio chunk details
            self.logger.debug(f"Audio chunk received with shape: {chunk.shape}, dtype: {chunk.dtype}")

            # Play the audio chunk for verification
            try:
                sd.play(chunk, samplerate=self.sample_rate)
                sd.wait()  # Wait until playback is finished
                self.logger.info("Audio chunk played back successfully.")
            except Exception as e:
                self.logger.error(f"Error during audio playback: {e}")

            # Calculate total buffer length
            total_samples = sum(len(chunk) for chunk in self.audio_buffer)
            self.logger.debug(f"Total buffer length after appending: {total_samples} samples")

            # Process buffer as soon as it reaches min_audio_length (best practice)
            if total_samples >= int(self.sample_rate * self.min_audio_length):
                self.logger.debug("Buffer reached min_audio_length, triggering processing immediately (best practice)")
                self._process_buffer()
                
    def set_result_callback(self, callback) -> None:
        """Set callback function to receive transcription results"""
        self.result_callback = callback
        
    def _process_buffer(self) -> None:
        """Process the current audio buffer"""
        with self.buffer_lock:
            if not self.audio_buffer:
                self.logger.debug("Audio buffer is empty, skipping processing.")
                return
                
            # Concatenate buffer chunks
            audio_data = np.concatenate(self.audio_buffer)
            self.logger.debug(f"Buffer size before processing: {len(audio_data)} samples")
            
            # Clear buffer
            self.audio_buffer = []
            
        # Add to processing queue if space is available
        if self.queue_semaphore.acquire(blocking=False):
            self.processing_queue.put(audio_data)
        else:
            self.logger.warning("Processing queue is full, storing buffer for retry")
            self.dropped_buffers.append(audio_data)  # Store dropped buffer
        
    def _retry_dropped_buffers(self):
        """Retry processing dropped buffers"""
        while self.dropped_buffers:
            buffer = self.dropped_buffers.pop(0)
            if self.queue_semaphore.acquire(blocking=False):
                self.processing_queue.put(buffer)
            else:
                self.logger.warning("Retry failed, re-storing buffer")
                self.dropped_buffers.insert(0, buffer)
                break
        
    def _processing_loop(self) -> None:
        """Background thread for processing audio"""
        while True:
            try:
                # Retry dropped buffers
                self._retry_dropped_buffers()

                # Measure queue retrieval time
                start_queue = time.time()
                audio_data = self.processing_queue.get(timeout=1)
                self.queue_semaphore.release()  # Release semaphore when processing starts
                logger.debug(f"Queue retrieval took: {time.time() - start_queue:.4f} seconds")
                
                # Process audio
                start_transcription = time.time()
                result = self._transcribe_chunk(audio_data)
                logger.debug(f"Transcription took: {time.time() - start_transcription:.4f} seconds")
                
                self.logger.debug(f"Processed chunk in {time.time() - start_transcription:.2f}s")
                
                # Send result via callback
                if self.result_callback and result:
                    self.result_callback(result)
                    
                self.processing_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error in processing loop: {e}")
                
    def _transcribe_chunk(self, audio: np.ndarray) -> str:
        """Transcribe audio chunk with best-in-class settings"""
        try:
            # Skip processing if audio is too quiet
            if np.abs(audio).max() < 0.01:
                return ""
                
            # Normalize audio
            audio_float = audio.astype(np.float32)
            max_val = np.max(np.abs(audio_float))
            audio_float = audio_float / (max_val if max_val > 0 else 1.0)
            self.logger.debug(f"Normalized audio chunk with shape: {audio_float.shape}")
            
            # Debug: Log audio data before processing
            self.logger.debug(f"Processing audio data with shape: {audio.shape}, dtype: {audio.dtype}")

            # Use optimized transcription settings for svenska
            segments, _ = self.model.transcribe(
                audio_float,
                beam_size=2,
                best_of=2,
                language=self.language,
                task="transcribe",
                condition_on_previous_text=True,
                initial_prompt=self.initial_prompt,
                vad_filter=True,
                no_speech_threshold=0.2,
                word_timestamps=False
            )
            self.logger.debug(f"Transcription produced {len(list(segments))} segments")
            
            # Collect text
            text = " ".join([s.text for s in segments if s.text.strip()])
            self.logger.debug(f"Final transcribed text: '{text}'")
            
            # Debug: Log transcription result
            self.logger.debug(f"Generated transcription: {text}")

            return text.strip()
            
        except Exception as e:
            self.logger.error(f"Transcription error: {e}")
            return ""
            
    def process_streaming(self, audio_chunk: np.ndarray) -> str:
        """Process a small audio chunk for streaming transcription with best-in-class settings."""
        try:
            # Log some audio samples for debugging
            if len(audio_chunk) > 0:
                sample_view = audio_chunk[:min(100, len(audio_chunk))]
                self.logger.debug(f"First few samples: min={np.min(sample_view):.4f}, max={np.max(sample_view):.4f}, mean={np.mean(np.abs(sample_view)):.4f}")
                
                # Check if we have any non-zero values (real audio)
                if np.count_nonzero(sample_view) == 0:
                    self.logger.warning("Audio contains only zeros! Check audio source.")
                
            # Check if audio chunk is too short
            if len(audio_chunk) < int(self.sample_rate * self.min_audio_length):
                self.logger.debug(f"Audio chunk too short ({len(audio_chunk)} samples < {int(self.sample_rate * self.min_audio_length)}), buffering...")
                return ""
                
            # Skip processing if audio is too quiet
            max_val = np.abs(audio_chunk).max()
            self.logger.debug(f"Audio max level: {max_val:.4f}, Silence threshold: {self.silence_threshold}")
            if max_val < self.silence_threshold:
                self.logger.debug(f"Audio too quiet ({max_val:.4f} < {self.silence_threshold})")
                return ""
                
            # Normalize audio and convert to float32
            audio_float = audio_chunk.astype(np.float32)
            max_val = np.abs(audio_float).max()
            audio_float = audio_float / (max_val if max_val > 0 else 1.0)
            self.logger.info(f"Processing audio chunk: shape={audio_float.shape}, duration={len(audio_float)/self.sample_rate:.2f}s")
            
            # Measure the time for the transcription
            start_time = time.time()
            
            # Transcribe med svenska optimering - VAD DISABLED for debugging
            segments, info = self.model.transcribe(
                audio_float,
                beam_size=2,
                best_of=2,
                language=self.language,
                task="transcribe",
                vad_filter=True,
                initial_prompt=self.initial_prompt,
                condition_on_previous_text=True,
                no_speech_threshold=0.2,
                compression_ratio_threshold=2.4
            )
            
            # Convert segment iterator to list to use it multiple times
            segment_list = list(segments)
            processing_time = time.time() - start_time
            
            self.logger.info(f"Transcription completed in {processing_time:.2f}s: {len(segment_list)} segments detected")
            
            # No text was returned
            if len(segment_list) == 0:
                self.logger.warning("No segments detected in audio - might be silence or filtered")
                return ""
            
            # Collect the text
            text = " ".join([s.text for s in segment_list if s.text.strip()])

            # === Filtrering mot hallucination/prompt ===
            prompt_clean = self.initial_prompt.strip().lower()
            text_clean = text.strip().lower()
            # Skicka inte om texten är tom, identisk med prompten, eller för kort
            if not text_clean:
                self.logger.info("Ingen text genererad (tom sträng)")
                return ""
            if text_clean == prompt_clean:
                self.logger.info("Transkribering matchar prompten - skickas ej till frontend")
                return ""
            if len(text_clean) < 8:  # Justera längdtröskel vid behov
                self.logger.info(f"Transkribering för kort ('{text_clean}') - skickas ej till frontend")
                return ""

            if text.strip():
                self.logger.info(f"🎯 Got transcription segments: {len(segment_list)}")
                self.logger.info(f"✨ Transcribed text: '{text}'")
            else:
                self.logger.warning("Empty text returned after transcription")
            
            return text.strip()
            
        except torch.cuda.OutOfMemoryError as e:
            self.logger.error(f"CUDA out of memory error: {e}")
            torch.cuda.empty_cache()
            return ""
        except Exception as e:
            self.logger.error(f"Streaming error: {e}")
            import traceback
            self.logger.error(f"Transcription error trace: {traceback.format_exc()}")
            return ""
            
    def flush(self) -> str:
        """Process any remaining audio in the buffer"""
        with self.buffer_lock:
            if not self.audio_buffer:
                return ""
                
            # Concatenate buffer chunks
            audio_data = np.concatenate(self.audio_buffer)
            
            # Clear buffer
            self.audio_buffer = []
            
        # Process directly
        return self._transcribe_chunk(audio_data)
        
    # Metod för att träna en anpassad modell (framtida implementering)
    def fine_tune_model(self, training_data_path: str, domain: str = "sälj_teknik"):
        """
        Förbereder för finjustering av Whisper-modellen för specifik domän
        Notera: Denna metod är en plats för framtida implementering
        
        Args:
            training_data_path: Sökväg till träningsdata (ljud + transkriptioner)
            domain: Vilken domän modellen ska optimeras för (t.ex. "sälj", "teknik")
        """
        self.logger.info(f"Fine-tuning för domän {domain} planerad (framtida funktion)")
        # TODO: Implementera finjustering med HuggingFace eller liknande

# TODO: Lägg till stöd för domänspecifik prompt och dialektigenkänning
