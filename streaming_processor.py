import numpy as np
import os
from faster_whisper import WhisperModel
import torch
from logging_config import setup_logging
from typing import Dict, Any, Optional, List, Tuple
import time
import queue
import threading

logger = setup_logging("StreamingProcessor")

class StreamingProcessor:
    """Optimized processor for real-time streaming transcription with lower latency"""
    
    def __init__(self, config: Dict[str, Any]):
        self.logger = logger
        self.config = config
        self.sample_rate = config.get("sample_rate", int(os.getenv("SAMPLE_RATE", 16000)))
        self.whisper_model = os.getenv("WHISPER_MODEL", "small")  # Use smaller model for streaming
        self.compute_type = os.getenv("COMPUTE_TYPE", "int8")  # Use int8 for faster processing
        
        self.buffer_size = 4096  # Reduced buffer size for lower latency
        
        # Initialize model with optimized settings
        try:
            self.logger.info(f"Initializing streaming model: {self.whisper_model}")
            self.model = WhisperModel(
                self.whisper_model,
                device=config.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
                compute_type=self.compute_type,
                download_root=os.path.join(os.path.dirname(__file__), "models")
            )
            self.logger.info("✅ Streaming model initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize streaming model: {e}")
            raise
            
        # Buffer for accumulating audio
        self.audio_buffer = []
        self.buffer_lock = threading.Lock()
        self.max_buffer_size = 5 * self.sample_rate  # 5 seconds max
        
        # Processing queue and thread
        self.processing_queue = queue.Queue()
        self.result_callback = None
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
    def add_audio_chunk(self, chunk: np.ndarray) -> None:
        """Add a chunk of audio to the buffer"""
        with self.buffer_lock:
            self.audio_buffer.append(chunk.copy())
            
            # Calculate total buffer length
            total_samples = sum(len(c) for c in self.audio_buffer)
            
            # If buffer exceeds max size, process it
            if total_samples > self.max_buffer_size:
                self._process_buffer()
                
    def set_result_callback(self, callback) -> None:
        """Set callback function to receive transcription results"""
        self.result_callback = callback
        
    def _process_buffer(self) -> None:
        """Process the current audio buffer"""
        with self.buffer_lock:
            if not self.audio_buffer:
                return
                
            # Concatenate buffer chunks
            audio_data = np.concatenate(self.audio_buffer)
            
            # Clear buffer
            self.audio_buffer = []
            
        # Add to processing queue
        self.processing_queue.put(audio_data)
        
    def _processing_loop(self) -> None:
        """Background thread for processing audio"""
        while True:
            try:
                # Get audio from queue
                audio_data = self.processing_queue.get(timeout=1)
                
                # Process audio
                start_time = time.time()
                result = self._transcribe_chunk(audio_data)
                processing_time = time.time() - start_time
                
                self.logger.debug(f"Processed chunk in {processing_time:.2f}s")
                
                # Send result via callback
                if self.result_callback and result:
                    self.result_callback(result)
                    
                self.processing_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error in processing loop: {e}")
                
    def _transcribe_chunk(self, audio: np.ndarray) -> str:
        """Transcribe audio chunk with optimized settings"""
        try:
            # Normalize audio
            audio_float = audio.astype(np.float32)
            max_val = np.max(np.abs(audio_float))
            audio_float = audio_float / (max_val if max_val > 0 else 1.0)
            self.logger.debug(f"Normalized audio chunk with shape: {audio_float.shape}")
            
            # Use optimized transcription settings
            segments, _ = self.model.transcribe(
                audio_float,
                beam_size=1,  # Reduced beam size
                language=os.getenv("WHISPER_LANGUAGE", "sv"),
                task="transcribe",
                vad_filter=True,  # Enable VAD to skip silence
                initial_prompt=None,  # No prompt for faster processing
                word_timestamps=False  # Disable word timestamps
            )
            self.logger.debug(f"Transcription produced {len(list(segments))} segments")
            
            # Collect text
            text = " ".join([s.text for s in segments])
            self.logger.debug(f"Final transcribed text: '{text}'")
            return text.strip()
            
        except Exception as e:
            self.logger.error(f"Transcription error: {e}")
            return ""
            
    def process_streaming(self, audio_chunk: np.ndarray) -> str:
        """Process a small audio chunk for streaming transcription"""
        try:
            # Logga inkommande ljuddata
            self.logger.debug(f"Received audio chunk: shape={audio_chunk.shape}, dtype={audio_chunk.dtype}, min={audio_chunk.min()}, max={audio_chunk.max()}")
            
            # Kontrollera om ljudet innehåller data
            if np.all(audio_chunk == 0) or np.abs(audio_chunk).max() < 0.001:
                self.logger.warning("Audio chunk contains no significant data (silence or zeros)")
                return ""
                
            # Normalisera ljudet
            audio_float = audio_chunk.astype(np.float32)
            max_val = np.max(np.abs(audio_float))
            audio_float = audio_float / (max_val if max_val > 0 else 1.0)
            self.logger.debug(f"Normalized audio chunk with shape: {audio_float.shape}")
            
            # Mät tiden för transkriberingen
            start_time = time.time()
            
            # Transkribera
            segments, info = self.model.transcribe(
                audio_float,
                beam_size=1,
                language=os.getenv("WHISPER_LANGUAGE", "sv"),
                task="transcribe",
                vad_filter=True
            )
            
            # Konvertera segment-iterator till lista för att kunna använda den flera gånger
            segment_list = list(segments)
            processing_time = time.time() - start_time
            
            self.logger.debug(f"Transcription completed in {processing_time:.2f}s: {len(segment_list)} segments detected")
            
            # Ingen text returnerades
            if len(segment_list) == 0:
                self.logger.warning("No segments detected in audio - might be silence or filtered by VAD")
                return ""
            
            # Samla ihop texten
            text = " ".join([s.text for s in segment_list])
            
            if text.strip():
                self.logger.debug(f"Transcribed text: '{text}'")
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
