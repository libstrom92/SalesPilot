import numpy as np
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
import noisereduce as nr
import time
from huggingface_hub import HfApi
import os
from dotenv import load_dotenv
from hf_token_manager import HFTokenManager
import torch
from tenacity import retry, stop_after_attempt, wait_exponential
from logging_config import setup_logging
from typing import Dict, Any, Optional, Tuple

# Load environment variables
load_dotenv()

class AudioProcessingError(Exception):
    """Base exception for audio processing errors"""
    pass

class AudioDeviceError(AudioProcessingError):
    """Raised when there are issues with the audio device"""
    pass

class TranscriptionError(AudioProcessingError):
    """Raised when transcription fails"""
    pass

class DiarizationError(AudioProcessingError):
    """Raised when speaker diarization fails"""
    pass

class AudioProcessor:
    def __init__(self, config: Dict[str, Any]):
        self.logger = setup_logging("AudioProcessor")
        self.config = config
        self.min_segment_length = 0.5  # Minimum length in seconds
        self.buffer = np.array([])  # Audio buffer for accumulating short segments
        self.sample_rate = config.get("sample_rate", int(os.getenv("SAMPLE_RATE", 16000)))
        self.whisper_model = (
            config.get("whisper_model") or
            config.get("model", {}).get("size") or
            "large-v2"
        )
        self.compute_type = (
            config.get("compute_type") or
            config.get("model", {}).get("compute_type") or
            "int8"
        )
        # Ta bort float16 helt, använd int8 som standard
        self.device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.hf_token = os.getenv("HF_AUTH_TOKEN", "").strip() or HFTokenManager.get_token()
        
        try:
            from transformers import AutoTokenizer  # Import tokenizer from transformers
            
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")  # Initialize tokenizer
            self.model = WhisperModel(
                self.whisper_model,
                device=config.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
                compute_type=self.compute_type,
                download_root=os.path.join(os.path.dirname(__file__), "models")
            )
            self.speaker_diarization = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.0",
                use_auth_token=self.hf_token
            )
            self.logger.info("🎯 Models initialized successfully")
        except ValueError as ve:
            self.logger.critical(f"Initialization error: {ve}")
            raise AudioProcessingError(f"Could not initialize models: {ve}")
        except Exception as e:
            self.logger.error(f"Initialization error: {e}")
            raise AudioProcessingError(f"Unexpected error during initialization: {e}")

    def _preprocess_audio(self, audio: np.ndarray) -> np.ndarray:
        """Preprocess audio with noise reduction and normalization"""
        try:
            if torch.cuda.is_available():
                self.logger.debug("Using GPU for noise reduction")
                audio_tensor = torch.from_numpy(audio).cuda()
                self.logger.warning("GPU noise reduction is not supported. Falling back to CPU.")
                raise NotImplementedError("GPU noise reduction is not available in the noisereduce module.")
        except Exception as e:
            self.logger.warning(f"GPU noise reduction failed, using CPU: {e}")

        audio_float = audio.astype(np.float32)  # Initialize audio_float outside the try block
        
        # Quick mode for very short clips to reduce latency
        if len(audio_float) < 256:
            self.logger.warning("Audio clip is too short for advanced noise reduction")
            return audio_float / (np.max(np.abs(audio_float)) + 1e-7)
            
        try:
            if len(audio_float) < 2 * self.sample_rate:  # Less than 2 seconds
                # Use faster, simpler noise reduction for short clips
                reduced = audio_float - np.mean(audio_float)
                return reduced / (np.max(np.abs(reduced)) + 1e-7)
        except Exception as e:
            self.logger.warning(f"Fast noise reduction failed: {e}")
            # Continue with standard noise reduction

        # Use standard noise reduction for longer clips
        try:
            if len(audio_float) < 256:
                self.logger.warning("Audio clip is too short for advanced noise reduction")
                return audio_float / (np.max(np.abs(audio_float)) + 1e-7)
                
            reduced = nr.reduce_noise(
                y=audio_float,
                sr=self.sample_rate,
                stationary=True,
                prop_decrease=0.75,
                n_fft=512,
                win_length=512,
                hop_length=128  # This ensures noverlap (384) is less than nperseg (512)
            )
            return reduced / (np.max(np.abs(reduced)) + 1e-7)
        except Exception as e:
            self.logger.warning(f"Noise reduction failed: {e}")
            return audio_float / (np.max(np.abs(audio_float)) + 1e-7)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _transcribe_audio(self, audio: np.ndarray) -> str:
        """Transcribe audio with error handling and retries"""
        try:
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)

            segments, _ = self.model.transcribe(
                audio,
                beam_size=5,
                best_of=2,  # Added for slightly better accuracy
                language=os.getenv("WHISPER_LANGUAGE", "sv"),  # Set default language to Swedish
                task="transcribe",
                condition_on_previous_text=True,  # Enable context awareness
                vad_filter=True,  # Enable VAD
                # vad_segments=True,  # (BORTTAGET: stöds ej i denna version)
                no_speech_threshold=0.05  # Lower threshold for detecting speech
            )
            segments = list(segments)
            
            self.logger.debug(f"Transcribed {len(segments)} segments")
            
            if not segments:
                return ""

            transcription = " ".join([s.text for s in segments])
            self.logger.info(f"✨ Transcription complete: {transcription[:100]}...")
            return transcription
        except Exception as e:
            self.logger.error(f"Transcription error: {e}")
            raise TranscriptionError(f"Could not transcribe audio: {e}")

    def process_streaming(self, audio_chunk: np.ndarray) -> str:
        """Process a small audio chunk for streaming transcription with detailed logging."""
        try:
            # Filter out silent audio segments before processing
            if np.abs(audio_chunk).max() < 0.001:
                self.logger.debug("Skipping silent audio segment")
                return ""

            # Add new chunk to buffer
            self.logger.debug(f"Received audio chunk of size: {len(audio_chunk)}")
            self.buffer = np.concatenate([self.buffer, audio_chunk]) if self.buffer.size else audio_chunk

            # Add logging for audio levels in dBFS
            rms = np.sqrt(np.mean(np.square(audio_chunk)))
            dbfs = 20 * np.log10(rms) if rms > 0 else -float('inf')
            self.logger.debug(f"Ljudnivå (dBFS): {dbfs:.2f}")

            # Check if buffer is long enough
            min_samples = int(self.min_segment_length * self.sample_rate)
            if len(self.buffer) < min_samples:
                self.logger.debug(f"Buffer too short ({len(self.buffer)} samples), waiting for more audio...")
                return ""  # Buffer too short, wait for more audio

            # Process buffer
            audio_float = self.buffer.astype(np.float32)
            audio_float = audio_float / (np.max(np.abs(audio_float)) + 1e-7)

            # Clear buffer after processing
            self.buffer = np.array([])

            # Use faster settings for streaming
            self.logger.debug("Processing audio buffer for transcription...")
            segments, _ = self.model.transcribe(
                audio_float,
                beam_size=1,  # Reduced beam size for faster processing
                best_of=1,  # Simplified for real-time performance
                language=os.getenv("WHISPER_LANGUAGE", "sv"),  # Set default language to Swedish
                task="transcribe",
                condition_on_previous_text=False,  # Disable context awareness for speed
                vad_filter=True,  # Enable VAD
                no_speech_threshold=0.1  # Lower threshold for detecting speech
            )

            # Log transcription result
            if segments:
                transcription = " ".join([s.text for s in segments])
                self.logger.info(f"Generated transcription: {transcription}")
                return transcription
            else:
                self.logger.debug("No transcription generated for the current audio buffer.")
                return ""
        except Exception as e:
            self.logger.error(f"Streaming transcription error: {e}")
            return ""

    def process_streaming_with_context(self, audio_chunk: np.ndarray, previous_context: str = "") -> str:
        """Process a small audio chunk for streaming transcription with context awareness."""
        try:
            # Filter out silent audio segments before processing
            if np.abs(audio_chunk).max() < 0.001:
                self.logger.debug("Skipping silent audio segment")
                return ""

            # Add new chunk to buffer
            self.logger.debug(f"Received audio chunk of size: {len(audio_chunk)}")
            self.buffer = np.concatenate([self.buffer, audio_chunk]) if self.buffer.size else audio_chunk

            # Add logging for audio levels in dBFS
            rms = np.sqrt(np.mean(np.square(audio_chunk)))
            dbfs = 20 * np.log10(rms) if rms > 0 else -float('inf')
            self.logger.debug(f"Ljudnivå (dBFS): {dbfs:.2f}")

            # Check if buffer is long enough
            min_samples = int(self.min_segment_length * self.sample_rate)
            if len(self.buffer) < min_samples:
                self.logger.debug(f"Buffer too short ({len(self.buffer)} samples), waiting for more audio...")
                return ""  # Buffer too short, wait for more audio

            # Process buffer
            audio_float = self.buffer.astype(np.float32)
            audio_float = audio_float / (np.max(np.abs(audio_float)) + 1e-7)

            # Clear buffer after processing
            self.buffer = np.array([])

            # Use context-aware settings for streaming
            self.logger.debug("Processing audio buffer with context for transcription...")
            segments, _ = self.model.transcribe(
                audio_float,
                beam_size=3,  # Slightly higher beam size for better accuracy
                best_of=2,  # Balance between speed and quality
                language=os.getenv("WHISPER_LANGUAGE", "sv"),  # Set default language to Swedish
                task="transcribe",
                condition_on_previous_text=True,  # Enable context awareness
                initial_prompt=previous_context,  # Use previous context
                vad_filter=True,  # Enable VAD
                no_speech_threshold=0.1  # Lower threshold for detecting speech
            )

            # Log transcription result
            if segments:
                transcription = " ".join([s.text for s in segments])
                self.logger.info(f"Generated transcription with context: {transcription}")
                return transcription
            else:
                self.logger.debug("No transcription generated for the current audio buffer.")
                return ""
        except Exception as e:
            self.logger.error(f"Streaming transcription with context error: {e}")
            return ""

    def _identify_speakers(self, audio: np.ndarray) -> Dict[str, Any]:
        """Identify speakers with error handling"""
        try:
            if len(audio.shape) == 1:
                audio = np.expand_dims(audio, axis=0)
            elif audio.shape[1] != 1:
                audio = np.mean(audio, axis=1, keepdims=True)
            
            tensor_audio = torch.tensor(audio, dtype=torch.float32)
            if tensor_audio.shape[1] == 1:
                tensor_audio = tensor_audio.T

            self.logger.debug(f"Diarization input shape: {tensor_audio.shape}")

            if tensor_audio.ndim != 2 or tensor_audio.shape[0] != 1:
                raise ValueError(f"Incorrect tensor shape for diarization: {tensor_audio.shape}")

            diarization = self.speaker_diarization({
                "waveform": tensor_audio,
                "sample_rate": self.sample_rate
            })
            
            self.logger.info("✨ Speaker diarization complete")
            return diarization
        except Exception as e:
            self.logger.error(f"Speaker diarization failed: {e}")
            raise DiarizationError(f"Could not identify speakers: {e}")

    def process_audio(self, audio_array: np.ndarray) -> Tuple[str, str, Dict[str, Any]]:
        """Process audio through the complete pipeline with error handling and improved diarization"""
        # Ta bort trunkering, processa hela ljudet
        try:
            preprocessed = self._preprocess_audio(audio_array)

            # --- Förbättrad diarizering: segmentera ljudet ---
            segment_len = 10 * self.sample_rate  # 10 sekunder
            overlap = 2 * self.sample_rate       # 2 sekunder överlapp
            diarization_results = []
            for start in range(0, len(preprocessed), segment_len - overlap):
                end = min(start + segment_len, len(preprocessed))
                segment = preprocessed[start:end]
                if len(segment) < 2 * self.sample_rate:
                    continue  # hoppa över för korta segment
                try:
                    diarization = self._identify_speakers(segment)
                    diarization_results.append(diarization)
                except Exception as e:
                    self.logger.warning(f"Diarization failed for segment {start}-{end}: {e}")

            # Slå ihop diarization-resultat (enkel union, kan förbättras)
            diarization_summary = {'segments': [d for d in diarization_results if d]}

            # Transkribera hela klippet
            transcription = self._transcribe_audio(preprocessed)
            summary = self._generate_summary(preprocessed)
            self._update_conversation(transcription)
            return transcription, summary, diarization_summary

        except (TranscriptionError, DiarizationError) as e:
            self.logger.error(f"Processing error: {e}")
            raise AudioProcessingError(str(e))
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
            raise AudioProcessingError(f"Unexpected error during audio processing: {e}")

    def _generate_summary(self, audio_preprocessed: np.ndarray) -> str:
        """Generate a summary by analyzing speaker segments and transcription"""
        try:
            if not hasattr(self, '_conversation_buffer'):
                self._conversation_buffer = []
            
            transcription = self._transcribe_audio(audio_preprocessed)
            diarization = self._identify_speakers(audio_preprocessed)
            
            if not transcription or not diarization:
                return "No spoken text detected"
            
            self._conversation_buffer.append({
                'time': time.strftime('%H:%M:%S'),
                'text': transcription,
                'speakers': diarization
            })
            
            self._conversation_buffer = self._conversation_buffer[-10:]
            
            if len(self._conversation_buffer) > 1:
                summary = f"Latest conversation ({len(self._conversation_buffer)} segments):\n"
                for segment in self._conversation_buffer[-3:]:
                    summary += f"[{segment['time']}] {segment['text']}\n"
                return summary
            else:
                return transcription
        except Exception as e:
            self.logger.error(f"Error generating summary: {e}")
            return "Could not generate summary"

    def _update_conversation(self, transcription: str) -> None:
        """Update conversation history with new transcription"""
        try:
            if not hasattr(self, '_full_conversation'):
                self._full_conversation = []
            
            self._full_conversation.append({
                'timestamp': time.time(),
                'time': time.strftime('%Y-%m-%d %H:%M:%S'),
                'text': transcription
            })
            
            if len(self._full_conversation) % 10 == 0:
                self._save_conversation_history()
                
        except Exception as e:
            self.logger.error(f"Error updating conversation history: {e}")

    def _save_conversation_history(self) -> None:
        """Save conversation history to file"""
        try:
            if not hasattr(self, '_full_conversation'):
                return
                
            import json
            from datetime import datetime
            
            os.makedirs('conversation_logs', exist_okay=True)
            
            filename = f"conversation_logs/conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self._full_conversation, f, ensure_ascii=False, indent=2)
                
            self._full_conversation = self._full_conversation[-100:]
            self.logger.debug(f"Conversation history saved to {filename}")
            
        except Exception as e:
            self.logger.error(f"Error saving conversation history: {e}")

            # Use the same tokenizer initialized in the constructor to analyze text
        """Analyze text using the model

        Returns:
            str: The generated response from the model.
        """
        try:
            # Använd samma modell som för transkription för att analysera text
            prompt = "Provide a valid prompt here"  # Define the prompt variable
            inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
            
            # Generera svar
            with torch.no_grad():
                from transformers import GPT2LMHeadModel
                text_generation_model = GPT2LMHeadModel.from_pretrained("gpt2").to(self.device)
                outputs = text_generation_model.generate(
                    torch.tensor(inputs["input_ids"]).to(self.device),
                    max_length=200,
                    num_return_sequences=1,
                    temperature=0.7
                )
            
            # Avkoda svaret
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response # type: ignore
            
        except Exception as e:
            logger.error(f"Error in text analysis: {e}")
            raise AudioProcessingError(f"Could not analyze text: {str(e)}")

    def process_realtime(self, audio_stream: np.ndarray, segment_seconds: int = 5) -> str:
        """Transkriberar inkommande ljud i realtid, segment för segment (ingen diarizering)."""
        segment_len = segment_seconds * self.sample_rate
        results = []
        for start in range(0, len(audio_stream), segment_len):
            end = min(start + segment_len, len(audio_stream))
            segment = audio_stream[start:end]
            if len(segment) < int(0.5 * self.sample_rate):
                continue  # hoppa över för korta segment
            try:
                # Snabb transkribering, ingen diarizering
                text = self._transcribe_audio_realtime(segment)
                if text.strip():
                    results.append(text.strip())
            except Exception as e:
                self.logger.warning(f"Realtime transcription failed for segment {start}-{end}: {e}")
        return " ".join(results)

    def _transcribe_audio_realtime(self, audio: np.ndarray) -> str:
        """Snabb transkribering för realtid (beam_size=1, best_of=1, ingen context)."""
        try:
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            segments, _ = self.model.transcribe(
                audio,
                beam_size=1,
                best_of=1,
                language=os.getenv("WHISPER_LANGUAGE", "sv"),
                task="transcribe",
                condition_on_previous_text=False,
                vad_filter=True,
                no_speech_threshold=0.1
            )
            segments = list(segments)
            if not segments:
                return ""
            return " ".join([s.text for s in segments])
        except Exception as e:
            self.logger.error(f"Realtime transcription error: {e}")
            return ""

    def retroactive_diarization(self, audio_array: np.ndarray) -> Dict[str, Any]:
        """Kör diarizering på hela ljudet retroaktivt efter samtalet."""
        preprocessed = self._preprocess_audio(audio_array)
        try:
            diarization = self._identify_speakers(preprocessed)
            self.logger.info("Retroaktiv diarizering klar.")
            return diarization
        except Exception as e:
            self.logger.error(f"Retroaktiv diarizering misslyckades: {e}")
            return {"error": str(e)}

# Initialize logger for module-level logs
logger = setup_logging("audio_processor")
