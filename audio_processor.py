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
        self.sample_rate = config.get("sample_rate", int(os.getenv("SAMPLE_RATE", 16000)))
        self.whisper_model = config.get("whisper_model", os.getenv("WHISPER_MODEL", "small"))
        self.compute_type = config.get("compute_type", os.getenv("COMPUTE_TYPE", "int8"))
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
                language=os.getenv("WHISPER_LANGUAGE", "sv"),
                task="transcribe",
                vad_filter=False
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
        """Process a small audio chunk for streaming transcription"""
        try:
            # Simple preprocessing (minimal to reduce latency)
            audio_float = audio_chunk.astype(np.float32)
            audio_float = audio_float / (np.max(np.abs(audio_float)) + 1e-7)
            
            # Use faster settings for streaming
            segments, _ = self.model.transcribe(
                audio_float,
                beam_size=1,  # Reduced beam size
                language=os.getenv("WHISPER_LANGUAGE", "sv"),
                task="transcribe",
                vad_filter=True  # Enable VAD to skip silence
            )
            
            # Return just the text
            return " ".join([s.text for s in segments])
        except Exception as e:
            self.logger.error(f"Streaming transcription error: {e}")
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
        """Process audio through the complete pipeline with error handling"""
        MAX_AUDIO_LENGTH = 15 * self.sample_rate  # Reduced from 30 to 15 seconds
        if len(audio_array) > MAX_AUDIO_LENGTH:
            audio_array = audio_array[:MAX_AUDIO_LENGTH]
            self.logger.warning("Audio clip truncated to 30 seconds")

        try:
            # Preprocess audio
            preprocessed = self._preprocess_audio(audio_array)
            
            # Skip diarization if audio is short (for faster processing of short segments)
            if len(audio_array) < 2 * self.sample_rate:  # Less than 2 seconds
                # Skip diarization for very short clips
                transcription = self._transcribe_audio(preprocessed)
                self.logger.debug("Using quick mode for short audio clip")
                return transcription, transcription, {"quick_mode": True}
                
            self.logger.debug("Preprocessing complete")

            # Transcribe
            transcription = self._transcribe_audio(preprocessed)
            self.logger.debug("Transcription complete")

            # Diarize
            diarization = self._identify_speakers(preprocessed)
            self.logger.debug("Speaker diarization complete")

            # Generate summary
            summary = self._generate_summary(preprocessed)
            self.logger.debug("Summary generated")

            # Update conversation history
            self._update_conversation(transcription)
            
            return transcription, summary, diarization  # Return the raw diarization object

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
            
            os.makedirs('conversation_logs', exist_ok=True)
            
            filename = f"conversation_logs/conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self._full_conversation, f, ensure_ascii=False, indent=2)
                
            self._full_conversation = self._full_conversation[-100:]
            self.logger.debug(f"Conversation history saved to {filename}")
            
        except Exception as e:
            self.logger.error(f"Error saving conversation history: {e}")

            # Use the same tokenizer initialized in the constructor to analyze text
        """Analyze text using the model"""
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
            return response
            
        except Exception as e:
            logger.error(f"Error in text analysis: {e}")
            raise AudioProcessingError(f"Could not analyze text: {str(e)}")

# Initialize logger for module-level logs
logger = setup_logging("audio_processor")
