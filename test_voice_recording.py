import sounddevice as sd
import numpy as np
import os
from dotenv import load_dotenv
import time
from audio_processor import AudioProcessor
from config import load_config
import logging
from logging_config import setup_logging

# Setup logging
logger = setup_logging("VoiceTest")

def record_and_transcribe():
    """Record audio and transcribe it"""
    # Load configuration
    load_dotenv()
    config = load_config()
    
    # Get device ID from environment
    try:
        device_id = int(os.getenv("AUDIO_DEVICE_ID", "2").strip())
    except ValueError:
        logger.warning("Invalid AUDIO_DEVICE_ID, using default (2)")
        device_id = 2
    
    # Get sample rate from config
    sample_rate = config.get("sample_rate", 16000)
    
    # Initialize audio processor
    processor = AudioProcessor(config)
    
    # Record parameters
    duration = 5  # seconds
    
    print(f"\n=== VOICE RECORDING TEST ===")
    print(f"Recording from device ID: {device_id}")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Duration: {duration} seconds")
    print("\nSpeak now...")
    
    # Record audio
    try:
        audio = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype='float32',
            device=device_id
        )
        
        # Show progress
        for i in range(duration):
            print(f"Recording: {i+1}/{duration} seconds", end="\r")
            sd.sleep(1000)
        
        # Wait for recording to complete
        sd.wait()
        print("\nRecording complete!")
        
        # Process audio
        print("Processing audio...")
        audio_array = np.squeeze(audio)
        
        # Check audio data
        if np.isnan(audio_array).any():
            print("Warning: Audio contains NaN values")
            audio_array = np.nan_to_num(audio_array)
        
        if len(audio_array) == 0:
            print("Error: No audio data recorded")
            return
            
        # Print audio stats
        print(f"Audio shape: {audio_array.shape}")
        print(f"Audio min: {np.min(audio_array):.4f}, max: {np.max(audio_array):.4f}")
        print(f"Audio mean: {np.mean(np.abs(audio_array)):.4f}")
        
        # Transcribe
        try:
            transcription, summary, diarization = processor.process_audio(audio_array)
            
            print("\n=== RESULTS ===")
            print(f"Transcription: {transcription}")
            print(f"Summary: {summary}")
            print(f"Diarization: {diarization}")
            
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            print(f"Error during transcription: {e}")
        
    except Exception as e:
        logger.error(f"Recording error: {e}")
        print(f"Error during recording: {e}")
        
        # List available devices to help troubleshoot
        print("\nAvailable audio devices:")
        devices = sd.query_devices()
        for i, dev in enumerate(devices):
            if dev['max_input_channels'] > 0:
                print(f"ID {i}: {dev['name']} (inputs: {dev['max_input_channels']})")

if __name__ == "__main__":
    try:
        record_and_transcribe()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"Unexpected error: {e}")
