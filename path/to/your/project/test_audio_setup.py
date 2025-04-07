import os
import sys
import sounddevice as sd
import numpy as np

def test_audio_setup():
    print("Testing audio setup...")

    # Check if default audio device is available
    default_device = sd.default.device
    if default_device[0] == -1 or default_device[1] == -1:
        print("No default audio device found. Please check your audio setup.")
        sys.exit(1)

    # Test audio input
    print("Testing audio input...")
    duration = 5  # Record for 5 seconds
    sample_rate = 16000
    channels = 1
    myrecording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=channels)
    sd.wait()
    print("Audio input test complete.")

    # Test audio output
    print("Testing audio output...")
    test_tone = np.sin(2 * np.pi * np.arange(sample_rate * duration) / sample_rate * 440.0).astype(np.float32)
    sd.play(test_tone, blocking=True)
    print("Audio output test complete.")

    print("Audio setup test passed!")

if __name__ == "__main__":
    test_audio_setup()
