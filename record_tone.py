import sounddevice as sd
import numpy as np
import wave

SAMPLE_RATE = 16000
DURATION = 1.0  # sekunder
OUTPUT_FILE = "test_audio_output.wav"

print("Prata eller gör din app-ljudton nu (1 sekund)...")
recording = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
sd.wait()

with wave.open(OUTPUT_FILE, 'wb') as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)  # 16 bit = 2 byte
    wf.setframerate(SAMPLE_RATE)
    wf.writeframes(recording.tobytes())

print(f"✅ Sparad som {OUTPUT_FILE}. Du kan nu använda denna i main_server.py!")
