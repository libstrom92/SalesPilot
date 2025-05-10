import wave
import numpy as np

INPUT_FILE = "test_audio_output.wav"
OUTPUT_FILE = "test_audio_output_trimmed.wav"
SAMPLE_RATE = 16000
TRIM_DURATION = 0.22  # sekunder (justera vid behov)
FADE_MS = 25  # fade in/out i millisekunder

# Läs in ljudfilen
def read_wav(filename):
    with wave.open(filename, 'rb') as wf:
        frames = wf.readframes(wf.getnframes())
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    return audio

def write_wav(filename, audio, samplerate):
    audio_int16 = (audio * 32767).astype(np.int16)
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(samplerate)
        wf.writeframes(audio_int16.tobytes())

# Klipp, normalisera och fade
audio = read_wav(INPUT_FILE)
num_samples = int(TRIM_DURATION * SAMPLE_RATE)
audio = audio[:num_samples]

# Normalisera
audio = audio / (np.max(np.abs(audio)) + 1e-8)

# Fade in/out
fade_len = int(FADE_MS * SAMPLE_RATE / 1000)
fade_in = np.linspace(0, 1, fade_len)
fade_out = np.linspace(1, 0, fade_len)
audio[:fade_len] *= fade_in
audio[-fade_len:] *= fade_out

write_wav(OUTPUT_FILE, audio, SAMPLE_RATE)
print(f"✅ Trim och fade klar! Sparad som {OUTPUT_FILE}")
