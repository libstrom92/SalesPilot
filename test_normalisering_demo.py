import numpy as np
import sounddevice as sd
import time
from audio_normalizer import normalize_audio_to_target

SAMPLE_RATE = 16000
DURATION = 1.0  # sekunder

# Skapa låg och hög ton (låg pitch = 220 Hz, hög pitch = 1760 Hz), båda med samma volym
low_pitch_tone = 0.4 * np.sin(2 * np.pi * 220 * np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False))
high_pitch_tone = 0.4 * np.sin(2 * np.pi * 1760 * np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False))

# Spela in ett kort ljudprov
print("\n[TEST] Spela in 2 sekunder mikrofonljud...")
recorded = sd.rec(int(2 * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
sd.wait()
recorded = recorded.flatten()

# Endast normalisering enligt best practice
normalized = normalize_audio_to_target(recorded, target_peak=0.6)

print("\n[PLAYBACK] Låg ton (låg pitch, 220 Hz)")
sd.play(low_pitch_tone, SAMPLE_RATE)
sd.wait()
time.sleep(0.5)

print("[PLAYBACK] Originalinspelning (före normalisering)")
sd.play(recorded, SAMPLE_RATE)
sd.wait()
time.sleep(0.5)

print("[PLAYBACK] Normaliserad inspelning (endast normalisering, best practice)")
sd.play(normalized, SAMPLE_RATE)
sd.wait()

print("\n[KLART] Du har nu hört: låg pitch, original, normaliserat (best practice).")
