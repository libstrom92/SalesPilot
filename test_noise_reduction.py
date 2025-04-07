import sounddevice as sd
import numpy as np
import noisereduce as nr

# Spela in 5 sekunder bakgrundsbrus
input("Spela in brus (tryck Enter)...")
noise = sd.rec(int(5 * 16000), samplerate=16000, channels=1)
sd.wait()

# Spela in röst
input("Prata nu (tryck Enter)...")
audio = sd.rec(int(5 * 16000), samplerate=16000, channels=1)
sd.wait()

# Jämför
clean = nr.reduce_noise(y=audio.flatten(), y_noise=noise.flatten(), sr=16000)
sd.play(np.hstack([audio.flatten(), clean]), samplerate=16000)
sd.wait()
