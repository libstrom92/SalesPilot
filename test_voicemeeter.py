import sounddevice as sd
import numpy as np

SAMPLE_RATE = 48000
DEVICE_VOICEMEETER_B1 = 67  # Se till att detta är rätt ID

def mixed_audio_callback(indata, frames, time, status):
    if status:
        print(f"Status: {status}")
    if np.any(indata):  # Om det finns ljuddata
        print(f"Voicemeeter B1 volym: {np.linalg.norm(indata):.2f}")
        print(f"Exempeldata: {indata[:10]}")  # Visa första 10 samplen
    else:
        print("Ingen ljuddata...")

try:
    with sd.InputStream(callback=mixed_audio_callback, samplerate=SAMPLE_RATE, channels=2, device=DEVICE_VOICEMEETER_B1):
        print("Lyssnar på Voicemeeter B1... Tryck Ctrl+C för att stoppa.")
        while True:
            sd.sleep(100)
except KeyboardInterrupt:
    print("\nStänger av.")
except Exception as e:
    print(f"Fel: {e}")