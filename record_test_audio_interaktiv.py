import sounddevice as sd
import numpy as np
import wave

SAMPLE_RATE = 48000
DURATION = 5  # antal sekunder att spela in
OUTPUT_FILE = "test_voicemeeter.wav"

# Lista tillgängliga enheter
print("🎛️ Tillgängliga ljudenheter:\n")
devices = sd.query_devices()
input_devices = []

for i, dev in enumerate(devices):
    if dev['max_input_channels'] > 0:
        input_devices.append((i, dev['name']))
        print(f"[{i}] {dev['name']}")

# Användarval
while True:
    try:
        selected_id = int(input("\n🎚️ Ange ID för ljudingång att spela in från: "))
        if selected_id not in [d[0] for d in input_devices]:
            raise ValueError("Inte en giltig input-enhet.")
        break
    except ValueError as e:
        print(f"❌ Fel: {e}")

print(f"\n🎙️ Spelar in {DURATION} sekunder från enhet ID {selected_id}...")

recording = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16', device=selected_id)
sd.wait()

# Spara som WAV-fil
with wave.open(OUTPUT_FILE, 'wb') as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)  # 16 bit = 2 byte
    wf.setframerate(SAMPLE_RATE)
    wf.writeframes(recording.tobytes())

print(f"✅ Sparad som {OUTPUT_FILE}. Spela upp för att kontrollera ljudet.")
