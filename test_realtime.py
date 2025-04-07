import sounddevice as sd
import numpy as np
import logging
from audio_processor import AudioProcessor

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def main():
    # Konfiguration för AudioProcessor
    config = {
        "whisper_model": "medium",
        "device": "cpu",
        "compute_type": "int8",
        "auth_token": "your_auth_token_here",  # Ersätt med ditt faktiska token
        "sample_rate": 16000
    }

    processor = AudioProcessor(config)

    try:
        print("Tryck Ctrl+C för att avsluta inspelningen.")
        while True:
            # Spela in ljud i realtid
            duration = 5  # Sekunder
            print(f"Spelar in i {duration} sekunder...")
            audio = sd.rec(int(duration * config["sample_rate"]), samplerate=config["sample_rate"], channels=1, dtype='int16')
            sd.wait()  # Vänta tills inspelningen är klar

            # Bearbeta ljudet
            audio_array = np.squeeze(audio)
            transcription, summary, diarization = processor.process_audio(audio_array)

            # Visa resultat
            print("Transkription:", transcription)
            print("Sammanfattning:", summary)
            print("Diarization:", diarization)

    except KeyboardInterrupt:
        print("\nAvslutar...")
    except Exception as e:
        logger.error(f"Ett fel inträffade: {e}")

if __name__ == "__main__":
    main()
