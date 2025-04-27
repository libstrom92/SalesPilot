import os
import time
import json
import torch
import sounddevice as sd
import numpy as np
from dotenv import load_dotenv
from pathlib import Path

from logging_config import setup_logging
from audio_monitor import AudioLevelMonitor, create_level_meter
from config import load_config
from audio_processor import AudioProcessor

logger = setup_logging("AudioSetup")


def test_audio_device():
    """Test audio device configuration and capabilities"""
    try:
        device_id = int(os.getenv("AUDIO_DEVICE_ID", "2"))
        device_info = sd.query_devices(device_id, 'input')

        logger.info(f"✓ Hittade ljudenhet: {device_info['name']}")
        logger.info(f"  - Samplingsfrekvens: {device_info['default_samplerate']} Hz")
        logger.info(f"  - Kanaler: {device_info['max_input_channels']}")

        return True
    except Exception as e:
        logger.error(f"✗ Kunde inte hitta eller använda ljudenhet: {e}")
        return False


def test_recording(duration=5):
    """Test recording capabilities and play back"""
    device_id = int(os.getenv("AUDIO_DEVICE_ID", "2"))
    monitor = AudioLevelMonitor()
    recorded_data = []

    def callback(indata, frames, time, status):
        if status:
            logger.warning(f"Status: {status}")
        recorded_data.append(indata.copy())
        monitor.add_level(indata)
        level = monitor.get_average_level()
        if level is not None:
            meter = create_level_meter(level)
            print(f"\rLjudnivå: {meter}", end="")

    try:
        print(f"\nSpela in i {duration} sekunder...")
        with sd.InputStream(device=device_id,
                            channels=1,
                            callback=callback,
                            samplerate=16000):
            sd.sleep(int(duration * 1000))

        print("\n")  # radbryt efter meter

        if not recorded_data:
            logger.error("✗ Ingen ljuddata inspelad")
            return False

        audio_data = np.concatenate(recorded_data, axis=0)
        avg_level = np.abs(audio_data).mean()

        if avg_level < 0.01:
            logger.warning("⚠ Mycket låg ljudnivå detekterad")
        else:
            logger.info(f"✓ Lyckad inspelning, medelnivå: {avg_level:.3f}")

        # Spela upp ljud
        print("Spelar upp inspelningen...")
        sd.play(audio_data, samplerate=16000)
        sd.wait()

        # Transkribera
        config = load_config()
        processor = AudioProcessor(config)
        transcription = processor._transcribe_audio(audio_data)
        print(f"\nTranskriberad text: {transcription if transcription else '[Ingen text upptäckt]'}")

        return True

    except Exception as e:
        logger.error(f"✗ Inspelningstest misslyckades: {e}")
        return False


def test_gpu_support():
    """Test GPU support for audio processing"""
    try:
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            logger.info(f"✓ GPU tillgänglig: {device_name}")
            try:
                torch.zeros(1000000).cuda()
                logger.info("✓ GPU-minnesallokering OK")
                return True
            except Exception as e:
                logger.error(f"✗ GPU-minnestest misslyckades: {e}")
                return False
        else:
            logger.warning("⚠ Ingen GPU tillgänglig, kommer använda CPU")
            return True
    except Exception as e:
        logger.error(f"✗ GPU-test misslyckades: {e}")
        return False


def test_config_files():
    """Test configuration files"""
    try:
        if not Path(".env").exists():
            logger.error("✗ .env-fil saknas")
            return False

        load_dotenv()
        required_vars = ["HF_AUTH_TOKEN", "AUDIO_DEVICE_ID", "SAMPLE_RATE"]
        missing = [var for var in required_vars if not os.getenv(var)]

        if missing:
            logger.error(f"✗ Saknade miljövariabler: {', '.join(missing)}")
            return False

        if Path("audio_config.json").exists():
            with open("audio_config.json") as f:
                json.load(f)
            logger.info("✓ audio_config.json laddad")

        logger.info("✓ Alla konfigurationsfiler OK")
        return True
    except Exception as e:
        logger.error(f"✗ Konfigurationstest misslyckades: {e}")
        return False


def main():
    """Run all audio setup tests"""
    print("\n=== LJUDINSTÄLLNINGSTEST ===\n")

    tests = [
        ("Konfigurationsfiler", test_config_files),
        ("Ljudenhet", test_audio_device),
        ("GPU-stöd", test_gpu_support),
        ("Inspelning", test_recording)
    ]

    results = []
    for name, test_func in tests:
        print(f"\nTestar {name}...")
        result = test_func()
        results.append(result)

    print("\n=== SAMMANFATTNING ===")
    success_rate = sum(results) / len(results) * 100
    print(f"\nTestresultat: {success_rate:.1f}% godkända")

    for (name, _), result in zip(tests, results):
        status = "✓" if result else "✗"
        print(f"{status} {name}")

    if not all(results):
        print("\nTips för felsökning:")
        print("1. Kör 'python hitta_enhet.py' för att hitta rätt ljudenhet")
        print("2. Kontrollera att .env är korrekt konfigurerad")
        print("3. Testa ljudnivåer med 'python volym_visualiserare.py'")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nAvbruten av användaren")
    except Exception as e:
        logger.error(f"Oväntat fel: {e}")
    finally:
        print("\nTestavslutning slutförd")
