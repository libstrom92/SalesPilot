import sounddevice as sd
import numpy as np
from audio_monitor import AudioLevelMonitor, create_level_meter
from logging_config import setup_logging
import os
import time
from dotenv import load_dotenv
import argparse
import signal
import soundfile as sf
from config import load_config
from audio_processor import AudioProcessor
import threading
from audio_normalizer import normalize_audio_to_target

logger = setup_logging("AudioTest")

def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully"""
    print("\n\nAvslutar...")
    raise KeyboardInterrupt

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Testa ljudingång och nivåer eller putthrough")
    parser.add_argument("--putthrough", action="store_true", help="Kör direkt putthrough (mikrofon till högtalare)")
    parser.add_argument("--device", type=int,
                       help="Ljudenhets-ID (åsidosätter AUDIO_DEVICE_ID)")
    return parser.parse_args()

def get_device_id(args):
    """Get device ID from arguments or environment"""
    if args.device is not None:
        return args.device
        
    load_dotenv()
    try:
        return int(os.getenv("AUDIO_DEVICE_ID", "2"))
    except ValueError:
        logger.warning("Ogiltigt AUDIO_DEVICE_ID, använder standard (2)")
        return 2

def test_audio_input(device_id: int):
    """Test audio input with live monitoring and playback, tills Ctrl+C"""
    monitor = AudioLevelMonitor()
    max_level = 0
    min_level = 1
    recorded_audio = []
    start_time = time.time()
    stop = False

    def audio_callback(indata, frames, time_info, status):
        nonlocal max_level, min_level, recorded_audio
        if status:
            logger.warning(f"Ljudstatus: {status}")
        recorded_audio.append(indata.copy())
        monitor.add_level(indata)
        level = monitor.get_average_level()
        if level:
            max_level = max(max_level, level)
            min_level = min(min_level, level)
        meter = create_level_meter(level if level is not None else 0.0)
        elapsed = int(time.time() - start_time)
        print(f"\rNivå: {meter} | Max: {max_level:.2f} | Min: {min_level:.2f} | {elapsed}s ", end="")

    print(f"\nTestar ljudingång från enhet {device_id}...")
    print("Prata in i mikrofonen eller spela upp ljud. Avsluta med Ctrl+C...\n")
    try:
        with sd.InputStream(device=device_id, channels=1, callback=audio_callback, samplerate=16000):
            while True:
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n\nInspelning avbruten av användaren.")
    except Exception as e:
        logger.error(f"Ljudtest misslyckades: {e}")
        print("\nTips för felsökning:")
        print("1. Kör 'python hitta_enhet.py' för att se tillgängliga ljudenheter")
        print("2. Kontrollera att mikrofonen är ansluten och fungerar")
        print("3. Testa en annan ljudenhet med '--device ID'")
        return

    print(f"\n\nTestresultat:")
    print(f"Max nivå: {max_level:.2f}")
    print(f"Min nivå: {min_level:.2f}")
    if max_level < 0.1:
        print("\n⚠️ Varning: Mycket låg ljudnivå detekterad!")
        print("Tips: Kontrollera att rätt mikrofon är vald och att volymen är uppskruvad")
    elif max_level > 0.9:
        print("\n⚠️ Varning: Mycket hög ljudnivå detekterad!")
        print("Tips: Sänk mikrofonvolymen eller öka avståndet till mikrofonen")
    else:
        print("\n✅ Ljudnivåerna ser bra ut!")

    # Normalisera ljudet
    recorded_audio = np.concatenate(recorded_audio, axis=0)
    recorded_audio = normalize_audio_to_target(recorded_audio, target_peak=0.8)

    def transcribe_and_print(audio):
        print("\n📝 Transkriberar inspelat ljud...")
        config = load_config()
        processor = AudioProcessor(config)
        try:
            transcription, summary, diarization = processor.process_audio(audio)
            print("\n--- TRANSKRIBERING ---\n")
            print(transcription)
            print("\n---------------------\n")
        except Exception as e:
            print(f"Fel vid transkribering: {e}")


    def play_audio_safe(audio_data, samplerate):
        import sounddevice as sd
        try:
            # Försök spela upp på default-enheten
            sd.play(audio_data, samplerate)
            sd.wait()
            return True
        except Exception as e:
            print(f"[VARNING] Standarduppspelning misslyckades: {e}")
            # Försök hitta första tillgängliga output device
            try:
                devices = sd.query_devices()
                output_devices = [i for i, d in enumerate(devices) if d['max_output_channels'] > 0]
                for dev_id in output_devices:
                    try:
                        print(f"Försöker spela upp på enhet {dev_id}: {devices[dev_id]['name']}")
                        sd.play(audio_data, samplerate, device=dev_id)
                        sd.wait()
                        print(f"Uppspelning lyckades på enhet {dev_id}: {devices[dev_id]['name']}")
                        return True
                    except Exception as e2:
                        print(f"Misslyckades på enhet {dev_id}: {e2}")
                print("Ingen output-enhet kunde användas för uppspelning.")
            except Exception as e3:
                print(f"Fel vid felsökning av output-enheter: {e3}")
        return False

    print("\n🔊 Spelar upp inspelat ljud...")
    transcribe_thread = threading.Thread(target=transcribe_and_print, args=(recorded_audio,))
    transcribe_thread.start()
    play_audio_safe(recorded_audio, 16000)
    transcribe_thread.join()

def main():
    """Main entry point"""
    signal.signal(signal.SIGINT, signal_handler)
    args = parse_args()
    device_id = get_device_id(args)
    print("\n--- STARTA TEST ---")
    test_audio_input(device_id)

if __name__ == "__main__":
    args = parse_args()
    device_id = get_device_id(args)
    # Ta bort putthrough-läget, kör alltid vanlig inspelning/test
    main()