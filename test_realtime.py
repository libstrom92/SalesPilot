import sounddevice as sd
import numpy as np
import logging
import time
import argparse
from streaming_processor import StreamingProcessor
from config import load_config

# Konfigurera loggning
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Undvik att loggen skrivs dubbelt
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(level=logging.INFO, format='%(asctime)s] [%(levelname)s] %(message)s', datefmt='%H:%M:%S')

# ANSI-färger för konsolutskrift
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
BLUE = '\033[94m'
RESET = '\033[0m'
BOLD = '\033[1m'

def visualize_audio_level(level):
    """Visualisera ljudnivå med en grafisk indikator"""
    threshold = 0.005  # Samma som i StreamingProcessor
    
    max_bars = 50
    bars = int(level * max_bars)
    bars = min(bars, max_bars)
    
    if level < threshold:
        color = RED
    elif level < 0.1:
        color = YELLOW
    else:
        color = GREEN
        
    bar = f"{color}{'|' * bars}{' ' * (max_bars - bars)} {level:.4f}{RESET}"
    print(f"Ljudnivå: {bar}")

def main():
    parser = argparse.ArgumentParser(description="Testa realtidstranskribering med StreamingProcessor")
    parser.add_argument("--duration", type=int, default=5, help="Inspelningslängd per omgång (sekunder)")
    parser.add_argument("--device", type=int, default=None, help="Ljudenhet ID")
    parser.add_argument("--threshold", type=float, default=0.005, help="Silence threshold")
    args = parser.parse_args()
    
    # Ladda konfiguration
    config = load_config()
    
    # Visa tillgängliga ljudenheter
    print(f"{BLUE}{BOLD}Tillgängliga ljudenheter:{RESET}")
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        print(f"{i}: {device['name']}")
    
    # Välj ljudenhet om ingen specificerats
    device_id = args.device
    if device_id is None:
        device_id = int(input("Ange ljudenhetens ID: "))
    
    # Konfigurera parametrar
    sample_rate = int(config.get("sample_rate", 16000))
    print(f"{BLUE}{BOLD}Använder enhet:{RESET} {devices[device_id]['name']}")
    print(f"{BLUE}{BOLD}Sample rate:{RESET} {sample_rate} Hz")
    print(f"{BLUE}{BOLD}Silence threshold:{RESET} {args.threshold}")
    
    # Initiera StreamingProcessor
    processor = StreamingProcessor(config)
    processor.silence_threshold = args.threshold  # Sätt silence threshold
    
    try:
        print(f"{GREEN}{BOLD}== Realtidstranskribering startar =={RESET}")
        print("Tryck Ctrl+C för att avsluta inspelningen.")
        
        test_count = 1
        while True:
            # Header för varje testomgång
            print(f"\n{BLUE}{BOLD}=== Test {test_count} ==={RESET}")
            test_count += 1
            
            # Spela in ljud i realtid
            duration = args.duration
            print(f"Spelar in i {duration} sekunder...")
            audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
            
            # Visa ljudnivåer under inspelning
            nan_count = 0
            nan_limit = 3

            start_time = time.time()
            while time.time() - start_time < duration:
                if len(audio) > 0:
                    current_frame = int((time.time() - start_time) * sample_rate)
                    if current_frame < len(audio):
                        current_audio = audio[:current_frame]
                        if len(current_audio) > 0:
                            level = np.abs(current_audio).max()

                            if np.isnan(level) or np.isinf(level) or level < 0:
                                if nan_count < nan_limit:
                                    print("[Hoppar över ogiltig ljudnivå: nan]")
                                nan_count += 1
                                continue

                            visualize_audio_level(level)
                time.sleep(0.1)
            
            sd.wait()  # Vänta tills inspelningen är klar
            
            # Visa information om inspelningen
            audio_array = np.squeeze(audio)
            max_level = np.abs(audio_array).max()
            mean_level = np.mean(np.abs(audio_array))
            
            print(f"{YELLOW}Max ljudnivå: {max_level:.4f}{RESET}")
            print(f"{YELLOW}Medel ljudnivå: {mean_level:.4f}{RESET}")
            
            # Kontrollera om ljudet är för tyst
            if max_level < processor.silence_threshold:
                print(f"{RED}VARNING: Ljudet är för tyst för att transkriberas (under threshold).{RESET}")
            
            # Process start time
            process_start = time.time()
            
            # Bearbeta ljudet med StreamingProcessor
            text = processor.process_streaming(audio_array)
            
            # Visa resultat
            process_time = time.time() - process_start
            if text.strip():
                print(f"{GREEN}Transkribering ({process_time:.2f}s): {BOLD}{text}{RESET}")
            else:
                print(f"{RED}Ingen transkribering genererades ({process_time:.2f}s){RESET}")
                print("Möjliga orsaker:")
                print("1. Inget tal detekterades i ljudet")
                print("2. Ljudnivån var för låg")
                print("3. VAD-filtret filtrerade bort allt ljud")
                print("4. Whisper-modellen kunde inte generera text")
            
            # Fråga användaren om fortsättning
            if input("\nFörtsätta? (j/n): ").lower() != 'j':
                break
                
    except KeyboardInterrupt:
        print(f"\n{YELLOW}Avslutar...{RESET}")
    except Exception as e:
        logger.error(f"Ett fel inträffade: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
