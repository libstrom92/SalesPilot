import sounddevice as sd
import numpy as np
from audio_monitor import AudioLevelMonitor, create_level_meter
from logging_config import setup_logging
import os
import time
from dotenv import load_dotenv
import argparse
import signal

logger = setup_logging("AudioTest")

def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully"""
    print("\n\nAvslutar...")
    raise KeyboardInterrupt

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Testa ljudingång och nivåer")
    parser.add_argument("--duration", type=int, default=10,
                       help="Testlängd i sekunder (standard: 10)")
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

def test_audio_input(device_id: int, duration: int = 10):
    """Test audio input with live monitoring"""
    monitor = AudioLevelMonitor()
    start_time = time.time()
    max_level = 0
    min_level = 1
    
    def audio_callback(indata, frames, time_info, status):
        nonlocal max_level, min_level
        if status:
            logger.warning(f"Ljudstatus: {status}")
            
        # Update levels
        monitor.add_level(indata)
        level = monitor.get_average_level()
        if level:
            max_level = max(max_level, level)
            min_level = min(min_level, level)
            
        # Show live meter
        elapsed = int(time.time() - start_time)
        remaining = max(0, duration - elapsed)
        meter = create_level_meter(level)
        print(f"\rNivå: {meter} | Max: {max_level:.2f} | Min: {min_level:.2f} | {remaining}s kvar ", end="")
    
    try:
        print(f"\nTestar ljudingång från enhet {device_id}...")
        print("Prata in i mikrofonen eller spela upp ljud...\n")
        
        with sd.InputStream(device=device_id,
                          channels=1,
                          callback=audio_callback,
                          samplerate=16000):
            sd.sleep(duration * 1000)
            
        print("\n\nTestresultat:")
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
            
    except Exception as e:
        logger.error(f"Ljudtest misslyckades: {e}")
        print("\nTips för felsökning:")
        print("1. Kör 'python hitta_enhet.py' för att se tillgängliga ljudenheter")
        print("2. Kontrollera att mikrofonen är ansluten och fungerar")
        print("3. Testa en annan ljudenhet med '--device ID'")

def main():
    """Main entry point"""
    # Setup signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    
    # Parse arguments
    args = parse_args()
    
    # Get device ID
    device_id = get_device_id(args)
    
    try:
        # Run the test
        test_audio_input(device_id, args.duration)
    except KeyboardInterrupt:
        print("\nTest avbrutet av användaren")
    except Exception as e:
        logger.error(f"Oväntat fel: {e}")

if __name__ == "__main__":
    main()