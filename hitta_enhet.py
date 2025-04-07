import sounddevice as sd
import numpy as np
from colorama import init, Fore, Style

# Initialize colorama for colored output
init()

def test_device(device_id):
    """Test if a device can be opened for recording"""
    try:
        with sd.InputStream(device=device_id, channels=1, samplerate=16000):
            return True
    except sd.PortAudioError:
        return False

def main():
    print(f"\n{Fore.YELLOW}Looking for audio input devices...{Style.RESET_ALL}\n")

    devices = sd.query_devices()
    input_devices = []
    
    for i, dev in enumerate(devices):
        if dev['max_input_channels'] > 0:
            status = test_device(i)
            input_devices.append((i, dev, status))
            
    if not input_devices:
        print(f"{Fore.RED}No audio input devices found!{Style.RESET_ALL}")
        return

    for i, dev, status in input_devices:
        # Format device information
        color = Fore.GREEN if status else Fore.RED
        status_text = "✓ Available" if status else "✗ Unavailable"
        
        print(f"{color}ID: {i}{Style.RESET_ALL}")
        print(f"  Name: {dev['name']}")
        print(f"  Channels: {dev['max_input_channels']}")
        print(f"  Sample rate: {dev.get('default_samplerate', 'N/A')} Hz")
        print(f"  Status: {status_text}")
        print("-" * 50 + "\n")
        print(f"{Fore.CYAN}Recommended: Choose a device marked as ✓ Available{Style.RESET_ALL}")

    print(f"\n{Fore.YELLOW}To use a device, set AUDIO_DEVICE_ID in .env to the desired ID{Style.RESET_ALL}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
