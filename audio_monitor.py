import sounddevice as sd
import numpy as np
import os
from logging_config import setup_logging
import time
from typing import Optional, List
import threading

logger = setup_logging("AudioMonitor")

# Voicemeeter-specifik konfiguration
VOICEMEETER_NAMES = ["Voicemeeter Out B1", "VB-Audio Voicemeeter VAIO"]
SAMPLE_RATE = 48000

class AudioLevelMonitor:
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.levels: List[float] = []
        self.lock = threading.Lock()
        self.last_alert_time = 0
        self.alert_cooldown = 5  # seconds between alerts
        
    def add_level(self, audio_data: np.ndarray) -> None:
        """Add new audio level measurement"""
        try:
            level = float(np.abs(audio_data).mean())
            with self.lock:
                self.levels.append(level)
                if len(self.levels) > self.window_size:
                    self.levels.pop(0)
                    
            self._check_levels()
        except Exception as e:
            logger.error(f"Fel vid nivåmätning: {e}")
    
    def _check_levels(self) -> None:
        """Check audio levels and log warnings if needed"""
        current_time = time.time()
        if current_time - self.last_alert_time < self.alert_cooldown:
            return
            
        try:
            with self.lock:
                if not self.levels:
                    return
                    
                avg_level = np.mean(self.levels)
                if avg_level < 0.01:  # Very low signal
                    logger.warning("⚠️ Ljudnivån är mycket låg. Kontrollera mikrofonen.")
                    self.last_alert_time = current_time
                elif avg_level > 0.8:  # Near clipping
                    logger.warning("⚠️ Ljudnivån är för hög. Risk för distorsion.")
                    self.last_alert_time = current_time
                    
        except Exception as e:
            logger.error(f"Fel vid nivåkontroll: {e}")
    
    def get_average_level(self) -> Optional[float]:
        """Get the current average audio level"""
        try:
            with self.lock:
                if not self.levels:
                    return None
                return float(np.mean(self.levels))
        except Exception as e:
            logger.error(f"Fel vid medelnivåberäkning: {e}")
            return None
    
    def reset(self) -> None:
        """Reset level history"""
        with self.lock:
            self.levels.clear()
            
def create_level_meter(level: float, width: int = 40) -> str:
    """Create a text-based level meter"""
    try:
        if level is None:
            return "[" + "-" * width + "]"
            
        filled = int(min(1.0, level) * width)
        meter = "[" + "=" * filled + "-" * (width - filled) + "]"
        
        if level < 0.2:
            return f"\033[34m{meter}\033[0m"  # Blue for low
        elif level < 0.8:
            return f"\033[32m{meter}\033[0m"  # Green for good
        else:
            return f"\033[31m{meter}\033[0m"  # Red for high
            
    except Exception as e:
        logger.error(f"Fel vid skapande av nivåmätare: {e}")
        return "[ERROR]"

def find_voicemeeter_device():
    """Hitta Voicemeeter B1 automatiskt"""
    for device in sd.query_devices():
        if isinstance(device, dict) and any(name in device.get('name', '') for name in VOICEMEETER_NAMES):
            print(f"Hittade: {device['name']} (ID: {device['index']})")
            return device['index']
    raise Exception("Voicemeeter B1 hittades inte. Kontrollera att Voicemeeter är öppet.")

def audio_callback(indata, frames, time, status):
    """Avancerad volymdetektering för Voicemeeter"""
    monitor.add_level(indata)
    avg_level = monitor.get_average_level()
    meter = create_level_meter(avg_level if avg_level is not None else 0.0)
    print(f"VOICEMEETER VOLYM: {meter}", end='\r')

def main():
    print("=== VOICEMEETER LJUDBÖVERVAKNING ===")
    device_id = find_voicemeeter_device()
    
    try:
        with sd.InputStream(
            device=device_id,
            channels=2,  # Voicemeeter använder stereo
            callback=audio_callback,
            samplerate=SAMPLE_RATE
        ):
            print("\n✅ Lyssnar på Voicemeeter B1 | Tryck CTRL+C för att avsluta\n")
            while True:
                sd.sleep(1000)
    except Exception as e:
        print(f"\n❌ Fel: {str(e)}")
        print("Tips: Starta om Voicemeeter och försök igen")

if __name__ == "__main__":
    monitor = AudioLevelMonitor()
    try:
        main()
    except KeyboardInterrupt:
        print("\nAvslutar...")