import numpy as np
import time
from typing import Tuple
import logging

logger = logging.getLogger(__name__)

class AudioNormalizer:
    """Hanterar ljudnormalisering med realtidsvisualisering."""
    
    def __init__(self, target_level: float = 0.25, 
                 limiter_threshold: float = 0.9,
                 min_threshold: float = 0.005):
        self.target_level = target_level
        self.limiter_threshold = limiter_threshold
        self.min_threshold = min_threshold
        self.last_max_level = 0.0
        self.smoothing_factor = 0.3  # För mjukare nivåändringar
        
    def normalize(self, audio: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """
        Normaliserar ljudet och returnerar normaliserad audio + nivåer.
        
        Returns:
            Tuple[np.ndarray, float, float]: (normaliserad_audio, input_level, output_level)
        """
        if len(audio) == 0:
            return audio, 0.0, 0.0
            
        # Beräkna input-nivå
        input_level = np.max(np.abs(audio))
        
        # Skip processing if too quiet
        if input_level < self.min_threshold:
            return audio, input_level, input_level
            
        # Smooth the gain changes
        target_gain = self.target_level / input_level if input_level > 0 else 1.0
        current_gain = self.target_level / self.last_max_level if self.last_max_level > 0 else target_gain
        smoothed_gain = (current_gain * (1 - self.smoothing_factor) + 
                        target_gain * self.smoothing_factor)
        
        # Apply gain and limiting
        normalized = audio * smoothed_gain
        normalized = np.clip(normalized, -self.limiter_threshold, self.limiter_threshold)
        
        # Measure output level
        output_level = np.max(np.abs(normalized))
        self.last_max_level = input_level  # Update for next iteration
        
        return normalized, input_level, output_level
        
    def visualize_levels(self, input_level: float, output_level: float, width: int = 50):
        """Visar en ASCII-baserad VU-meter för input/output-nivåer."""
        def _make_meter(level: float, label: str) -> str:
            normalized_level = min(1.0, level / self.limiter_threshold)
            bars = int(normalized_level * width)
            meter = f"{label}: "
            
            # Färgkodning baserat på nivå
            if level < self.min_threshold:
                color = "\033[91m"  # Röd (för låg)
            elif level > self.limiter_threshold:
                color = "\033[91m"  # Röd (för hög/klippning)
            elif self.min_threshold <= level <= self.target_level:
                color = "\033[92m"  # Grön (optimal)
            else:
                color = "\033[93m"  # Gul (varning)
                
            meter += color
            meter += "█" * bars
            meter += "░" * (width - bars)
            meter += f" {level:.3f}"
            meter += "\033[0m"  # Återställ färg
            return meter
            
        input_meter = _make_meter(input_level, "Input ")
        output_meter = _make_meter(output_level, "Output")
        
        print("\033[2K\033[G", end="")  # Rensa rad och återgå till början
        print(f"{input_meter}\n{output_meter}", end="\r")
        
    def get_status_message(self, input_level: float, output_level: float) -> str:
        """Genererar ett statusmeddelande baserat på ljudnivåer."""
        if input_level < self.min_threshold:
            return "⚠️ För låg insignal - öka källvolymen"
        elif input_level > self.limiter_threshold:
            return "⚠️ För hög insignal - minska källvolymen"
        elif self.min_threshold <= output_level <= self.target_level:
            return "✅ Optimal ljudnivå"
        else:
            return "⚠️ Justera källvolymen för bättre kvalitet"

def normalize_audio_to_target(audio: np.ndarray, target_peak: float = 0.8) -> np.ndarray:
    """Normalisera ljudet så att dess toppvärde når target_peak (t.ex. 0.8)."""
    if not isinstance(audio, np.ndarray):
        audio = np.array(audio)
    peak = np.max(np.abs(audio))
    if peak == 0:
        return audio
    return audio * (target_peak / peak)

def simple_noise_gate(audio: np.ndarray, threshold: float = 0.01) -> np.ndarray:
    """Nollställer alla sample under tröskeln (enkel noise gate för att ta bort whitenoise)."""
    gated = np.where(np.abs(audio) < threshold, 0, audio)
    return gated

def soft_noise_gate(audio: np.ndarray, threshold: float = 0.01, ratio: float = 0.2) -> np.ndarray:
    """Mjuk expander/gate: sänker volymen på svaga partier istället för att nollställa dem."""
    gated = np.where(np.abs(audio) < threshold, audio * ratio, audio)
    return gated

# Exempelanvändning:
# audio = soft_noise_gate(audio, threshold=0.003, ratio=0.2)
# audio = normalize_audio_to_target(audio, target_peak=0.6)
# audio = simple_noise_gate(audio, threshold=0.01)