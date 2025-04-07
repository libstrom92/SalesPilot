import numpy as np
import sounddevice as sd
import curses
import time
from audio_monitor import AudioLevelMonitor, create_level_meter
from logging_config import setup_logging
import os

logger = setup_logging("VolymVisualiserare")

def find_input_device():
    """Find the configured input device"""
    try:
        device_id = int(os.getenv("AUDIO_DEVICE_ID", "2"))
        device_info = sd.query_devices(device_id, 'input')
        logger.info(f"Använder ljudenhet: {device_info['name']}")
        return device_id
    except Exception as e:
        logger.error(f"Kunde inte hitta ljudenhet: {e}")
        return None

def draw_meter(stdscr, monitor):
    """Draw the volume meter in the terminal"""
    try:
        # Clear screen
        stdscr.clear()
        
        # Get terminal size
        height, width = stdscr.getmaxyx()
        
        # Draw title
        title = "REALTIDS LJUDNIVÅMÄTARE"
        stdscr.addstr(0, (width - len(title)) // 2, title)
        
        # Draw instructions
        instructions = "Tryck 'q' för att avsluta"
        stdscr.addstr(height-1, (width - len(instructions)) // 2, instructions)
        
        # Get current level
        level = monitor.get_average_level()
        if level is None:
            level = 0
            
        # Draw meter
        meter_width = width - 20
        filled = int(min(1.0, level) * meter_width)
        meter = "█" * filled + "░" * (meter_width - filled)
        
        # Add color based on level
        if level < 0.2:
            color = curses.color_pair(1)  # Blue
        elif level < 0.8:
            color = curses.color_pair(2)  # Green
        else:
            color = curses.color_pair(3)  # Red
            
        # Draw meter with color
        stdscr.addstr(height//2, 10, meter, color)
        
        # Draw numeric value
        value_str = f"{level*100:3.0f}%"
        stdscr.addstr(height//2 + 1, (width - len(value_str)) // 2, value_str)
        
        # Draw peak indicator
        if level > 0.8:
            warning = "! VARNING: HÖG NIVÅ !"
            stdscr.addstr(2, (width - len(warning)) // 2, warning, curses.color_pair(3) | curses.A_BOLD)
            
        stdscr.refresh()
        
    except Exception as e:
        logger.error(f"Fel vid ritning av mätare: {e}")

def audio_callback(indata, frames, time, status):
    """Process incoming audio data"""
    if status:
        logger.warning(f"Status: {status}")
    monitor.add_level(indata)

def main(stdscr):
    # Setup colors
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_BLUE, -1)
    curses.init_pair(2, curses.COLOR_GREEN, -1)
    curses.init_pair(3, curses.COLOR_RED, -1)
    
    # Hide cursor
    curses.curs_set(0)
    
    # Find audio device
    device_id = find_input_device()
    if device_id is None:
        return
        
    try:
        with sd.InputStream(device=device_id,
                          channels=1,
                          callback=audio_callback,
                          samplerate=16000):
            
            while True:
                # Check for 'q' key
                if stdscr.getch() == ord('q'):
                    break
                    
                # Update display
                draw_meter(stdscr, monitor)
                time.sleep(0.1)  # Reduce CPU usage
                
    except Exception as e:
        logger.error(f"Fel vid ljudinspelning: {e}")
        stdscr.addstr(0, 0, f"Fel: {str(e)}")
        stdscr.refresh()
        time.sleep(2)

if __name__ == "__main__":
    monitor = AudioLevelMonitor()
    try:
        curses.wrapper(main)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logger.error(f"Oväntad exception: {e}")
    finally:
        print("\nAvslutar ljudnivåmätaren...")
