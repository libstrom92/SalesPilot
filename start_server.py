import os
import sys
import webbrowser
import subprocess
import signal
import time
try:
    import psutil # type: ignore
except ImportError:
    print("ERROR: The 'psutil' module is not installed. Please install it using 'pip install psutil'.")
    sys.exit(1)
from pathlib import Path
import logging
from logging_config import setup_logging

logger = setup_logging("ServerManager")

def is_server_running():
    """Check if the transcription server is already running"""
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if proc.info['name'] == 'python.exe' and 'main_server.py' in ' '.join(proc.info['cmdline']):
                logger.info(f"Found running server with PID {proc.pid}")
                return proc.pid
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    logger.info("No running server found")
    return None

def start_server():
    """Start the transcription server"""
    if is_server_running():
        print("WARNING: Server is already running!")
        logger.warning("Attempted to start server when already running")
        return False

    script_dir = Path(__file__).parent
    venv_python = script_dir / "myenv" / "Scripts" / "python.exe"
    server_script = script_dir / "main_server.py"

    if not venv_python.exists():
        print("ERROR: Could not find Python in virtual environment!")
        logger.error("Python not found in virtual environment")
        return False

    try:
        # Start server in background
        subprocess.Popen([
            str(venv_python),
            str(server_script)
        ], creationflags=subprocess.CREATE_NEW_CONSOLE)
        
        print("INFO: Starting server...")
        logger.info("Server process started")
        time.sleep(2)  # Wait for server to start

        # Open web interface
        webbrowser.open('http://localhost:9091')
        logger.info("Opened web interface")
        return True
    except Exception as e:
        print(f"ERROR: Error starting server: {e}")
        logger.error(f"Error starting server: {e}")
        return False

def stop_server():
    """Stop the transcription server"""
    pid = is_server_running()
    if pid:
        try:
            os.kill(pid, signal.SIGTERM)
            print("INFO: Server stopped")
            logger.info(f"Stopped server with PID {pid}")
            return True
        except Exception as e:
            print(f"ERROR: Could not stop server: {e}")
            logger.error(f"Error stopping server: {e}")
            return False
    else:
        print("INFO: Server is not running")
        logger.info("Attempted to stop server when not running")
        return False
