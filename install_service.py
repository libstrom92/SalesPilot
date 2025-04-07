import sys
import os
import win32serviceutil
import win32service
import win32event
import servicemanager
import socket
import asyncio
import subprocess
from pathlib import Path

class VoiceTranscriptionService(win32serviceutil.ServiceFramework):
    _svc_name_ = "VoiceTranscriptionService"
    _svc_display_name_ = "Voice Transcription Service"
    _svc_description_ = "Runs the voice transcription WebSocket server automatically"

    def __init__(self, args):
        win32serviceutil.ServiceFramework.__init__(self, args)
        self.stop_event = win32event.CreateEvent(None, 0, 0, None)
        self.process = None

    def SvcStop(self):
        self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
        win32event.SetEvent(self.stop_event)
        if self.process:
            self.process.terminate()

    def SvcDoRun(self):
        try:
            # Get the path to the virtual environment Python
            script_dir = Path(__file__).parent
            venv_python = script_dir / "myenv" / "Scripts" / "python.exe"
            server_script = script_dir / "main_server.py"

            # Start the server process
            self.process = subprocess.Popen([
                str(venv_python),
                str(server_script)
            ])

            # Wait for the stop event
            win32event.WaitForSingleObject(self.stop_event, win32event.INFINITE)

        except Exception as e:
            servicemanager.LogErrorMsg(f"Error in service: {e}")
            raise

if __name__ == '__main__':
    if len(sys.argv) == 1:
        servicemanager.Initialize()
        servicemanager.PrepareToHostSingle(VoiceTranscriptionService)
        servicemanager.StartServiceCtrlDispatcher()
    else:
        win32serviceutil.HandleCommandLine(VoiceTranscriptionService)