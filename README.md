# Voice Transcription System

Real-time voice transcription system using Whisper and speaker diarization.

## Quick Start

1. Double-click `activate_env.bat` and follow the prompts
2. Choose option 2 to run audio diagnostics if this is your first time
3. Choose option 1 to start the transcription system

## Detailed Setup

1. Create a Python virtual environment (automatically handled by activate_env.bat):
```bash
python -m venv myenv
source myenv/bin/activate  # On Linux/Mac
myenv\Scripts\activate     # On Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure your settings in .env (will be created automatically):
   - Get your Hugging Face token from https://huggingface.co/settings/tokens
   - Set `HF_AUTH_TOKEN` in the .env file
   - Adjust audio device and other settings as needed

## Diagnostic Tools

The system includes several diagnostic tools to help you set up and troubleshoot:

- `test_audio.bat` - Menu-driven audio diagnostics suite
- `hitta_enhet.py` - List available audio devices
- `volym_visualiserare.py` - Real-time volume level monitor
- `test_audio_input.py` - Quick audio input test
- `test_audio_setup.py` - Comprehensive system check

## Configuration

Key settings in `.env`:
- `HF_AUTH_TOKEN`: Your Hugging Face API token (required)
- `AUDIO_DEVICE_ID`: ID of your audio input device (default: 2)
- `SAMPLE_RATE`: Audio sample rate (default: 16000)
- `WHISPER_MODEL`: Whisper model size (default: medium)
- `COMPUTE_TYPE`: Computation type (default: int8)
- `WHISPER_LANGUAGE`: Transcription language (default: sv)

## Features

- Real-time transcription with Whisper
- Speaker diarization
- Noise reduction with GPU acceleration when available
- Audio level monitoring and warnings
- Conversation history with automatic saving
- Comprehensive logging system
- Error recovery and retry mechanisms

## Troubleshooting

If you experience issues:

1. Run `test_audio.bat` and follow the diagnostic steps
2. Check the logs in the `logs` directory
3. Verify your audio device settings with `hitta_enhet.py`
4. Monitor audio levels with `volym_visualiserare.py`

## Architecture

- `main_server.py`: WebSocket server and audio processing coordination
- `audio_processor.py`: Core audio processing and transcription logic
- `transcribe_live.py`: Main entry point for the transcription server
- `audio_monitor.py`: Audio level monitoring and visualization
- `logging_config.py`: Centralized logging configuration

## Development

- Logs are stored in `logs/transcription.log` with rotation
- Conversation history is saved in `conversation_logs/`
- Unit tests are available in test_audio_processor.py
- Run tests with `run_tests.bat`