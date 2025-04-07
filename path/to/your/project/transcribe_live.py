import os
import logging
from datetime import datetime
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
from transformers import pipeline

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
socketio = SocketIO(app)

# Set up logging
logging.basicConfig(filename='logs/transcription.log', level=logging.INFO,
                    format='%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# Load the speech-to-text model
transcriber = pipeline('automatic-speech-recognition', model='facebook/wav2vec2-base-960h')

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('audio_data')
def handle_audio_data(data):
    try:
        # Transcribe the audio data
        result = list(transcriber(data, return_timestamps=False))
        text = result[0]['text'] if result else ''
        
        # Save the transcription to a file
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        filename = f'conversation_logs/transcription_{timestamp}.txt'
        with open(filename, 'w') as f:
            f.write(text)
        
        # Emit the transcription to the client
        emit('transcription', text)
        
        # Log the transcription
        logging.info(f'Transcription: {text}')
    except Exception as e:
        logging.error(f'Error transcribing audio: {e}')
        emit('error', 'Error transcribing audio')

if __name__ == '__main__':
    # Create the logs and conversation_logs directories if they don't exist
    os.makedirs('logs', exist_ok=True)
    os.makedirs('conversation_logs', exist_ok=True)
    
    # Run the Flask app
    socketio.run(app, host='0.0.0.0', port=5000)
