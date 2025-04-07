import unittest
import numpy as np
import logging
from audio_processor import AudioProcessor
from main_server import start_server
import asyncio
import websockets
import json
import threading
import time
from config import load_config

logger = logging.getLogger(__name__)

class TestIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Setup test environment"""
        cls.config = load_config()
        cls.processor = AudioProcessor(cls.config)
        
        # Start server in separate thread
        cls.server_thread = threading.Thread(
            target=lambda: asyncio.run(cls._run_server()),
            daemon=True
        )
        cls.server_thread.start()
        time.sleep(2)  # Wait for server to start

    @staticmethod
    async def _run_server():
        """Run WebSocket server"""
        try:
            await start_server(AudioProcessor(load_config()), load_config())
        except Exception as e:
            logger.error(f"Server error: {e}")

    async def _connect_client(self):
        """Connect test client to WebSocket server"""
        uri = f"ws://localhost:{self.config['websocket_port']}"
        return await websockets.connect(uri)

    def test_end_to_end(self):
        """Test complete audio processing pipeline"""
        # Generate test audio
        duration = 3
        sample_rate = self.config["sample_rate"]
        test_audio = np.sin(2 * np.pi * 440 * np.linspace(
            0, duration, int(duration * sample_rate)
        )).astype(np.float32)

        # Process audio
        try:
            transcription, summary, diarization = self.processor.process_audio(test_audio)
            
            # Verify results
            self.assertIsInstance(transcription, str)
            self.assertIsInstance(summary, str)
            self.assertIsInstance(diarization, dict)
            
            logger.info("✅ End-to-end test completed successfully")
        except Exception as e:
            logger.error(f"End-to-end test failed: {e}")
            raise

    def test_websocket_communication(self):
        """Test WebSocket communication"""
        async def run_test():
            client = await self._connect_client()
            try:
                # Send test message
                test_data = {"type": "test", "data": "hello"}
                await client.send(json.dumps(test_data))
                
                # Wait for response
                response = await client.recv()
                response_data = json.loads(response)
                
                self.assertIsInstance(response_data, dict)
                logger.info("✅ WebSocket communication test completed")
            finally:
                await client.close()

        asyncio.run(run_test())

    @classmethod
    def tearDownClass(cls):
        """Cleanup after tests"""
        # Stop server
        if hasattr(cls, 'server_thread'):
            cls.server_thread.join(timeout=5)

if __name__ == '__main__':
    unittest.main()
