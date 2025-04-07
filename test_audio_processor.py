import unittest
import numpy as np
import logging
from audio_processor import AudioProcessor
import os
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class TestAudioProcessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Runs once before all tests"""
        load_dotenv()
        cls.config = {
            "whisper_model": "medium",
            "device": "cpu",
            "compute_type": "int8",
            "sample_rate": 16000
        }
        try:
            cls.processor = AudioProcessor(cls.config)
            logger.info("AudioProcessor initialized for tests")
        except Exception as e:
            logger.error(f"Could not initialize AudioProcessor: {e}")
            raise

    def generate_test_audio(self, duration=3, frequency=440):
        """Generate a test tone"""
        t = np.linspace(0, duration, int(self.config["sample_rate"] * duration))
        return np.sin(2 * np.pi * frequency * t).astype(np.float32)

    def test_initialization(self):
        """Test that AudioProcessor is initialized correctly"""
        self.assertIsNotNone(self.processor)
        
        # Test model initialization
        self.assertIsNotNone(self.processor.model)
        self.assertIsNotNone(self.processor.speaker_diarization)
        
        # Test configuration
        self.assertEqual(self.processor.sample_rate, self.config["sample_rate"])
        self.assertEqual(self.processor.whisper_model, self.config["whisper_model"])
        self.assertEqual(self.processor.compute_type, self.config["compute_type"])
        
        logger.info("✅ Initialization test completed")

    def test_preprocess_audio(self):
        """Test audio preprocessing"""
        test_audio = self.generate_test_audio()
        try:
            processed = self.processor._preprocess_audio(test_audio)
            self.assertIsNotNone(processed)
            self.assertEqual(processed.shape, test_audio.shape)
            logger.info("✅ Preprocessing test completed")
        except Exception as e:
            logger.error(f"Preprocessing test failed: {e}")
            raise

    def test_noise_reduction(self):
        """Test noise reduction"""
        clean_signal = self.generate_test_audio()
        
        # Test different noise levels
        noise_levels = [0.1, 0.3, 0.5]
        for level in noise_levels:
            noise = np.random.normal(0, level, clean_signal.shape)
            noisy_signal = clean_signal + noise
            processed = self.processor._preprocess_audio(noisy_signal)
            self.assertLess(np.std(processed), np.std(noisy_signal))

    def test_transcribe_audio(self):
        """Test transcription"""
        test_audio = self.generate_test_audio(duration=5)  # 5 seconds of test audio
        try:
            transcription = self.processor._transcribe_audio(test_audio)
            self.assertIsInstance(transcription, str)
            logger.info("✅ Transcription test completed")
        except Exception as e:
            logger.error(f"Transcription test failed: {e}")
            raise

    def test_process_audio_pipeline(self):
        """Test the entire processing pipeline"""
        test_audio = self.generate_test_audio(duration=3)
        try:
            results = self.processor.process_audio(test_audio)
            self.assertIsInstance(results, dict)
            self.assertIn('preprocess', results)
            self.assertIn('transcribe', results)
            self.assertIn('diarize', results)
            logger.info("✅ Pipeline test completed")
        except Exception as e:
            logger.error(f"Pipeline test failed: {e}")
            raise

    def test_generate_summary(self):
        """Test summary generation functionality"""
        test_audio = self.generate_test_audio(duration=3)
        try:
            # Test with single segment
            summary = self.processor._generate_summary(test_audio)
            self.assertIsInstance(summary, str)
            
            # Test with multiple segments
            for _ in range(3):
                self.processor._generate_summary(test_audio)
            
            # Verify conversation buffer exists and contains items
            self.assertTrue(hasattr(self.processor, '_conversation_buffer'))
            self.assertGreater(len(self.processor._conversation_buffer), 0)
            
            logger.info("✅ Summary generation test completed")
        except Exception as e:
            logger.error(f"Summary generation test failed: {e}")
            raise

    def test_conversation_history(self):
        """Test conversation history management"""
        test_audio = self.generate_test_audio(duration=2)
        try:
            # Test conversation update
            self.processor._update_conversation("Test transcription")
            self.assertTrue(hasattr(self.processor, '_full_conversation'))
            self.assertGreater(len(self.processor._full_conversation), 0)
            
            # Test conversation saving
            test_entry = self.processor._full_conversation[-1]
            self.assertIn('timestamp', test_entry)
            self.assertIn('time', test_entry)
            self.assertIn('text', test_entry)
            
            # Test history limit
            for _ in range(150):  # Add more than the limit
                self.processor._update_conversation(f"Test {_}")
            self.assertLessEqual(len(self.processor._full_conversation), 100)
            
            logger.info("✅ Conversation history test completed")
        except Exception as e:
            logger.error(f"Conversation history test failed: {e}")
            raise

    def tearDown(self):
        """Clean up after each test"""
        # Clean up any generated conversation logs
        import shutil
        if os.path.exists('conversation_logs'):
            shutil.rmtree('conversation_logs')

if __name__ == '__main__':
    unittest.main()
