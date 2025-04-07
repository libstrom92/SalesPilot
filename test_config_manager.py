import unittest
import json
import os
import shutil
from pathlib import Path
from config_manager import ConfigurationManager
from datetime import datetime

class TestConfigurationManager(unittest.TestCase):
    def setUp(self):
        # Create test directories
        self.test_config_dir = "test_config"
        self.test_frameworks_dir = "test_frameworks"
        Path(self.test_config_dir).mkdir(exist_ok=True)
        Path(self.test_frameworks_dir).mkdir(exist_ok=True)
        
        self.config_manager = ConfigurationManager(
            config_dir=self.test_config_dir,
            frameworks_dir=self.test_frameworks_dir
        )

    def tearDown(self):
        # Clean up test directories
        shutil.rmtree(self.test_config_dir, ignore_errors=True)
        shutil.rmtree(self.test_frameworks_dir, ignore_errors=True)

    def test_default_settings(self):
        """Test that default settings are created and loaded correctly"""
        settings = self.config_manager.get_active_settings()
        self.assertIsNotNone(settings)
        self.assertIn('audio', settings)
        self.assertIn('language', settings)
        self.assertIn('model', settings)
        self.assertEqual(settings['language']['primary'], 'sv')

    def test_create_framework(self):
        """Test framework creation"""
        framework_id = self.config_manager.create_framework(
            name="Test Framework",
            description="Test Description",
            settings={
                "audio": {
                    "noise_threshold": 30,
                    "sample_rate": 44100
                }
            }
        )
        
        self.assertTrue(framework_id)
        framework = self.config_manager.get_framework(framework_id)
        self.assertIsNotNone(framework)
        self.assertEqual(framework['name'], "Test Framework")
        self.assertEqual(framework['settings']['audio']['noise_threshold'], 30)

    def test_activate_framework(self):
        """Test framework activation"""
        # Create a test framework first
        framework_id = self.config_manager.create_framework(
            name="Test Framework",
            description="Test Description",
            settings={
                "audio": {
                    "noise_threshold": 30,
                    "sample_rate": 44100
                }
            }
        )
        
        # Activate the framework
        success = self.config_manager.activate_framework(framework_id)
        self.assertTrue(success)
        
        # Check if settings were updated
        active_settings = self.config_manager.get_active_settings()
        self.assertEqual(active_settings['audio']['noise_threshold'], 30)
        self.assertEqual(active_settings['audio']['sample_rate'], 44100)

    def test_update_framework(self):
        """Test framework updates"""
        # Create initial framework
        framework_id = self.config_manager.create_framework(
            name="Test Framework",
            description="Test Description",
            settings={"audio": {"noise_threshold": 30}}
        )
        
        # Update framework
        new_settings = {
            "audio": {
                "noise_threshold": 40,
                "sample_rate": 48000
            }
        }
        success = self.config_manager.update_framework(framework_id, new_settings)
        self.assertTrue(success)
        
        # Verify update
        framework = self.config_manager.get_framework(framework_id)
        self.assertEqual(framework['settings']['audio']['noise_threshold'], 40)
        self.assertEqual(framework['settings']['audio']['sample_rate'], 48000)

    def test_delete_framework(self):
        """Test framework deletion"""
        # Create framework to delete
        framework_id = self.config_manager.create_framework(
            name="Test Framework",
            description="To be deleted",
            settings={}
        )
        
        # Verify it exists
        self.assertIsNotNone(self.config_manager.get_framework(framework_id))
        
        # Delete it
        success = self.config_manager.delete_framework(framework_id)
        self.assertTrue(success)
        
        # Verify it's gone
        self.assertIsNone(self.config_manager.get_framework(framework_id))

    def test_list_frameworks(self):
        """Test listing all frameworks"""
        # Create multiple frameworks
        framework_ids = []
        for i in range(3):
            framework_id = self.config_manager.create_framework(
                name=f"Framework {i}",
                description=f"Test Framework {i}",
                settings={}
            )
            framework_ids.append(framework_id)
        
        # List frameworks
        frameworks = self.config_manager.list_frameworks()
        self.assertEqual(len(frameworks), 3)
        
        # Verify all created frameworks are in the list
        listed_ids = [f['id'] for f in frameworks]
        for framework_id in framework_ids:
            self.assertIn(framework_id, listed_ids)

if __name__ == '__main__':
    unittest.main()