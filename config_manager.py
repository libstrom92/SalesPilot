import json
import logging
from pathlib import Path
from typing import Dict, Optional
import shutil
import os
from datetime import datetime

logger = logging.getLogger(__name__)

class ConfigurationManager:
    def __init__(self, config_dir: str = "config", frameworks_dir: str = "frameworks"):
        self.config_dir = Path(config_dir)
        self.frameworks_dir = Path(frameworks_dir)
        self.config_dir.mkdir(exist_ok=True)
        self.frameworks_dir.mkdir(exist_ok=True)
        
        self.default_config_path = self.config_dir / "default_settings.json"
        self.active_config_path = self.config_dir / "active_settings.json"
        self.active_framework = None
        self.settings = self.load_default_settings()

    def load_default_settings(self) -> dict:
        """Load default settings from file"""
        try:
            if self.default_config_path.exists():
                with open(self.default_config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                # Create default settings if they don't exist
                default_settings = {
                    "audio": {
                        "noise_threshold": 20,
                        "sample_rate": 16000,
                        "channels": 1,
                        "chunk_size": 512,
                        "buffer_duration": 2
                    },
                    "language": {
                        "primary": "sv",
                        "secondary": None
                    },
                    "model": {
                        "size": "medium",
                        "compute_type": "float16"
                    },
                    "transcription": {
                        "mode": "balanced",
                        "save_path": "conversation_logs"
                    },
                    "server": {
                        "port": 9091,
                        "websocket_host": "0.0.0.0"
                    }
                }
                self.save_default_settings(default_settings)
                return default_settings
        except Exception as e:
            logger.error(f"Error loading default settings: {e}")
            raise

    def save_default_settings(self, settings: dict) -> bool:
        """Save default settings to file"""
        try:
            with open(self.default_config_path, 'w', encoding='utf-8') as f:
                json.dump(settings, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            logger.error(f"Error saving default settings: {e}")
            return False

    def get_active_settings(self) -> dict:
        """Get current active settings"""
        return self.settings

    def save_active_settings(self) -> bool:
        """Save current settings as active"""
        try:
            with open(self.active_config_path, 'w', encoding='utf-8') as f:
                json.dump(self.settings, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            logger.error(f"Error saving active settings: {e}")
            return False

    def activate_framework(self, framework_id: str) -> bool:
        """Activate a specific framework"""
        try:
            framework_path = self.frameworks_dir / f"{framework_id}.json"
            if not framework_path.exists():
                logger.error(f"Framework {framework_id} not found")
                return False

            with open(framework_path, 'r', encoding='utf-8') as f:
                framework = json.load(f)

            # Backup current settings before applying framework
            if self.active_config_path.exists():
                backup_path = self.config_dir / f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                shutil.copy2(self.active_config_path, backup_path)

            # Apply framework settings
            self.settings.update(framework.get('settings', {}))
            self.active_framework = framework_id
            self.save_active_settings()
            
            logger.info(f"Activated framework: {framework.get('name', framework_id)}")
            return True
        except Exception as e:
            logger.error(f"Error activating framework: {e}")
            return False

    def create_framework(self, name: str, description: str, settings: Optional[dict] = None) -> str:
        """Create a new framework"""
        try:
            framework_id = datetime.now().strftime('%Y%m%d_%H%M%S')
            framework = {
                "id": framework_id,
                "name": name,
                "description": description,
                "settings": settings or self.settings,
                "created": datetime.now().isoformat(),
                "modified": datetime.now().isoformat()
            }

            framework_path = self.frameworks_dir / f"{framework_id}.json"
            with open(framework_path, 'w', encoding='utf-8') as f:
                json.dump(framework, f, indent=2, ensure_ascii=False)

            logger.info(f"Created new framework: {name} ({framework_id})")
            return framework_id
        except Exception as e:
            logger.error(f"Error creating framework: {e}")
            return ""

    def update_framework(self, framework_id: str, settings: dict) -> bool:
        """Update an existing framework"""
        try:
            framework_path = self.frameworks_dir / f"{framework_id}.json"
            if not framework_path.exists():
                logger.error(f"Framework {framework_id} not found")
                return False

            with open(framework_path, 'r', encoding='utf-8') as f:
                framework = json.load(f)

            framework['settings'] = settings
            framework['modified'] = datetime.now().isoformat()

            with open(framework_path, 'w', encoding='utf-8') as f:
                json.dump(framework, f, indent=2, ensure_ascii=False)

            logger.info(f"Updated framework: {framework.get('name', framework_id)}")
            return True
        except Exception as e:
            logger.error(f"Error updating framework: {e}")
            return False

    def delete_framework(self, framework_id: str) -> bool:
        """Delete a framework"""
        try:
            framework_path = self.frameworks_dir / f"{framework_id}.json"
            if not framework_path.exists():
                logger.error(f"Framework {framework_id} not found")
                return False

            # If this is the active framework, deactivate it first
            if self.active_framework == framework_id:
                self.settings = self.load_default_settings()
                self.active_framework = None
                self.save_active_settings()

            framework_path.unlink()
            logger.info(f"Deleted framework: {framework_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting framework: {e}")
            return False

    def list_frameworks(self) -> list:
        """List all available frameworks"""
        frameworks = []
        try:
            for file in self.frameworks_dir.glob("*.json"):
                try:
                    with open(file, 'r', encoding='utf-8') as f:
                        framework = json.load(f)
                        frameworks.append(framework)
                except Exception as e:
                    logger.error(f"Error reading framework {file}: {e}")
        except Exception as e:
            logger.error(f"Error listing frameworks: {e}")
        return frameworks

    def get_framework(self, framework_id: str) -> Optional[dict]:
        """Get a specific framework by ID"""
        try:
            framework_path = self.frameworks_dir / f"{framework_id}.json"
            if not framework_path.exists():
                return None

            with open(framework_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error getting framework {framework_id}: {e}")
            return None