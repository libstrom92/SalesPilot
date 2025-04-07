import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path
import coloredlogs

def setup_logging(name="VoiceTranscriber", level=logging.INFO):
    """Setup logging configuration with both file and console output"""
    
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear any existing handlers
    logger.handlers = []
    
    # File handler with rotation
    file_handler = RotatingFileHandler(
        log_dir / "transcription.log",
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(level)
    file_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Console handler with colors
    coloredlogs.install(
        level=level,
        logger=logger,
        fmt='%(asctime)s %(name)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S',
        field_styles={
            'asctime': {'color': 'green'},
            'levelname': {'bold': True, 'color': 'black'},
            'name': {'color': 'blue'}
        },
        level_styles={
            'debug': {'color': 'black', 'bright': True},
            'info': {'color': 'black'},
            'warning': {'color': 'yellow'},
            'error': {'color': 'red'},
            'critical': {'color': 'red', 'bold': True}
        }
    )
    
    return logger

def get_logger(name="VoiceTranscriber"):
    """Get or create a logger with the specified name"""
    return logging.getLogger(name)