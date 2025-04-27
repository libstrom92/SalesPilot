import logging
import os
from pathlib import Path
import coloredlogs

def setup_logging(name="VoiceTranscriber", level=logging.INFO):
    """Setup logging configuration with both file and console output"""
    
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers = []

    # En enkel FileHandler utan rotation
    file_handler = logging.FileHandler(
        log_dir / "transcription.log",
        mode='a',
        encoding='utf-8'
    )
    file_handler.setLevel(level)
    file_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

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