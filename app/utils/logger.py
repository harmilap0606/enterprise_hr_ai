"""
Logging Configuration for Enterprise HR AI Application.
Per DOCX §20 specification, configures standardized logging to both console
and a persistent log file (logs/app.log) using the required format:
"%(asctime)s | %(levelname)s | %(message)s"
"""

import os
import sys
import logging
from pathlib import Path

# Project root logs directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE_PATH = LOGS_DIR / "app.log"

FORMAT = "%(asctime)s | %(levelname)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logger(name: str = "enterprise_hr_ai", level: int = logging.INFO) -> logging.Logger:
    """Configures and returns a centralized application logger with file and console handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid duplicate handlers if already initialized
    if not logger.handlers:
        formatter = logging.Formatter(fmt=FORMAT, datefmt=DATE_FORMAT)
        
        # 1. File Handler (logs/app.log)
        file_handler = logging.FileHandler(LOG_FILE_PATH, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # 2. Console Handler (stdout)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        logger.propagate = False
        
    return logger


logger = setup_logger()
