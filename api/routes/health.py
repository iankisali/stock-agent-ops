from fastapi import Depends
from src.config import Config
from src.logger import get_logger

def get_config():
    """Provide Config instance."""
    return Config()

def get_logger():
    """Provide logger instance."""
    return get_logger()