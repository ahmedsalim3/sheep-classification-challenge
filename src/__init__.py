from .utils.config import ConfigManager
from .utils.logger import Logger

CONFIG = ConfigManager()

__all__ = ["CONFIG", "Logger"]
