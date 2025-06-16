from .utils.config import ConfigManager, SeedManager
from .utils.logger import Logger

CONFIG = ConfigManager()
SeedManager(CONFIG.seed).seed_everything()

__all__ = ["CONFIG", "Logger"]
