from src.utils.config import ConfigManager, SeedManager
from src.utils.logger import Logger

CONFIG = ConfigManager()
SeedManager(CONFIG.seed).seed_everything()

__all__ = ["CONFIG", "Logger"]
