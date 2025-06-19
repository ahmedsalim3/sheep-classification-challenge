import torch
from pathlib import Path
from typing import Any, Dict

import yaml
import numpy as np
import random

from src.utils.logger import Logger

logger = Logger()


class SeedManager:
    def __init__(self, seed: int):
        self.seed = seed

    def seed_everything(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.info(f"Seeds set to: {self.seed}")


class ConfigManager:
    """Configuration manager for the project."""

    def __init__(self, config_dir: str = "configs"):
        """
        Initialize configuration manager.

        Args:
        ----
            config_dir: Directory containing configuration files

        """
        self.config_dir = Path(config_dir)
        assert (
            self.config_dir.exists()
        ), f"Configuration directory not found: {self.config_dir}"
        self.config = self._load_configs()

        Path(self.config["models_dir"]).mkdir(parents=True, exist_ok=True)
        Path(self.config["results_dir"]).mkdir(parents=True, exist_ok=True)
        Path(self.config["processed_data_dir"]).mkdir(parents=True, exist_ok=True)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def __getattr__(self, name):
        if name in self.config:
            return self.config[name]
        raise AttributeError(f"'ConfigManager' object has no attribute '{name}'")

    def _load_configs(self) -> Dict[str, Any]:
        """Load and merge all configuration files."""
        config = {}

        paths_config = self._load_yaml(self.config_dir / "config.yml")
        config.update(paths_config)

        return config

    def _load_yaml(self, file_path: Path) -> Dict[str, Any]:
        """Load a YAML file and return its contents as a dictionary."""
        assert file_path.exists(), f"Configuration file not found: {file_path}"

        with open(file_path, "r") as f:
            return yaml.safe_load(f)

    def update(self, key: str, value: Any):
        """Update configuration value."""
        self.config[key] = value
