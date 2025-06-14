from pathlib import Path
from typing import Any, Dict

import yaml


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

        output_dir = Path(self.config["output_dir"])
        self.config["model_save_path"] = output_dir / "models"
        self.config["results_save_path"] = output_dir / "results"
        self.config["model_save_path"].mkdir(parents=True, exist_ok=True)
        self.config["results_save_path"].mkdir(parents=True, exist_ok=True)

    def _load_configs(self) -> Dict[str, Any]:
        """Load and merge all configuration files."""
        config = {}

        # Load paths configuration
        paths_config = self._load_yaml(self.config_dir / "paths.yml")
        config.update(paths_config)

        # Load model configuration
        model_config = self._load_yaml(self.config_dir / "model.yml")
        config.update(model_config)

        return config

    def _load_yaml(self, file_path: Path) -> Dict[str, Any]:
        """Load a YAML file and return its contents as a dictionary."""
        assert file_path.exists(), f"Configuration file not found: {file_path}"

        with open(file_path, "r") as f:
            return yaml.safe_load(f)

    def update(self, key: str, value: Any):
        """Update configuration value."""
        self.config[key] = value
