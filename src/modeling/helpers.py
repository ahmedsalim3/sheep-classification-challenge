import os
import torch
from tqdm import tqdm

from src.modeling.classifier import ViTClassifier
from src.utils.config import ConfigManager
from src.utils.logger import Logger

CONFIG = ConfigManager()
logger = Logger()


def load_models(model_paths):
    models = []
    for model_file in tqdm(model_paths, desc="Loading models"):
        model_path = os.path.join(CONFIG.models_dir, model_file)
        model = ViTClassifier(CONFIG.model_name, CONFIG.num_classes).to(CONFIG.device)
        state_dict = torch.load(
            model_path, map_location=CONFIG.device, weights_only=True
        )
        model.load_state_dict(state_dict)
        model.eval()
        models.append(model)
    logger.info(f"\nLoaded {len(models)} models")
    return models
