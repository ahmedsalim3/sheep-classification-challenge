import numpy as np
from tqdm import tqdm

from src.utils.config import ConfigManager
from src.utils.logger import Logger

CONFIG = ConfigManager()
logger = Logger()


def extract_features(models, loader):
    assert len(models) > 0, "No models provided"

    logger.info("Using {} models for feature extraction".format(len(models)))
    all_features = []
    all_filenames = []
    for images, filenames in tqdm(loader, desc="Extracting features"):
        images = images.to(CONFIG.device)
        batch_feats = []
        for model in models:
            feats = model.get_features(images)
            batch_feats.append(feats.cpu().numpy())
        # Average ensemble features from all models
        batch_feats = np.stack(batch_feats).mean(axis=0)
        all_features.append(batch_feats)
        all_filenames.extend(filenames)
    all_features = np.concatenate(all_features, axis=0)
    return all_features, all_filenames
