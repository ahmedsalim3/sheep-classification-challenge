import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

from src.utils.config import ConfigManager
from src.utils.logger import Logger

CONFIG = ConfigManager()
logger = Logger()


def generate_pseudo_labels(models, test_loader, threshold):
    assert len(models) > 0, "No models provided"
    assert len(test_loader) > 0, "No test loader provided"

    pseudo_data = []
    for images, filenames in tqdm(
        test_loader, desc=f"Generating pseudo labels ≥ {threshold:.2f} confidence"
    ):
        images = images.to(CONFIG.device)
        batch_logits = []

        with torch.no_grad():
            for model in models:
                with torch.amp.autocast(device_type=CONFIG.device):
                    outputs = model(images)
                    probs = torch.softmax(outputs, dim=1)
                    batch_logits.append(probs.cpu().numpy())

        avg_probs = np.mean(batch_logits, axis=0)
        preds = np.argmax(avg_probs, axis=1)
        confidences = np.max(avg_probs, axis=1)

        for fname, pred, conf in zip(filenames, preds, confidences):
            if conf >= threshold:
                pseudo_data.append(
                    {"filename": fname, "label": pred, "confidence": conf}
                )

    pseudo_df = pd.DataFrame(pseudo_data)
    pseudo_df["source"] = (
        "pseudo"  # adding this column to know the source of the pseudo labels
    )
    logger.info(
        f"Generated {len(pseudo_df)} pseudo-labels out of {len(test_loader.dataset)} test images"
    )
    logger.info(
        f"Excluded {len(test_loader.dataset) - len(pseudo_df)} low-confidence predictions"
    )
    return pseudo_df


def load_pseudo_labels(pseudo_df):
    # create a map filename -> (label, confidence)
    label_map = dict(zip(pseudo_df["filename"], pseudo_df["label"]))
    conf_map = dict(zip(pseudo_df["filename"], pseudo_df["confidence"]))
    return label_map, conf_map
