import numpy as np
import torch
from tqdm import tqdm

from src.utils.config import ConfigManager
from src.utils.logger import Logger

CONFIG = ConfigManager()
logger = Logger()


def ensemble_predict(models, test_loader, fold_scores=None):
    """
    Generate ensemble predictions from a list of models
    """
    if fold_scores is not None:
        assert len(fold_scores) == len(models), (
            f"Length of model scores ({len(fold_scores)}) "
            f"must match number of models ({len(models)})."
        )

        if isinstance(fold_scores, list):
            fold_scores = np.asarray(fold_scores)
        # Normalize weights
        model_weights = fold_scores / np.sum(fold_scores)
    else:
        model_weights = None

    all_preds, all_confidences, all_filenames, all_class_probs = [], [], [], []

    for images, filenames in tqdm(test_loader, desc="Predicting"):
        images = images.to(CONFIG.device)
        batch_logits = []

        with torch.no_grad():
            for model in models:
                with torch.amp.autocast(device_type=CONFIG.device):
                    outputs = model(images)
                    probs = torch.softmax(outputs, dim=1)
                    batch_logits.append(probs.cpu().numpy())

        for i, logit in enumerate(batch_logits):
            assert (
                logit.shape == batch_logits[0].shape
            ), f"Inconsistent shape in model {i}"

        if model_weights is None:
            avg_probs = np.mean(batch_logits, axis=0)
        else:
            # Weighted soft voting with model weights
            #
            stacked_logits = np.stack(batch_logits, axis=0)  # shape: (num_models, B, C)
            avg_probs = np.average(
                stacked_logits, axis=0, weights=model_weights
            )  # shape: (B, C)

        preds = np.argmax(avg_probs, axis=1)
        confidences = np.max(avg_probs, axis=1)

        all_preds.extend(preds)
        all_confidences.extend(confidences)
        all_filenames.extend(filenames)
        all_class_probs.extend(avg_probs)

    # ===== Print some stats =====
    logger.info(f"\nTotal predictions: {len(all_preds)}")
    logger.info(f"Average confidence: {np.mean(all_confidences):.4f}")
    logger.info(f"Min confidence: {np.min(all_confidences):.4f}")
    logger.info(f"Max confidence: {np.max(all_confidences):.4f}\n")

    # ===== Print model weights and fold scores =====
    if model_weights is not None:
        for i, (w, s) in enumerate(zip(model_weights, fold_scores)):
            logger.info(
                f"Model {i+1} | Weight: {w:.3f} | Fold score: {s:.4f} | Type: {'Initial' if i < CONFIG.n_folds else 'Pseudo'}"
            )

    return all_preds, all_confidences, all_filenames, all_class_probs
