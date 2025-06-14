import torch
import numpy as np
from sklearn.metrics import f1_score
from typing import Tuple, Union


def evaluate_vit_clip_model(
    model,
    data_loader,
    device=None,
    return_predictions: bool = False,
) -> Union[float, Tuple[float, np.ndarray, np.ndarray]]:
    """
    Evaluate a hybrid ViT+CLIP model and return Macro F1 score.

    Args:
        model (torch.nn.Module): Trained hybrid model.
        data_loader (DataLoader): Validation data loader.
        device (torch.device, optional): Device to run evaluation on.
        return_predictions (bool): Whether to return predictions and true labels.

    Returns:
        Union[float, Tuple[float, np.ndarray, np.ndarray]]:
            If return_predictions is False: Macro F1 score
            If return_predictions is True: Tuple of (Macro F1 score, true labels, predictions)
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in data_loader:
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            logits = model(batch["pixel_values"], batch["clip_images"])
            preds = logits.argmax(dim=-1).cpu().numpy()
            labels = batch["labels"].cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels)

    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    if return_predictions:
        return f1, np.array(all_labels), np.array(all_preds)
    return f1
