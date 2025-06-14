from typing import Tuple, List

import torch
import torch.nn.functional as F
from transformers import ViTFeatureExtractor, ViTForImageClassification
import clip

from .vit_clip_fusion import ViTCLIPFusionClassifier


def predict_vit_clip_model(
    model, data_loader, class_mappings
) -> Tuple[List[str], List[str], List[float]]:
    """Predict using a ViT+CLIP model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    predictions = []
    confidences = []
    filenames = []

    with torch.no_grad():
        for batch in data_loader:
            batch_device = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            logits = model(batch_device["pixel_values"], batch_device["clip_images"])
            probs = F.softmax(logits, dim=-1)
            max_probs, preds = torch.max(probs, dim=-1)

            for i, filename in enumerate(batch["filenames"]):
                pred_id = str(preds[i].item())
                label = class_mappings["id_to_label"][pred_id]
                confidence = max_probs[i].item()

                predictions.append(label)
                confidences.append(confidence)
                filenames.append(filename)

    return filenames, predictions, confidences


def load_vit_clip_model(
    model_state_path, class_mappings, clip_model_name, vit_model_name
):
    feature_extractor = ViTFeatureExtractor.from_pretrained(vit_model_name)

    vit_model = ViTForImageClassification.from_pretrained(
        vit_model_name,
        num_labels=len(class_mappings["label_to_id"]),
        label2id=class_mappings["label_to_id"],
        id2label=class_mappings["id_to_label"],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_model, clip_preprocess = clip.load(clip_model_name, device=device)

    vit_clip_model = ViTCLIPFusionClassifier(
        vit_model=vit_model,
        clip_model=clip_model,
        num_classes=len(class_mappings["label_to_id"]),
    )

    vit_clip_model.load_state_dict(torch.load(model_state_path, map_location=device))

    return vit_clip_model, feature_extractor, clip_preprocess
