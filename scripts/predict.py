import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader

from src.data import SheepDataset, get_valid_transforms
from src.modeling import ViTClassifier
from src import CONFIG, Logger

logger = Logger()


def main(model_paths):
    df = pd.read_csv(CONFIG.train_csv)
    label2idx = {label: i for i, label in enumerate(sorted(df["label"].unique()))}
    idx2label = {v: k for k, v in label2idx.items()}

    test_files = sorted(
        [f for f in os.listdir(CONFIG.test_dir) if f.lower().endswith(".jpg")]
    )
    test_ds = SheepDataset(
        image_dir=CONFIG.test_dir, transform=get_valid_transforms(), is_test=True
    )
    test_ds.img_files = test_files

    test_loader = DataLoader(
        test_ds,
        batch_size=CONFIG.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    all_preds = []
    all_confidences = []
    all_filenames = []

    models = []
    for model_file in model_paths:
        model_path = os.path.join(CONFIG.models_dir, model_file)
        model = ViTClassifier(CONFIG.model_name, CONFIG.num_classes).to(CONFIG.device)
        state_dict = torch.load(
            model_path, map_location=CONFIG.device, weights_only=True
        )
        model.load_state_dict(state_dict)
        model.eval()
        models.append(model)

    # Predict in batches
    for images, filenames in tqdm(test_loader, desc="Predicting"):
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

        all_preds.extend(preds)
        all_confidences.extend(confidences)
        all_filenames.extend(filenames)

    all_labels = [idx2label[pred] for pred in all_preds]

    df1 = pd.DataFrame({"filename": all_filenames, "label": all_labels})
    df2 = pd.DataFrame(
        {"filename": all_filenames, "label": all_labels, "confidence": all_confidences}
    )

    os.makedirs(CONFIG.results_dir, exist_ok=True)
    out1 = os.path.join(CONFIG.results_dir, "submission.csv")
    df1.to_csv(out1, index=False)

    out2 = os.path.join(CONFIG.results_dir, "submission_with_confidence.csv")
    df2.to_csv(out2, index=False)

    # Print some statistics
    logger.info(f"Total predictions: {len(all_preds)}")
    logger.info(f"Average confidence: {np.mean(all_confidences):.4f}")
    logger.info(f"Min confidence: {np.min(all_confidences):.4f}")
    logger.info(f"Max confidence: {np.max(all_confidences):.4f}")

    return df1, df2


if __name__ == "__main__":
    model_files = [f for f in os.listdir(CONFIG.models_dir) if f.endswith(".pth")]
    df, df_conf = main(model_files)
