import os

import torch
from torch.utils.data import DataLoader
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.data.transforms import get_train_transforms, get_valid_transforms
from src.data.dataset import SheepDataset
from src.modeling.classifier import ViTClassifier
from src.utils.config import ConfigManager

CONFIG = ConfigManager()


def get_label_maps():
    df = pd.read_csv(CONFIG.train_csv)
    labels = sorted(df["label"].unique())
    label2idx = {v: i for i, v in enumerate(labels)}
    idx2label = {i: v for v, i in label2idx.items()}
    return label2idx, idx2label


def denormalize(img_tensor, mean, std):
    # img_tensor shape: (C, H, W)
    img = img_tensor.cpu().numpy()
    for i in range(3):
        img[i] = img[i] * std[i] + mean[i]
    img = img.clip(0, 1)
    # transpose from (C, H, W) to (H, W, C) for imshow
    img = img.transpose(1, 2, 0)
    return img


def plot_augmented_samples(image_path, n=6):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    aug = get_train_transforms()
    _, axs = plt.subplots(1, n, figsize=(18, 6))
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    for i in range(n):
        augmented = aug(image=img)
        denorm_img = denormalize(augmented["image"], mean, std)
        axs[i].imshow(denorm_img)
        axs[i].axis("off")
    plt.tight_layout()
    plt.show()


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
    print(f"\nLoaded {len(models)} models")
    return models


def load_test_data():
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
    print(f"Test set size: {len(test_loader.dataset)}, batch size: {CONFIG.batch_size}")
    return test_loader
