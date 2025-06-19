import os

import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image


class SheepDataset(Dataset):
    def __init__(self, df=None, image_dir=None, transform=None, is_test=False):
        self.image_dir = image_dir
        self.transform = transform
        self.is_test = is_test

        if self.is_test:
            # For test set, just get sorted list of image files
            self.img_files = sorted(os.listdir(image_dir))
        else:
            # For train/val set, use dataframe with filenames and labels
            self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.img_files) if self.is_test else len(self.df)

    def __getitem__(self, idx):
        if self.is_test:
            filename = self.img_files[idx]
            img_path = os.path.join(self.image_dir, filename)
            image = Image.open(img_path).convert("RGB")
            image = np.array(image)
            if self.transform:
                image = self.transform(image=image)["image"]
            return image, filename
        else:
            row = self.df.iloc[idx]
            img_path = os.path.join(self.image_dir, row["filename"])
            image = Image.open(img_path).convert("RGB")
            image = np.array(image)
            if self.transform:
                image = self.transform(image=image)["image"]
            label = row["label"]
            return image, torch.tensor(label, dtype=torch.long)


class PseudoDataset(Dataset):
    def __init__(self, df, train_dir, test_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filename = row["filename"]
        label = row["label"]
        source = row.get("source", "train")
        confidence = row.get(
            "confidence", 1.0
        )  # default to 1.0 for clean training data

        img_dir = (
            self.test_dir
            if source == "pseudo" or source == "cluster"
            else self.train_dir
        )
        img_path = os.path.join(img_dir, filename)

        image = Image.open(img_path).convert("RGB")
        image = np.array(image)

        if self.transform:
            image = self.transform(image=image)["image"]

        return (
            image,
            torch.tensor(label, dtype=torch.long),
            torch.tensor(confidence, dtype=torch.float),
        )
