import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset
from PIL import Image


class TrainDataset(ImageFolder):
    @staticmethod
    def create_dataset_split(
        data_dir: str, val_split: float, seed: int
    ) -> Tuple[Subset, Subset, List[str]]:
        """Create train and validation splits from a dataset directory."""
        torch.manual_seed(seed)
        ds = ImageFolder(data_dir)
        indices = torch.randperm(len(ds)).tolist()
        n_val = math.floor(len(indices) * val_split)
        train_ds = Subset(ds, indices[:-n_val])
        val_ds = Subset(ds, indices[-n_val:])
        return train_ds, val_ds, ds.classes

    @staticmethod
    def create_label_mappings(classes: List[str], save_path: Path) -> Dict[str, Any]:
        """Create and save label mappings for the dataset."""
        label2id = {cls: str(i) for i, cls in enumerate(classes)}
        id2label = {str(i): cls for i, cls in enumerate(classes)}
        class_mappings = {
            "classes": classes,
            "label_to_id": label2id,
            "id_to_label": id2label,
        }
        with open(save_path / "class_mappings.json", "w") as f:
            json.dump(class_mappings, f)
        return class_mappings

    @staticmethod
    def create_data_loaders(
        train_ds: Subset,
        val_ds: Subset,
        collator: Any,
        batch_size: int,
        num_workers: int = 2,
    ) -> Tuple[DataLoader, DataLoader]:
        """Create DataLoader instances for training and validation."""
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            collate_fn=collator,
            num_workers=num_workers,
            shuffle=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, collate_fn=collator, num_workers=num_workers
        )
        return train_loader, val_loader


class TestDataset(Dataset):
    def __init__(self, image_dir, image_files):
        self.image_dir = Path(image_dir)
        self.image_files = image_files

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_dir / self.image_files[idx]
        image = Image.open(img_path).convert("RGB")
        return image, self.image_files[idx]
