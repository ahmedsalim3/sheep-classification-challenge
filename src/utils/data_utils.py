import pandas as pd
from pathlib import Path
import shutil
import os
from .logger import Logger

logger = Logger()


def sort_images_for_imagefolder(train_dir, labels_file, train_sorted):
    """Sort images for ImageFolder dataset."""

    df = pd.read_csv(labels_file)

    train_sorted = Path(train_sorted)
    train_sorted.mkdir(parents=True, exist_ok=True)

    for _, row in df.iterrows():
        img_id = row["filename"]
        label = str(row["label"])
        src_path = Path(train_dir) / img_id
        dst_dir = train_sorted / label
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst_path = dst_dir / img_id

        if src_path.exists():
            shutil.copy2(src_path, dst_path)
        else:
            logger.warning(f"Warning: {src_path} not found.")

    logger.info(f"Images organized in: {train_sorted}")
    return train_sorted


def get_test_image_files(test_dir, extensions=(".jpg")):
    """Get sorted list of test image files."""
    test_dir = Path(test_dir)
    return sorted([f for f in os.listdir(test_dir) if f.lower().endswith(extensions)])
