import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

from src.utils.config import ConfigManager

CONFIG = ConfigManager()


def get_train_transforms():
    return A.Compose(
        [
            A.Resize(CONFIG.img_size, CONFIG.img_size),
            # Geometric augmentations
            A.HorizontalFlip(p=0.5),
            A.Affine(
                scale=(0.9, 1.1),
                translate_percent=(0.0, 0.1),
                rotate=(-15, 15),
                shear=5,
                border_mode=cv2.BORDER_CONSTANT,
                fill=(0, 0, 0),
                p=0.7,
            ),
            # Color & contrast adjustments
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.2),
            A.HueSaturationValue(
                hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.3
            ),
            # Blur / distortion
            A.OneOf(
                [
                    A.MotionBlur(blur_limit=3, p=1.0),
                    A.MedianBlur(blur_limit=3, p=1.0),
                ],
                p=0.2,
            ),
            A.GridDistortion(num_steps=5, distort_limit=0.03, p=0.1),
            A.ElasticTransform(alpha=1, sigma=50, p=0.1),
            # Weather simulation
            A.RandomFog(
                fog_coef_range=(0.1, 0.3),  # tuple of min and max fog intensity
                alpha_coef=0.08,
                p=0.2,
            ),
            A.RandomRain(blur_value=3, brightness_coefficient=0.9, p=0.1),
            # Occlusion
            A.CoarseDropout(
                num_holes_range=(3, 6),
                hole_height_range=(10, 32),
                hole_width_range=(10, 32),
                fill=0,
                p=0.3,
            ),
            # Normalize and tensor
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )


def get_valid_transforms():
    return A.Compose(
        [
            A.Resize(CONFIG.img_size, CONFIG.img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )
