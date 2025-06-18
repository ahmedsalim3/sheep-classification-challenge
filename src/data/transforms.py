import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from .. import CONFIG


def get_train_transforms():
    return A.Compose(
        [
            A.Resize(CONFIG.img_size, CONFIG.img_size),
            A.HorizontalFlip(p=0.6),
            A.VerticalFlip(p=0.2),
            A.Affine(
                scale=(0.9, 1.1),
                translate_percent=(0.0, 0.1),
                rotate=(-20, 20),
                shear=0,
                fit_output=False,
                mode=cv2.BORDER_CONSTANT,
                cval=(0, 0, 0),
                p=0.6,
            ),
            A.ColorJitter(
                brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1, p=0.5
            ),
            A.OneOf(
                [
                    A.MotionBlur(blur_limit=(3, 5), p=1.0),
                    A.MedianBlur(blur_limit=3, p=1.0),
                ],
                p=0.3,
            ),
            A.OneOf(
                [
                    A.GridDistortion(num_steps=5, distort_limit=0.05, p=1.0),
                    A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=1.0),
                ],
                p=0.2,
            ),
            A.CoarseDropout(
                num_holes_range=8, hole_height_range=32, hole_width_range=32, p=0.4
            ),
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
