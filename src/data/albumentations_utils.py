import albumentations as A
from albumentations.pytorch import ToTensorV2

from .. import CONFIG


def get_train_transforms():
    return A.Compose(
        [
            A.Resize(CONFIG.img_size, CONFIG.img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.3),
            A.Affine(
                p=0.6, translate_percent=0.15, scale=(0.85, 1.15), rotate=(-30, 30)
            ),
            A.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15, p=0.6
            ),
            A.OneOf(
                [
                    A.MotionBlur(p=1.0),
                    A.GaussianBlur(blur_limit=(3, 9), p=1.0),
                ],
                p=0.3,
            ),
            A.OneOf(
                [
                    A.GridDistortion(p=1.0),
                    A.OpticalDistortion(distort_limit=0.1, p=1.0),
                ],
                p=0.2,
            ),
            A.CoarseDropout(p=0.4),
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
