import torch


class TrainCollator:
    """
    Collator for training with ViT + CLIP models.
    Processes (image, label) pairs by applying both ViT and CLIP preprocessing.
    Returns pixel values for ViT, processed images for CLIP, and corresponding labels.
    """

    def __init__(self, feature_extractor, clip_preprocess):
        self.feature_extractor = feature_extractor
        self.clip_preprocess = clip_preprocess

    def __call__(self, batch):
        # Process images for ViT
        vit_encodings = self.feature_extractor(
            [img for img, _ in batch], return_tensors="pt"
        )

        # Process images for CLIP
        clip_images = torch.stack([self.clip_preprocess(img) for img, _ in batch])

        # Combine encodings
        encodings = {
            "pixel_values": vit_encodings["pixel_values"],
            "clip_images": clip_images,
            "labels": torch.tensor([x[1] for x in batch], dtype=torch.long),
        }
        return encodings


class TestCollator:
    """
    Collator for inference/testing with ViT + optionally CLIP.
    Processes (image, filename) pairs and returns ViT-ready images and filenames.
    Does not return labels.
    """

    def __init__(self, feature_extractor, clip_preprocess):
        self.feature_extractor = feature_extractor
        self.clip_preprocess = clip_preprocess

    def __call__(self, batch):
        images, filenames = zip(*batch)

        vit_encodings = self.feature_extractor(list(images), return_tensors="pt")

        result = {"pixel_values": vit_encodings["pixel_values"], "filenames": filenames}

        clip_images = torch.stack([self.clip_preprocess(img) for img in images])
        result["clip_images"] = clip_images

        return result
