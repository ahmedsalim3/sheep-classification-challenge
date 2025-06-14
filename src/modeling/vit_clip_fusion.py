from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy, F1Score
import clip


class ViTCLIPFusionClassifier(pl.LightningModule):
    """
    A hybrid image classifier that combines features from a Vision Transformer (ViT)
    and CLIP.

    Supports different fusion methods ('concat' or 'attention') to combine the features.
    """

    def __init__(
        self,
        vit_model,
        clip_model,
        num_classes,
        fusion_method="concat",
        lr: float = 2e-5,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters("lr", *list(kwargs))
        self.vit_model = vit_model
        self.clip_model = clip_model

        # Freeze the CLIP model
        # to avoid updating its weights during training
        for param in self.clip_model.parameters():
            param.requires_grad = False

        # Dimensions of extracted features from both models
        self.vit_dim = vit_model.config.hidden_size  # e.g., 768 for ViT-Base
        self.clip_dim = (
            512 if vit_model.config.hidden_size == 768 else 768
        )  # CLIP image embedding size (e.g., ViT-B/32)

        # Fusion layer
        self.fusion_method = fusion_method
        if fusion_method == "concat":
            fusion_dim = self.vit_dim + self.clip_dim
        elif fusion_method == "attention":
            fusion_dim = self.vit_dim
            self.attention = nn.MultiheadAttention(
                embed_dim=fusion_dim, num_heads=8, batch_first=True
            )
            self.clip_proj = nn.Linear(self.clip_dim, self.vit_dim)
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes),
        )

        # Validation metrics
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_f1_macro = F1Score(
            task="multiclass", num_classes=num_classes, average="macro"
        )
        self.val_f1_micro = F1Score(
            task="multiclass", num_classes=num_classes, average="micro"
        )

        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.train_accuracies: List[float] = []
        self.val_accuracies: List[float] = []
        self.val_f1_macro_scores: List[float] = []
        self.val_f1_micro_scores: List[float] = []

    def forward(self, pixel_values, clip_images):
        # Extract ViT features
        vit_outputs = self.vit_model(
            pixel_values=pixel_values, output_hidden_states=True
        )

        # Use the [CLS] token from the last hidden layer as a summary representation of the image
        # Shape: (batch_size, hidden_dim)
        vit_features = vit_outputs.hidden_states[-1][:, 0]

        # Extract CLIP features (frozen, inference mode only)
        with torch.no_grad():
            clip_features = self.clip_model.encode_image(clip_images)
            clip_features = F.normalize(clip_features, dim=-1)

        clip_features = clip_features.to(vit_features.dtype)

        # Fuse ViT and CLIP features
        if self.fusion_method == "concat":
            fused_features = torch.cat([vit_features, clip_features], dim=1)
        elif self.fusion_method == "attention":
            clip_features = self.clip_proj(clip_features)  # [batch_size, 768]
            stacked = torch.stack(
                [vit_features, clip_features], dim=1
            )  # [batch_size, 2, 768]
            attended, _ = self.attention(stacked, stacked, stacked)  # Self-attention
            fused_features = attended.mean(dim=1)  # Aggregate

        # Classification
        logits = self.classifier(fused_features)
        return logits

    def training_step(self, batch, batch_idx):
        logits = self(batch["pixel_values"], batch["clip_images"])
        loss = F.cross_entropy(logits, batch["labels"])
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch["pixel_values"], batch["clip_images"])
        loss = F.cross_entropy(logits, batch["labels"])
        self.log("val_loss", loss)

        # Calculate metrics
        acc = self.val_acc(logits.argmax(1), batch["labels"])
        f1_macro = self.val_f1_macro(logits.argmax(1), batch["labels"])
        f1_micro = self.val_f1_micro(logits.argmax(1), batch["labels"])

        # Log metrics
        self.log("val_acc", acc, prog_bar=True)
        self.log("val_f1_macro", f1_macro)
        self.log("val_f1_micro", f1_micro)

        return loss

    def on_train_epoch_end(self):
        # Log training metrics
        self.train_losses.append(
            self.trainer.callback_metrics.get("train_loss", torch.tensor(0.0)).item()
        )
        self.train_accuracies.append(
            self.trainer.callback_metrics.get("train_acc", torch.tensor(0.0)).item()
        )

    def on_validation_epoch_end(self):
        # Log validation metrics
        self.val_losses.append(
            self.trainer.callback_metrics.get("val_loss", torch.tensor(0.0)).item()
        )
        self.val_accuracies.append(
            self.trainer.callback_metrics.get("val_acc", torch.tensor(0.0)).item()
        )
        self.val_f1_macro_scores.append(
            self.trainer.callback_metrics.get("val_f1_macro", torch.tensor(0.0)).item()
        )
        self.val_f1_micro_scores.append(
            self.trainer.callback_metrics.get("val_f1_micro", torch.tensor(0.0)).item()
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


def build_vit_clip_classifier(
    vit_model, num_classes, fusion_method="concat", clip_model_name="ViT-B/32"
):
    """Build a hybrid Vision Transformer + CLIP classifier."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_model, clip_preprocess = clip.load(clip_model_name, device=device)
    vit_model = vit_model.to(device)

    for param in vit_model.parameters():
        param.requires_grad = True

    model = ViTCLIPFusionClassifier(
        vit_model=vit_model,
        clip_model=clip_model,
        num_classes=num_classes,
        fusion_method=fusion_method,
    )

    return model, clip_preprocess
