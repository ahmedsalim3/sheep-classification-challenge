import torch
import torch.nn as nn
import numpy as np

from .. import CONFIG


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(
            inputs, targets, weight=self.alpha, reduction="none"
        )
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.counter = 0
        self.best_weights = None

    def __call__(self, val_score, model):
        if self.best_score is None:
            self.best_score = val_score
            self.save_checkpoint(model)
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights:
                    model.load_state_dict(self.best_weights)
                return True
        else:
            self.best_score = val_score
            self.counter = 0
            self.save_checkpoint(model)
        return False

    def save_checkpoint(self, model):
        self.best_weights = model.state_dict().copy()


def get_optimizer_scheduler(model, train_loader, epochs):
    """
    An optimizer and a learning rate scheduler
    with different learning rates for backbone
    and head
    Uses AdamW optimizer, CosineAnnealingWarmRestarts scheduler
    """
    param_groups = [
        {
            "params": model.backbone.parameters(),
            "lr": float(CONFIG.lr) * 0.1,
        },  # Backbone with lower LR
        {
            "params": model.classifier.parameters(),
            "lr": float(CONFIG.lr),
        },  # Classifier with base LR
    ]

    optimizer = torch.optim.AdamW(
        param_groups, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8
    )

    # Cosine annealing with warm restarts
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=len(train_loader) * 3,  # Restart every 3 epochs
        T_mult=2,
        eta_min=float(CONFIG.min_lr),
    )

    return optimizer, scheduler


def compute_class_weights(labels, method="balanced"):
    from sklearn.utils.class_weight import compute_class_weight

    classes = np.unique(labels)

    if method == "balanced":
        weights = compute_class_weight("balanced", classes=classes, y=labels)
    elif method == "effective":
        # Effective number of samples
        beta = 0.9999
        effective_num = 1.0 - np.power(beta, np.bincount(labels))
        weights = (1.0 - beta) / np.array(effective_num)
        weights = weights / weights.sum() * len(classes)

    return torch.tensor(weights, dtype=torch.float)
