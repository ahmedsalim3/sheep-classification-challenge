import torch
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

from .. import CONFIG


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

    head_params = []
    backbone_params = []
    for name, param in model.named_parameters():
        if "classifier" in name:
            head_params.append(param)
        else:
            backbone_params.append(param)
    param_groups = [
        {
            "params": backbone_params,
            "lr": float(CONFIG.lr) * 0.1,
        },  # Backbone with lower LR
        {
            "params": head_params,
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
    # based on the Hugging Face implementation:
    # https://github.com/huggingface/transformers/blob/v4.23.1/src/transformers/optimization.py#L104-L135
    # from transformers.optimization import get_cosine_schedule_with_warmup
    # num_training_steps = epochs * len(train_loader)
    # warmup_steps = int(0.1 * num_training_steps)  # 10% warmup
    # scheduler = get_cosine_schedule_with_warmup(
    #     optimizer=optimizer,
    #     num_warmup_steps=warmup_steps,
    #     num_training_steps=num_training_steps,
    # )

    return optimizer, scheduler


def compute_class_weights(labels, method="balanced"):
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
