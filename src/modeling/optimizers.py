import torch

from src.utils.config import ConfigManager
from src.utils.logger import Logger

CONFIG = ConfigManager()
logger = Logger()


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
