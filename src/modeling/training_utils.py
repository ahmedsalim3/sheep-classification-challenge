import torch
import numpy as np
from sklearn.utils.class_weight import compute_class_weight


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


def compute_class_weights(labels, method="balanced"):
    labels = np.asarray(labels)
    assert np.issubdtype(labels.dtype, np.integer), "Labels must be integers."
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
