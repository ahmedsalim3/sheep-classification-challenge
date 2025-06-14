import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
import torch
from typing import Dict, List


def train_pl(
    classifier, train_loader, val_loader, max_epochs, seed, patience, default_root_dir
) -> Dict[str, List[float]]:
    """
    Train a PyTorch Lightning model.

    Args:
        classifier: The PyTorch Lightning model to train
        train_loader: The DataLoader for the training set
        val_loader: The DataLoader for the validation set
        max_epochs: The maximum number of epochs to train
        default_root_dir: The directory to save the model and logs

    Returns:
        Dict[str, List[float]]: Dictionary containing training history metrics
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "auto")
    pl.seed_everything(seed)

    early_stop_callback = EarlyStopping(
        monitor="val_f1_macro", mode="max", patience=patience, verbose=True
    )

    trainer = pl.Trainer(
        accelerator=str(device),
        devices=1,
        precision="16-mixed",
        max_epochs=max_epochs,
        default_root_dir=default_root_dir,
        logger=False,
        callbacks=[early_stop_callback],
    )
    trainer.fit(classifier, train_loader, val_loader)

    # Get training history from the classifier
    history = {
        "train_loss": classifier.train_losses,
        "val_loss": classifier.val_losses,
        "train_acc": classifier.train_accuracies,
        "val_acc": classifier.val_accuracies,
        "val_f1_macro": classifier.val_f1_macro_scores,
        "val_f1_micro": classifier.val_f1_micro_scores,
    }

    return history
