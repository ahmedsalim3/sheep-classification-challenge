import os

import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.data import SheepDataset, get_train_transforms, get_valid_transforms
from src.modeling import (
    ViTClassifier,
    FocalLoss,
    compute_class_weights,
    get_optimizer_scheduler,
    EarlyStopping,
    train_one_epoch,
    evaluate,
)
from src.utils.visualization import plot_metrics

from src import CONFIG, Logger

logger = Logger()


def main():
    torch.manual_seed(CONFIG.seed)  # for reproducibility
    np.random.seed(CONFIG.seed)

    df = pd.read_csv(CONFIG.train_csv)

    label2idx = {label: i for i, label in enumerate(sorted(df["label"].unique()))}
    # idx2label = {v: k for k, v in label2idx.items()}
    df["label"] = df["label"].map(label2idx)

    class_weights = compute_class_weights(df["label"].values, method="effective").to(
        CONFIG.device
    )
    logger.info(f"Class weights: {class_weights}")

    criterion = FocalLoss(alpha=class_weights, gamma=2.0)
    skf = StratifiedKFold(
        n_splits=CONFIG.n_folds, shuffle=True, random_state=CONFIG.seed
    )

    fold_scores = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df.label)):
        logger.info(f"\n----------------------- Fold {fold+1} -----------------------")
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]

        # Print fold class distribution
        logger.info(
            f"Train distribution: {train_df['label'].value_counts().sort_index().tolist()}"
        )
        logger.info(
            f"Val distribution: {val_df['label'].value_counts().sort_index().tolist()}"
        )

        train_ds = SheepDataset(train_df, CONFIG.train_dir, get_train_transforms())
        val_ds = SheepDataset(val_df, CONFIG.train_dir, get_valid_transforms())

        train_loader = DataLoader(
            train_ds,
            batch_size=CONFIG.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=CONFIG.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        model = ViTClassifier(CONFIG.model_name, CONFIG.num_classes).to(CONFIG.device)
        optimizer, scheduler = get_optimizer_scheduler(
            model, train_loader, CONFIG.epochs
        )
        scaler = torch.amp.GradScaler(device=CONFIG.device)
        early_stopping = EarlyStopping(patience=CONFIG.patience)

        # Initialize history tracking for this fold
        history = {
            "val_loss": [],
            "val_acc": [],
            "val_f1_macro": [],
            "val_f1_weighted": [],
        }

        best_f1 = 0

        for epoch in range(CONFIG.epochs):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, optimizer, criterion, scheduler, scaler, epoch
            )

            # Save class report only at the last epoch or every 5th
            save_report = epoch == CONFIG.epochs - 1
            val_f1_macro, val_f1_weighted, val_acc, val_loss, report_txt = evaluate(
                model, val_loader, criterion, class_report=save_report
            )

            # Store metrics in history
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            history["val_f1_macro"].append(val_f1_macro)
            history["val_f1_weighted"].append(val_f1_weighted)

            logger.info(
                f"Fold {fold+1} | Epoch {epoch} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                f"Val F1 Macro: {val_f1_macro:.4f} | Val F1 Weighted: {val_f1_weighted:.4f}"
            )

            if val_f1_macro > best_f1:
                best_f1 = val_f1_macro
                torch.save(
                    model.state_dict(),
                    os.path.join(CONFIG.models_dir, f"best_model_fold{fold}.pth"),
                )
                # Save best classification report
                if report_txt:
                    report_path = os.path.join(
                        CONFIG.results_dir, f"fold_{fold}_report.txt"
                    )
                    with open(report_path, "w") as f:
                        f.write(report_txt)

            # Early stopping
            if early_stopping(val_f1_macro, model):
                logger.info(f"Early stopping at epoch {epoch}")
                break

        # Plot metrics for this fold
        plot_metrics(history, f"folde_{fold}_metrics")

        fold_scores.append(best_f1)
        logger.info(f"Fold {fold} best F1: {best_f1:.4f}")

    logger.info("\nCross-validation results:")
    logger.info(f"Mean F1: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")
    logger.info(f"Individual fold scores: {fold_scores}")


if __name__ == "__main__":
    main()
