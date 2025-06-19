import os
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from src.modeling.classifier import ViTClassifier
from src.modeling.losses import FocalLoss
from src.modeling.optimizers import get_optimizer_scheduler
from src.modeling.training_utils import compute_class_weights, EarlyStopping
from src.data.dataset import SheepDataset, PseudoDataset
from src.data.transforms import get_train_transforms, get_valid_transforms
from src.utils.visualization import plot_metrics
from src.utils.config import ConfigManager
from src.utils.logger import Logger

CONFIG = ConfigManager()
logger = Logger()


def train_one_epoch(model, loader, optimizer, criterion, scheduler, scaler, epoch):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch+1}")
    for batch in pbar:
        if len(batch) == 3:
            images, labels, confidences = batch
            confidences = confidences.to(CONFIG.device)
        else:
            images, labels = batch
            confidences = None

        images = images.to(CONFIG.device)
        labels = labels.to(CONFIG.device)

        optimizer.zero_grad()

        with torch.amp.autocast(device_type=CONFIG.device):
            outputs = model(images)
            loss = criterion(outputs, labels, weights=confidences)

        scaler.scale(loss).backward()

        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()
        # _, predicted = torch.max(outputs.data, 1)
        predicted = outputs.argmax(dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        # Update progress bar
        pbar.set_postfix(
            {"loss": f"{loss.item():.4f}", "acc": f"{100.*correct/total:.2f}%"}
        )

    return total_loss / len(loader), correct / total


def evaluate(model, loader, criterion=None):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating"):
            if len(batch) == 3:
                images, labels, confidences = batch
                confidences = confidences.to(CONFIG.device)
            else:
                images, labels = batch
                confidences = None

            images = images.to(CONFIG.device)
            labels = labels.to(CONFIG.device)

            with torch.amp.autocast(device_type=CONFIG.device):
                outputs = model(images)
                if criterion is not None:
                    if confidences is not None:
                        loss = criterion(outputs, labels, weights=confidences)
                    else:
                        loss = criterion(outputs, labels)
                    total_loss += loss.item()

            preds = torch.argmax(outputs, 1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    f1_macro = f1_score(all_labels, all_preds, average="macro")
    f1_weighted = f1_score(all_labels, all_preds, average="weighted")

    accuracy = correct / total if total > 0 else 0
    avg_loss = total_loss / len(loader) if criterion is not None else 0

    return {
        "metrics": {
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "accuracy": accuracy,
            "avg_loss": avg_loss,
        },
        "predictions": {
            "all_preds": all_preds,
            "all_labels": all_labels,
        },
    }


def train_cross_validation(df, pseudo_train=False, results_dir=None):
    if results_dir is None:
        results_dir = CONFIG.results_dir

    class_weights = compute_class_weights(df["label"].values, method="effective").to(
        CONFIG.device
    )
    # logger.info(f"Class weights: {class_weights}")

    criterion = FocalLoss(alpha=class_weights, gamma=2.0)
    skf = StratifiedKFold(
        n_splits=CONFIG.n_folds, shuffle=True, random_state=CONFIG.seed
    )

    fold_scores = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df.label)):
        logger.info(f"\n{'=' * 70} FOLD {fold+1} {'=' * 70}")
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]

        # --- Create fold-specific directories ---
        if pseudo_train:
            fold_results_dir = os.path.join(results_dir, f"fold_{fold+1}")
            model_name = f"pseudo_fold_{fold+1}.pth"
        else:
            fold_results_dir = os.path.join(results_dir, f"fold_{fold+1}")
            model_name = f"cv_fold_{fold+1}.pth"
        os.makedirs(fold_results_dir, exist_ok=True)
        # ----------------------------------------

        # Print fold class distribution
        logger.info(
            f"Train distribution: {train_df['label'].value_counts().sort_index().tolist()}, Length: {len(train_df)}"
        )
        logger.info(
            f"Val distribution: {val_df['label'].value_counts().sort_index().tolist()}, Length: {len(val_df)}"
        )
        if pseudo_train:
            train_ds = PseudoDataset(
                train_df, CONFIG.train_dir, CONFIG.test_dir, get_train_transforms()
            )
            val_ds = PseudoDataset(
                val_df, CONFIG.train_dir, CONFIG.test_dir, get_valid_transforms()
            )
        else:
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
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "val_f1_macro": [],
            "val_f1_weighted": [],
        }

        best_f1 = 0
        class_report, cm = "", ""

        for epoch in range(CONFIG.epochs):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, optimizer, criterion, scheduler, scaler, epoch
            )

            eval_results = evaluate(model, val_loader, criterion)

            val_f1_macro = eval_results["metrics"]["f1_macro"]
            val_f1_weighted = eval_results["metrics"]["f1_weighted"]
            val_acc = eval_results["metrics"]["accuracy"]
            val_loss = eval_results["metrics"]["avg_loss"]
            all_labels = eval_results["predictions"]["all_labels"]
            all_preds = eval_results["predictions"]["all_preds"]

            # Store metrics in history
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            history["val_f1_macro"].append(val_f1_macro)
            history["val_f1_weighted"].append(val_f1_weighted)

            logger.info(
                f"Fold {fold+1} | Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                f"Val F1 Macro: {val_f1_macro:.4f} | Val F1 Weighted: {val_f1_weighted:.4f}"
            )

            if val_f1_macro > best_f1:
                best_f1 = val_f1_macro
                torch.save(
                    model.state_dict(),
                    os.path.join(CONFIG.models_dir, model_name),
                )
                # Generate and store the classification report only when a new best model is found
                class_report = classification_report(all_labels, all_preds, digits=4)
                logger.info(
                    f"New best F1-macro for Fold {fold+1} at epoch {epoch+1}. Model saved."
                    f" Current Best F1-macro: {best_f1:.4f}"
                )
                # Generate confusion matrix
                cm = confusion_matrix(all_labels, all_preds)
                cm = str(cm)  # Convert numpy array to string for saving

            if early_stopping(val_f1_macro, model):
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

        # --- Save best classification report for this fold ---
        if class_report:
            report_path = os.path.join(fold_results_dir, f"fold_{fold+1}_report.txt")
            with open(report_path, "w") as f:
                f.write(class_report)
            logger.info("\n----- Classification Report -----")
            logger.info(class_report)

        # --- Save best confusion matrix for this fold ---
        if cm:
            cm_path = os.path.join(
                fold_results_dir, f"fold_{fold+1}_confusion_matrix.txt"
            )
            with open(cm_path, "w") as f:
                f.write(cm)
            logger.info("\n----- Confusion Matrix -----")
            logger.info(cm)

        # --- Plot metrics ---
        plot_metrics(
            history, os.path.join(fold_results_dir, f"fold_{fold+1}_metrics.png")
        )

        # --- Save history ---
        pd.DataFrame(history).to_csv(
            os.path.join(fold_results_dir, f"history_fold_{fold+1}.csv"), index=False
        )

        fold_scores.append(best_f1)
        logger.info(f"\nFold {fold+1} best F1: {best_f1:.4f}")

    logger.info("\nCross-validation results:")
    logger.info(f"Mean F1: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")
    logger.info(f"Individual fold scores: {fold_scores}")

    if pseudo_train:
        fold_score_path = os.path.join(CONFIG.models_dir, "pseudo_fold_scores.npy")
    else:
        fold_score_path = os.path.join(CONFIG.models_dir, "cv_fold_scores.npy")
    np.save(fold_score_path, np.array(fold_scores))

    return fold_scores


def predict_cross_validation(model_paths):
    df = pd.read_csv(CONFIG.train_csv)
    label2idx = {label: i for i, label in enumerate(sorted(df["label"].unique()))}
    idx2label = {v: k for k, v in label2idx.items()}

    test_files = sorted(
        [f for f in os.listdir(CONFIG.test_dir) if f.lower().endswith(".jpg")]
    )
    test_ds = SheepDataset(
        image_dir=CONFIG.test_dir, transform=get_valid_transforms(), is_test=True
    )
    test_ds.img_files = test_files

    test_loader = DataLoader(
        test_ds,
        batch_size=CONFIG.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    all_preds = []
    all_confidences = []
    all_filenames = []

    models = []
    for model_file in model_paths:
        model_path = os.path.join(CONFIG.models_dir, model_file)
        model = ViTClassifier(CONFIG.model_name, CONFIG.num_classes).to(CONFIG.device)
        state_dict = torch.load(
            model_path, map_location=CONFIG.device, weights_only=True
        )
        model.load_state_dict(state_dict)
        model.eval()
        models.append(model)

    # Predict in batches
    for images, filenames in tqdm(test_loader, desc="Predicting"):
        images = images.to(CONFIG.device)
        batch_logits = []

        with torch.no_grad():
            for model in models:
                with torch.amp.autocast(device_type=CONFIG.device):
                    outputs = model(images)
                    probs = torch.softmax(outputs, dim=1)
                    batch_logits.append(probs.cpu().numpy())

        avg_probs = np.mean(batch_logits, axis=0)

        preds = np.argmax(avg_probs, axis=1)
        confidences = np.max(avg_probs, axis=1)

        all_preds.extend(preds)
        all_confidences.extend(confidences)
        all_filenames.extend(filenames)

    all_labels = [idx2label[pred] for pred in all_preds]

    df1 = pd.DataFrame({"filename": all_filenames, "label": all_labels})
    df2 = pd.DataFrame(
        {"filename": all_filenames, "label": all_labels, "confidence": all_confidences}
    )

    os.makedirs(CONFIG.results_dir, exist_ok=True)
    out1 = os.path.join(CONFIG.results_dir, "submission.csv")
    df1.to_csv(out1, index=False)

    out2 = os.path.join(CONFIG.results_dir, "submission_with_confidence.csv")
    df2.to_csv(out2, index=False)

    # Print some statistics
    logger.info(f"Total predictions: {len(all_preds)}")
    logger.info(f"Average confidence: {np.mean(all_confidences):.4f}")
    logger.info(f"Min confidence: {np.min(all_confidences):.4f}")
    logger.info(f"Max confidence: {np.max(all_confidences):.4f}")

    return df1, df2


def train_normal(
    train_df, val_df, pseudo_train=False, results_dir=None, model_name="best_model.pth"
):
    """Train model normally with train/validation split (no cross-validation)."""
    if results_dir is None:
        results_dir = CONFIG.results_dir

    class_weights = compute_class_weights(
        train_df["label"].values, method="effective"
    ).to(CONFIG.device)
    logger.info(f"Class weights: {class_weights}")

    criterion = FocalLoss(alpha=class_weights, gamma=2.0)

    logger.info(
        f"Train distribution: {train_df['label'].value_counts().sort_index().tolist()}, Length: {len(train_df)}"
    )
    logger.info(
        f"Val distribution: {val_df['label'].value_counts().sort_index().tolist()}, Length: {len(val_df)}"
    )

    if pseudo_train:
        train_ds = PseudoDataset(
            train_df, CONFIG.train_dir, CONFIG.test_dir, get_train_transforms()
        )
        val_ds = PseudoDataset(
            val_df, CONFIG.train_dir, CONFIG.test_dir, get_valid_transforms()
        )
    else:
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
    optimizer, scheduler = get_optimizer_scheduler(model, train_loader, CONFIG.epochs)
    scaler = torch.amp.GradScaler(device=CONFIG.device)
    early_stopping = EarlyStopping(patience=CONFIG.patience)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1_macro": [],
        "val_f1_weighted": [],
    }

    best_f1 = 0
    best_epoch = 0
    class_report, cm = "", ""

    logger.info(f"\n{'=' * 70} TRAINING START {'=' * 70}")

    for epoch in range(CONFIG.epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, scheduler, scaler, epoch
        )

        eval_results = evaluate(model, val_loader, criterion)

        val_f1_macro = eval_results["metrics"]["f1_macro"]
        val_f1_weighted = eval_results["metrics"]["f1_weighted"]
        val_acc = eval_results["metrics"]["accuracy"]
        val_loss = eval_results["metrics"]["avg_loss"]
        all_labels = eval_results["predictions"]["all_labels"]
        all_preds = eval_results["predictions"]["all_preds"]

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1_macro"].append(val_f1_macro)
        history["val_f1_weighted"].append(val_f1_weighted)

        logger.info(
            f"Epoch {epoch+1}/{CONFIG.epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
            f"Val F1 Macro: {val_f1_macro:.4f} | Val F1 Weighted: {val_f1_weighted:.4f}"
        )

        if val_f1_macro > best_f1:
            best_f1 = val_f1_macro
            best_epoch = epoch + 1

            best_f1 = val_f1_macro
            torch.save(
                model.state_dict(),
                os.path.join(CONFIG.models_dir, model_name),
            )

            class_report = classification_report(all_labels, all_preds, digits=4)
            cm = confusion_matrix(all_labels, all_preds)
            cm_str = str(cm)

            logger.info(
                f"New best F1-macro at epoch {epoch+1}: {best_f1:.4f} - Model saved!"
            )

        if early_stopping(val_f1_macro, model):
            logger.info(f"Early stopping at epoch {epoch+1}")
            break

    logger.info(f"\n{'=' * 70} TRAINING COMPLETE {'=' * 70}")
    logger.info(f"Best F1-macro: {best_f1:.4f} achieved at epoch {best_epoch}")

    if class_report:
        report_path = os.path.join(results_dir, "classification_report.txt")
        with open(report_path, "w") as f:
            f.write(class_report)
        logger.info("\n----- Final Classification Report -----")
        logger.info(class_report)

    if cm_str:
        cm_path = os.path.join(results_dir, "confusion_matrix.txt")
        with open(cm_path, "w") as f:
            f.write(cm_str)
        logger.info("\n----- Final Confusion Matrix -----")
        logger.info(cm_str)

    plot_metrics(history, os.path.join(results_dir, "training_metrics.png"))

    pd.DataFrame(history).to_csv(
        os.path.join(results_dir, "training_history.csv"), index=False
    )

    return {
        "best_f1": best_f1,
        "best_epoch": best_epoch,
        "history": history,
        "final_model_path": os.path.join(CONFIG.models_dir, model_name),
    }


def split_train_val_and_train(df, test_size=0.2, random_state=CONFIG.seed):
    train_df, val_df = train_test_split(
        df, test_size=test_size, stratify=df["label"], random_state=random_state
    )

    logger.info(f"Training set size: {len(train_df)}")
    logger.info(f"Validation set size: {len(val_df)}")

    # Train the model
    results = train_normal(train_df, val_df)

    return results, train_df, val_df
