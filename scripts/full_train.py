import os
import argparse
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd

from src.data import SheepDataset, get_valid_transforms
from src.modeling import ViTClassifier, train_cross_validation, compute_class_weights
from src.utils.pseudo_labeling import generate_pseudo_labels
from src import CONFIG, Logger

logger = Logger()


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pseudo_train", action="store_true")
    return parser.parse_args()


def load_models(model_paths):
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
    return models


def load_test_data():
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
    return test_loader


if __name__ == "__main__":
    args = arg_parse()
    pseudo_train = True  # args.pseudo_train

    INITIAL_RESULTS_DIR = os.path.join(CONFIG.results_dir, "initial_results")
    os.makedirs(INITIAL_RESULTS_DIR, exist_ok=True)
    if pseudo_train:
        PSEUDO_RESULTS_DIR = os.path.join(CONFIG.results_dir, "final_results")
        os.makedirs(PSEUDO_RESULTS_DIR, exist_ok=True)

    # ==== Load data ====
    df = pd.read_csv(CONFIG.train_csv)
    label2idx = {label: i for i, label in enumerate(sorted(df["label"].unique()))}
    idx2label = {v: k for k, v in label2idx.items()}

    df["label"] = df["label"].map(label2idx)

    class_weights = compute_class_weights(df["label"].values, method="effective").to(
        CONFIG.device
    )
    logger.info(f"Class weights: {class_weights}")

    # ==== Cross-validation training ====
    fold_scores = train_cross_validation(
        df, pseudo_train=False, results_dir=INITIAL_RESULTS_DIR
    )

    # ====== Predict on test set ======
    model_files = [f for f in os.listdir(CONFIG.models_dir) if f.endswith(".pth")]
    models = load_models(model_files)
    test_loader = load_test_data()

    # Predict in batches
    all_preds: list[int] = []
    all_confidences: list[float] = []
    all_filenames: list[str] = []
    for images, filenames in tqdm(test_loader, desc="Predicting"):
        images = images.to(CONFIG.device)
        batch_logits: list[np.ndarray] = []

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

    # ===== Save predictions =====
    preds_df = pd.DataFrame(
        {
            "filename": all_filenames,
            "label": all_labels,
        }
    )
    preds_df.to_csv(os.path.join(INITIAL_RESULTS_DIR, "predictions.csv"), index=False)

    # ===== Save predictions with confidence =====
    conf_preds_df = pd.DataFrame(
        {"filename": all_filenames, "label": all_labels, "confidence": all_confidences}
    )
    conf_preds_df.to_csv(
        os.path.join(INITIAL_RESULTS_DIR, "predictions_with_confidence.csv"),
        index=False,
    )

    # ===== Print some stats =====
    logger.info(f"Initial predictions saved to {INITIAL_RESULTS_DIR}")
    logger.info(f"Total predictions: {len(all_preds)}")
    logger.info(f"Average confidence: {np.mean(all_confidences):.4f}")
    logger.info(f"Min confidence: {np.min(all_confidences):.4f}")
    logger.info(f"Max confidence: {np.max(all_confidences):.4f}")
    # logger.info(f"Confidence distribution: {np.histogram(all_confidences, bins=20)}")

    # ===== Pseudo-labeling =====
    if pseudo_train:
        logger.info(f"\n{'-' * 60} Pseudo-labeling {'-' * 60}")
        test_loader = load_test_data()
        # ===== Generate pseudo-labels =====
        pseudo_df = generate_pseudo_labels(models, test_loader, threshold=0.1)
        print(pseudo_df.head())
        pseudo_df["label"] = pseudo_df["label"].map(idx2label)
        pseudo_df.to_csv(
            os.path.join(PSEUDO_RESULTS_DIR, "pseudo_labels.csv"), index=False
        )
        logger.info(f"Pseudo-labels saved to {PSEUDO_RESULTS_DIR}")
        logger.info(f"Number of pseudo labels: {len(pseudo_df)}")
        logger.info(f"Number of exluded images: {len(conf_preds_df) - len(pseudo_df)}")

        # ===== Prepare combined dataset =====
        logger.info("Preparing combined dataset...")

        orig_df = pd.read_csv(CONFIG.train_csv)
        logger.info("Length of original training set: {}".format(len(orig_df)))
        orig_df["confidence"] = 1.0
        orig_df["source"] = "train"

        logger.info("Length of pseudo-labels: {}".format(len(pseudo_df)))
        pseudo_df["confidence"] = pseudo_df["confidence"]
        pseudo_df["source"] = "pseudo"

        combined_df = pd.concat([orig_df, pseudo_df])
        logger.info("Length of combined dataset: {}".format(len(combined_df)))
        assert (
            combined_df.isnull().sum().sum() == 0
        ), "Error: Null values in combined dataset"
        assert isinstance(
            combined_df["label"].iloc[0], str
        ), "Error: Label is not a string"

        combined_df["label"] = combined_df["label"].map(label2idx)
        class_weights = compute_class_weights(
            combined_df["label"].values, method="effective"
        ).to(CONFIG.device)
        # logger.info(f"Class weights: {class_weights}")

        # ===== Train with combined dataset =====
        final_fold_scores = train_cross_validation(combined_df, pseudo_train=True)

        # ===== Final predictions =====

        # Load models, and compute weights
        model_files = [f for f in os.listdir(CONFIG.models_dir) if f.endswith(".pth")]
        models = load_models(model_files)
        test_loader = load_test_data()

        # Load scores
        cv_scores = np.load(os.path.join(CONFIG.models_dir, "cv_fold_scores.npy"))
        pseudo_scores = np.load(
            os.path.join(CONFIG.models_dir, "pseudo_fold_scores.npy")
        )

        # Option 1: downweight pseudo models manually (e.g. 0.5x)
        adjusted_scores = np.concatenate(
            [
                cv_scores,  # as those 5 models are clean-trained
                pseudo_scores * 0.5,  # penalize pseudo models
            ]
        )

        # Normalize weights
        model_weights = adjusted_scores / adjusted_scores.sum()

        for i, (w, s) in enumerate(zip(model_weights, adjusted_scores)):
            logger.info(
                f"Model {i+1} | Weight: {w:.3f} | Raw score: {s:.4f} | Type: {'Initial' if i < 5 else 'Pseudo'}"
            )

        # Option 2: Select top-k
        # top_k_idx = np.argsort(adjusted_scores)[-7:]
        # adjusted_scores = adjusted_scores[top_k_idx]
        # model_files = [model_files[i] for i in top_k_idx]

        # NOTE: We can also use the five pseudo-trained models,
        # with, or without adjusing scores, see below
        # Weighted soft voting Section
        # but its pretty risky, we should never trust pseudo-labels that much
        # model_weights = np.array(final_fold_scores)
        # model_weights = model_weights / model_weights.sum()  # normalize to sum=1
        logger.info(f"Model weights: {model_weights}")

        # Predict on test set
        all_preds = []
        all_confidences = []
        all_filenames = []
        all_class_probs = []
        for images, filenames in tqdm(test_loader, desc="Predicting"):
            images = images.to(CONFIG.device)
            batch_logits = []

            with torch.no_grad():
                for model in models:
                    with torch.amp.autocast(device_type=CONFIG.device):
                        outputs = model(images)
                        probs = torch.softmax(outputs, dim=1)
                        batch_logits.append(probs.cpu().numpy())

            #######################################
            # Weighted soft voting with model weights
            # here we are using 10 models, 5 initial and 5 pseudo, and setting their weights
            # to the scores we computed earlier
            stacked_logits = np.stack(batch_logits, axis=0)  # shape: (num_models, B, C)
            avg_probs = np.average(
                stacked_logits, axis=0, weights=model_weights
            )  # shape: (B, C)
            #######################################
            # avg_probs = np.mean(batch_logits, axis=0) # alternative
            #######################################
            preds = np.argmax(avg_probs, axis=1)
            confidences = np.max(avg_probs, axis=1)

            all_preds.extend(preds)
            all_confidences.extend(confidences)
            all_filenames.extend(filenames)
            all_class_probs.extend(avg_probs)

        logger.info(
            f"batch_logits shape: {stacked_logits.shape}"
        )  # (num_models, batch_size, num_classes)
        logger.info(f"model_weights: {model_weights}")

        # Convert predictions to labels (actual classes)
        all_labels = [idx2label[pred] for pred in all_preds]

        # Create dataframes for submission and confidence
        test_df = pd.DataFrame({"filename": all_filenames, "label": all_labels})
        test_conf_df = pd.DataFrame(
            {
                "filename": all_filenames,
                "label": all_labels,
                "confidence": all_confidences,
            }
        )
        assert len(test_df) == len(
            os.listdir(CONFIG.test_dir)
        ), "Mismatch in number of test files!"

        out1 = os.path.join(PSEUDO_RESULTS_DIR, "submission_soft_voting.csv")
        test_df.to_csv(out1, index=False)

        out2 = os.path.join(
            PSEUDO_RESULTS_DIR, "submission_with_confidence_soft_voting.csv"
        )
        test_conf_df.to_csv(out2, index=False)

        # Print some statistics
        logger.info(f"Total predictions: {len(all_preds)}")
        logger.info(f"Average confidence: {np.mean(all_confidences):.4f}")
        logger.info(f"Min confidence: {np.min(all_confidences):.4f}")
        logger.info(f"Max confidence: {np.max(all_confidences):.4f}")
