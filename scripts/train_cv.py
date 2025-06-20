import os
import argparse

import numpy as np
import pandas as pd

from src.utils.helpers import get_label_maps, load_models, load_test_data
from src.modeling.train_eval import train_cross_validation, compute_class_weights
from src.modeling.ensemble import ensemble_predict
from src.modeling.clustering import KMeansClustering
from src.data.pseudo_labeling import generate_pseudo_labels
from src import CONFIG, Logger

logger = Logger()


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pseudo_train", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parse()
    pseudo_train = True  # args.pseudo_train

    INITIAL_RESULTS_DIR = os.path.join(CONFIG.results_dir, "initial_results")
    os.makedirs(INITIAL_RESULTS_DIR, exist_ok=True)
    if pseudo_train:
        PSEUDO_RESULTS_DIR = os.path.join(CONFIG.results_dir, "final_results")
        os.makedirs(PSEUDO_RESULTS_DIR, exist_ok=True)

    # ==== Load data ====
    train_df = pd.read_csv(CONFIG.train_csv)
    label2idx, idx2label = get_label_maps()

    train_df["label"] = train_df["label"].map(label2idx)

    # ==== Cross-validation training ====

    logger.info(f"\n{'-' * 60} Initial Cross-Validation Training {'-' * 60}")

    fold_scores = train_cross_validation(
        train_df, pseudo_train=False, results_dir=INITIAL_RESULTS_DIR
    )

    # ====== Predict on test set ======
    # Load the 5 models we just trained
    model_files = [f for f in os.listdir(CONFIG.models_dir) if f.endswith(".pth")]
    sorted_model_files = sorted([f for f in model_files if f.startswith("cv_fold_")])

    models = load_models(sorted_model_files)

    # Load the test set
    test_loader = load_test_data()

    fold_scores = np.load(os.path.join(CONFIG.models_dir, "cv_fold_scores.npy"))
    all_preds, all_confidences, all_filenames, all_probs = ensemble_predict(
        models, test_loader, fold_scores=fold_scores
    )

    # You can also treat all models equally, without using cross-validation scores
    # all_preds, all_confidences, all_filenames, all_probs = ensemble_predict(models, test_loader)

    # Convert labels back to original labels
    all_labels = [idx2label[pred] for pred in all_preds]

    # ===== Save predictions =====
    preds_df = pd.DataFrame(
        {
            "filename": all_filenames,
            "label": all_labels,
        }
    )
    preds_df.to_csv(
        os.path.join(CONFIG.processed_data_dir, "initial_submission.csv"), index=False
    )

    # ===== Save predictions with confidence =====
    conf_preds_df = pd.DataFrame(
        {"filename": all_filenames, "label": all_labels, "confidence": all_confidences}
    )
    conf_preds_df.to_csv(
        os.path.join(
            CONFIG.processed_data_dir, "initial_submission_with_confidence.csv"
        ),
        index=False,
    )

    # ===== Pseudo-labeling =====
    if bool(CONFIG.pseudo_train):
        logger.info(f"\n{'-' * 60} Pseudo-labeling {'-' * 60}")
        test_loader = load_test_data()

        # ===== Generate pseudo-labels =====

        # Generate pseudo-labels with high threshold
        pseudo_df = generate_pseudo_labels(
            models, test_loader, threshold=float(CONFIG.pseudo_threshold)
        )
        assert (
            pseudo_df.isnull().sum().sum() == 0
        ), "Error: Null values in pseudo-labels, Try a lower threshold"

        # convert index to categorcal labels (to be matched with train_df when we combine)
        pseudo_df["label"] = pseudo_df["label"].map(idx2label)

        # Saving it
        pseudo_df.to_csv(
            os.path.join(CONFIG.processed_data_dir, "pseudo_labels.csv"), index=False
        )
        # Analyze pseudo-label distribution
        logger.info("\nPseudo-label class distribution:")
        logger.info(pseudo_df["label"].value_counts().sort_index())

        if bool(CONFIG.cluster_train):
            # ===== Cluster training =====
            logger.info("Clustering training...")
            # Load the pseudo-labels
            pseudo_df = pd.read_csv(
                os.path.join(CONFIG.processed_data_dir, "pseudo_labels.csv")
            )

            pseudo_df = pd.read_csv(
                os.path.join(CONFIG.processed_data_dir, "pseudo_labels.csv")
            )
            train_df = pd.read_csv(CONFIG.train_csv)
            output_dir = CONFIG.processed_data_dir

            clusterer = KMeansClustering(
                pseudo_df=pseudo_df,
                train_df=train_df,
                output_dir=output_dir,
                purity_threshold=float(CONFIG.purity_threshold),
            )
            df_clusters, final_df = clusterer.run()

        else:
            # ===== Prepare combined dataset =====
            logger.info("Preparing combined dataset...")
            orig_df = pd.read_csv(CONFIG.train_csv)
            logger.info("Length of original training set: {}".format(len(orig_df)))
            orig_df["confidence"] = 1.0
            orig_df["source"] = "train"

            logger.info("Length of pseudo-labels: {}".format(len(pseudo_df)))
            pseudo_df["confidence"] = pseudo_df["confidence"]
            pseudo_df["source"] = "pseudo"

            final_df = pd.concat([orig_df, pseudo_df])
            logger.info("Length of combined dataset: {}".format(len(final_df)))
            assert (
                final_df.isnull().sum().sum() == 0
            ), "Error: Null values in combined dataset"
            assert isinstance(
                final_df["label"].iloc[0], str
            ), "Error: Label is not a string"

            final_df["label"] = final_df["label"].map(label2idx)
            class_weights = compute_class_weights(
                final_df["label"].values, method="effective"
            ).to(CONFIG.device)
            # logger.info(f"Class weights: {class_weights}")

        # ===== Train with combined dataset =====

        label2idx, idx2label = get_label_maps()
        final_df["label"] = final_df["label"].map(label2idx)
        final_fold_scores = train_cross_validation(final_df, pseudo_train=True)

        # ===== Final predictions =====
        all_model_files = [
            f for f in os.listdir(CONFIG.models_dir) if f.endswith(".pth")
        ]

        cv_model_files = sorted(
            [f for f in all_model_files if f.startswith("cv_fold_")]
        )
        pseudo_model_files = sorted(
            [f for f in all_model_files if f.startswith("pseudo_fold_")]
        )
        sorted_model_files = cv_model_files + pseudo_model_files

        models = load_models(sorted_model_files)

        # Load the test set
        test_loader = load_test_data()

        # Load CV scores for weighting
        cv_scores = np.load(os.path.join(CONFIG.models_dir, "cv_fold_scores.npy"))
        pseudo_scores = np.load(
            os.path.join(CONFIG.models_dir, "pseudo_fold_scores.npy")
        )
        logger.info(f"\nCV Scores (Clean): {cv_scores}")
        logger.info(f"CV Scores (Pseudo-labeled + Clustered): {pseudo_scores}")

        # Option 1: downweight pseudo models manually (e.g. 0.5x)
        # Concatenate the scores
        scores = np.concatenate(
            [
                cv_scores,  # Initial 5 models (clean training)
                pseudo_scores,  # * 0.9,  # Pseudo 5 models (without penalty)
            ]
        )
        all_preds, all_confidences, all_filenames, all_probs = ensemble_predict(
            models, test_loader, fold_scores=scores
        )

        # Option 2: treat all models equally
        # NOTE: Here you will use all 10 models, and treat them equally ( no model weights )
        # all_preds, all_confidences, all_filenames, all_probs = ensemble_predict(models, test_loader)

        # Convert predictions to labels (actual classes)
        label2idx, idx2label = get_label_maps()
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

        out1 = os.path.join(CONFIG.processed_data_dir, "predictions.csv")
        test_df.to_csv(out1, index=False)

        out2 = os.path.join(
            CONFIG.processed_data_dir, "predictions_with_confidence.csv"
        )
        test_conf_df.to_csv(out2, index=False)

        # Print some statistics
        logger.info(f"Total predictions: {len(all_preds)}")
        logger.info(f"Average confidence: {np.mean(all_confidences):.4f}")
        logger.info(f"Min confidence: {np.min(all_confidences):.4f}")
        logger.info(f"Max confidence: {np.max(all_confidences):.4f}")
