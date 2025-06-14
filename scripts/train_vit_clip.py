from pathlib import Path
import torch
from transformers import ViTFeatureExtractor, ViTForImageClassification
from torch.utils.data import DataLoader
import pandas as pd
from src.utils import Logger, ConfigManager, sort_images_for_imagefolder, Visualizer
from src.modeling import (
    evaluate_vit_clip_model,
    train_pl,
    build_vit_clip_classifier,
    predict_vit_clip_model,
)
from src.data import TrainDataset, TestDataset, TrainCollator, TestCollator
from src.utils import get_test_image_files

torch.set_float32_matmul_precision("high")


class VitClipTrainer:
    """Main trainer class for Vision Transformer + CLIP hybrid model."""

    def __init__(self, config_dir: str = "configs"):
        self.log = Logger()
        self.config_manager = ConfigManager(config_dir)
        self.config = self.config_manager.config
        self.visualizer = Visualizer(self.config["output_dir"])

        # Initialize placeholders
        self.hybrid_model = None
        self.feature_extractor = None
        self.clip_preprocess = None
        self.class_mappings = None
        self.classes = None

    def setup_data(self):
        """Setup training and validation datasets."""
        self.log.info("Setting up training data...")

        # Sort images for ImageFolder structure
        data_dir = sort_images_for_imagefolder(
            train_dir=self.config["train_data_dir"],
            labels_file=self.config["labels_file"],
            train_sorted=self.config["train_sorted"],
        )

        # Create dataset splits
        train_dataset, val_dataset, self.classes = TrainDataset.create_dataset_split(
            data_dir,
            val_split=float(self.config["val_split"]),
            seed=int(self.config["seed"]),
        )

        self.log.info(
            f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}"
        )

        # Create label mappings
        model_save_path = self.config["model_save_path"]
        self.class_mappings = TrainDataset.create_label_mappings(
            self.classes, save_path=model_save_path
        )
        self.log.info(f"Label to ID: {self.class_mappings['label_to_id']}")

        return train_dataset, val_dataset

    def setup_models(self):
        """Initialize Vision Transformer and hybrid models."""
        self.log.info("Creating Vision Transformer model...")

        # Create ViT feature extractor
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(
            pretrained_model_name_or_path=str(self.config["vit"]["model_name"])
        )

        # Create ViT model
        vit_model = ViTForImageClassification.from_pretrained(
            pretrained_model_name_or_path=str(self.config["vit"]["model_name"]),
            num_labels=len(self.class_mappings["label_to_id"]),
            label2id=self.class_mappings["label_to_id"],
            id2label=self.class_mappings["id_to_label"],
        )

        # Create hybrid ViT + CLIP model
        self.log.info("Creating hybrid Vision Transformer + CLIP model...")
        self.hybrid_model, self.clip_preprocess = build_vit_clip_classifier(
            vit_model=vit_model,
            num_classes=len(self.classes),
            fusion_method=self.config["fusion"]["method"],
            clip_model_name=str(self.config["clip"]["model_name"]),
        )

    def create_data_loaders(self, train_dataset, val_dataset):
        """Create training and validation data loaders."""
        collator = TrainCollator(self.feature_extractor, self.clip_preprocess)
        train_loader, val_loader = TrainDataset.create_data_loaders(
            train_dataset,
            val_dataset,
            collator,
            batch_size=self.config["batch_size"],
            num_workers=self.config["num_workers"],
        )
        return train_loader, val_loader

    def train_model(self, train_loader, val_loader):
        """Train the hybrid model."""
        self.log.info("Starting training...")

        history = train_pl(
            classifier=self.hybrid_model,
            train_loader=train_loader,
            val_loader=val_loader,
            max_epochs=self.config["epochs"],
            seed=self.config["seed"],
            default_root_dir=self.config["model_save_path"],
            patience=self.config["patience"],
        )

        return history

    def save_model(self):
        """Save the trained model and feature extractor."""
        self.log.info("Saving model...")

        model_save_path = self.config["model_save_path"]
        torch.save(self.hybrid_model.state_dict(), model_save_path / "hybrid_model.pt")
        self.feature_extractor.save_pretrained(model_save_path)

        self.log.info(f"Hybrid model and feature extractor saved to {model_save_path}")

    def evaluate_model(self, val_loader):
        """Evaluate the trained model."""
        self.log.info("Evaluating hybrid model...")

        f1_score, y_true, y_pred = evaluate_vit_clip_model(
            self.hybrid_model, val_loader, return_predictions=True
        )

        self.log.info(
            f"Hybrid Vision Transformer + CLIP Validation Macro F1 Score: {f1_score:.4f}"
        )

        return f1_score, y_true, y_pred

    def create_visualizations(self, history, y_true, y_pred):
        """Create and save visualizations."""
        self.log.info("Creating visualizations...")

        # Plot validation metrics
        self.visualizer.plot_metrics(history)

        # Plot confusion matrix
        self.visualizer.plot_confusion_matrix(
            y_true=y_true,
            y_pred=y_pred,
            class_names=list(self.class_mappings["id_to_label"].values()),
        )

        # Save classification report
        self.visualizer.save_classification_report(
            y_true=y_true,
            y_pred=y_pred,
            class_names=list(self.class_mappings["id_to_label"].values()),
        )

        # Save training history
        self.visualizer.save_training_history(history)

        self.log.info(f"Visualizations saved to {self.config['output_dir']}")

    def generate_test_predictions(self):
        """Generate predictions for test data."""
        self.log.info("Generating predictions for test data...")

        # Setup test data
        test_data_dir = Path(self.config["test_data_dir"])
        test_files = get_test_image_files(test_data_dir)
        test_dataset = TestDataset(test_data_dir, test_files)

        # Create test data loader
        collator = TestCollator(self.feature_extractor, self.clip_preprocess)
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config["batch_size"],
            collate_fn=collator,
            num_workers=self.config["num_workers"],
        )

        # Generate predictions
        filenames, predictions, confidences = predict_vit_clip_model(
            self.hybrid_model, test_loader, self.class_mappings
        )

        return filenames, predictions, confidences

    def save_predictions(self, filenames, predictions, confidences):
        """Save prediction results to CSV files."""
        # Basic submission file
        submission_df = pd.DataFrame({"filename": filenames, "label": predictions})

        # Submission with confidence scores
        submission_with_conf_df = pd.DataFrame(
            {"filename": filenames, "label": predictions, "confidence": confidences}
        )

        # Save files
        output_dir = self.config["output_dir"]
        submission_df.to_csv(f"{output_dir}/submission.csv", index=False)
        submission_with_conf_df.to_csv(
            f"{output_dir}/submission_with_confidence.csv", index=False
        )

        self.log.info(f"Predictions saved to {output_dir}")

    def run_full_pipeline(self):
        """Execute the complete training and evaluation pipeline."""
        try:
            # Setup phase
            train_dataset, val_dataset = self.setup_data()
            self.setup_models()
            train_loader, val_loader = self.create_data_loaders(
                train_dataset, val_dataset
            )

            # Training phase
            history = self.train_model(train_loader, val_loader)
            self.save_model()

            # Evaluation phase
            f1_score, y_true, y_pred = self.evaluate_model(val_loader)
            self.create_visualizations(history, y_true, y_pred)

            # Prediction phase
            filenames, predictions, confidences = self.generate_test_predictions()
            self.save_predictions(filenames, predictions, confidences)

            self.log.info("Pipeline completed successfully!")
            return f1_score

        except Exception as e:
            self.log.error(f"Pipeline failed with error: {str(e)}")
            raise


def main():
    """Main entry point."""
    trainer = VitClipTrainer(config_dir="configs")
    final_f1_score = trainer.run_full_pipeline()
    print(f"Final F1 Score: {final_f1_score:.4f}")


if __name__ == "__main__":
    main()
