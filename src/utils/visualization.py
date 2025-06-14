from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import yaml
from sklearn.metrics import classification_report, confusion_matrix


class Visualizer:
    """Visualization tools for validation metrics and evaluation results."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set style
        plt.style.use("seaborn-v0_8")
        sns.set_theme(style="whitegrid")

    def plot_metrics(
        self, history: Dict[str, List[float]], save_name: str = "validation_metrics.png"
    ):
        """
        Plot validation metrics.

        Args:
        ----
            history: Dictionary containing training history
            save_name: Name of the output file

        """
        sns.set_style("darkgrid")
        colors = {"val": "#084d02"}

        min_len = min(len(v) for v in history.values())
        for k in history:
            history[k] = history[k][:min_len]

        epochs = range(1, min_len + 1)

        # Create figure with subplots
        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(2, 2)

        # Loss plot
        ax_loss = fig.add_subplot(gs[0, 0])
        ax_loss.plot(
            epochs,
            history["val_loss"],
            marker="s",
            markersize=6,
            linewidth=2,
            color=colors["val"],
            label="Validation",
        )
        ax_loss.set_title("Validation Loss", fontsize=14, fontweight="bold")
        ax_loss.set_xlabel("Epoch", fontsize=12)
        ax_loss.set_ylabel("Loss", fontsize=12)
        ax_loss.spines["top"].set_visible(False)
        ax_loss.spines["right"].set_visible(False)
        ax_loss.legend(loc="upper right", frameon=True, fontsize=11)
        ax_loss.grid(True, linestyle="--", alpha=0.6)

        # Accuracy plot
        ax_acc = fig.add_subplot(gs[0, 1])
        ax_acc.plot(
            epochs,
            history["val_acc"],
            marker="s",
            markersize=6,
            linewidth=2,
            color=colors["val"],
            label="Validation",
        )
        ax_acc.set_title("Validation Accuracy", fontsize=14, fontweight="bold")
        ax_acc.set_xlabel("Epoch", fontsize=12)
        ax_acc.set_ylabel("Accuracy", fontsize=12)
        ax_acc.set_ylim(0, 1)
        ax_acc.spines["top"].set_visible(False)
        ax_acc.spines["right"].set_visible(False)
        ax_acc.legend(loc="lower right", frameon=True, fontsize=11)
        ax_acc.grid(True, linestyle="--", alpha=0.6)

        # F1 Macro plot
        ax_f1_macro = fig.add_subplot(gs[1, 0])
        ax_f1_macro.plot(
            epochs,
            history["val_f1_macro"],
            marker="D",
            markersize=6,
            linewidth=2,
            color=colors["val"],
            label="F1 Macro",
        )
        ax_f1_macro.set_title("F1 Macro Score", fontsize=14, fontweight="bold")
        ax_f1_macro.set_xlabel("Epoch", fontsize=12)
        ax_f1_macro.set_ylabel("F1 Score", fontsize=12)
        ax_f1_macro.set_ylim(0, 1)
        ax_f1_macro.spines["top"].set_visible(False)
        ax_f1_macro.spines["right"].set_visible(False)
        ax_f1_macro.legend(loc="lower right", frameon=True, fontsize=11)
        ax_f1_macro.grid(True, linestyle="--", alpha=0.6)

        # F1 Micro plot
        ax_f1_micro = fig.add_subplot(gs[1, 1])
        ax_f1_micro.plot(
            epochs,
            history["val_f1_micro"],
            marker="D",
            markersize=6,
            linewidth=2,
            color=colors["val"],
            label="F1 Micro",
        )
        ax_f1_micro.set_title("F1 Micro Score", fontsize=14, fontweight="bold")
        ax_f1_micro.set_xlabel("Epoch", fontsize=12)
        ax_f1_micro.set_ylabel("F1 Score", fontsize=12)
        ax_f1_micro.set_ylim(0, 1)
        ax_f1_micro.spines["top"].set_visible(False)
        ax_f1_micro.spines["right"].set_visible(False)
        ax_f1_micro.legend(loc="lower right", frameon=True, fontsize=11)
        ax_f1_micro.grid(True, linestyle="--", alpha=0.6)

        plt.tight_layout()
        plt.savefig(
            self.output_dir / save_name, format="png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: List[str],
        save_name: str = "confusion_matrix.png",
    ):
        """
        Plot confusion matrix.

        Args:
        ----
            y_true: True labels
            y_pred: Predicted labels
            class_names: List of class names
            save_name: Name of the output file

        """
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(12, 10))

        # Create heatmap
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
        )

        plt.title("Confusion Matrix", fontsize=14, fontweight="bold", pad=20)
        plt.xlabel("Predicted", fontsize=12, labelpad=10)
        plt.ylabel("True", fontsize=12, labelpad=10)
        # plt.xticks(rotation=45, ha='right')
        # plt.yticks(rotation=0)
        plt.tight_layout()
        save_path = self.output_dir / "results"
        save_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path / save_name, format="png", dpi=300, bbox_inches="tight")
        plt.close()

    def save_classification_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: List[str],
        save_name: str = "classification_report.txt",
    ):
        """
        Save classification report.

        Args:
        ----
            y_true: True labels
            y_pred: Predicted labels
            class_names: List of class names
            save_name: Name of the output file

        """
        report = classification_report(
            y_true, y_pred, target_names=class_names, digits=4
        )
        save_path = self.output_dir / "results"
        save_path.mkdir(parents=True, exist_ok=True)

        with open(save_path / save_name, "w") as f:
            f.write(report)

    def save_training_history(
        self, history: Dict[str, List[float]], save_name: str = "training_history.yml"
    ):
        """
        Save training history to YAML file.

        Args:
        ----
            history: Dictionary containing training history
            save_name: Name of the output file

        """
        with open(self.output_dir / save_name, "w") as f:
            yaml.dump(history, f)
