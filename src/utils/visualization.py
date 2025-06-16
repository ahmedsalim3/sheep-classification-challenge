import matplotlib.pyplot as plt
import seaborn as sns

from .. import ConfigManager

CONFIG = ConfigManager()


def plot_metrics(history, save_path):
    sns.set_style("darkgrid")
    colors = {"train": "#124467", "val": "#084d02"}

    # Make all metrics same length (minimum length across)
    min_len = min(len(v) for v in history.values())
    for k in history:
        history[k] = history[k][:min_len]
    epochs = range(1, min_len + 1)

    fig, axes = plt.subplots(1, 4, figsize=(24, 5))  # 1 row, 4 columns

    # Loss Plot
    ax = axes[0]
    ax.plot(
        epochs,
        history["train_loss"],
        marker="o",
        markersize=6,
        linewidth=2,
        color=colors["train"],
        label="Train Loss",
    )
    ax.plot(
        epochs,
        history["val_loss"],
        marker="s",
        markersize=6,
        linewidth=2,
        color=colors["val"],
        label="Val Loss",
    )
    ax.set_title("Train and Val Loss", fontsize=14, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right", frameon=True, fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.6)

    # Acc Plot
    ax = axes[1]
    ax.plot(
        epochs,
        history["train_acc"],
        marker="o",
        markersize=6,
        linewidth=2,
        color=colors["train"],
        label="Train Acc",
    )
    ax.plot(
        epochs,
        history["val_acc"],
        marker="s",
        markersize=6,
        linewidth=2,
        color=colors["val"],
        label="Val Acc",
    )
    ax.set_title("Train and Val Acc", fontsize=14, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_ylim(0, 1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower right", frameon=True, fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.6)

    # F1 Macro Score
    ax = axes[2]
    ax.plot(
        epochs,
        history["val_f1_macro"],
        marker="o",
        markersize=6,
        linewidth=2,
        color=colors["val"],
        label="Val F1 Macro",
    )
    ax.set_title("Val F1 Macro", fontsize=14, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("F1 Score", fontsize=12)
    ax.set_ylim(0, 1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower right", frameon=True, fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.6)

    # F1 Weighted Score
    ax = axes[3]
    ax.plot(
        epochs,
        history["val_f1_weighted"],
        marker="D",
        markersize=6,
        linewidth=2,
        color=colors["val"],
        label="Val F1 Weighted",
    )
    ax.set_title("Val F1 Weighted", fontsize=14, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("F1 Score", fontsize=12)
    ax.set_ylim(0, 1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower right", frameon=True, fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(
        save_path,
        format="png",
        dpi=300,
        bbox_inches="tight",
    )
    # plt.show()

    return fig
