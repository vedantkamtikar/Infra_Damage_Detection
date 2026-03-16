"""
plot_results.py

Plots training metrics from YOLO results.csv.
Saves a combined figure to results/training_curves.png

Usage:
    python scripts/plot_results.py
    python scripts/plot_results.py --csv runs/sdnet_yolo11m/results.csv
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

ROOT_DIR        = Path(__file__).resolve().parent.parent
DEFAULT_CSV     = ROOT_DIR / "runs" / "sdnet_yolo11m" / "results.csv"
OUTPUT_DIR      = ROOT_DIR / "results"
OUTPUT_FILE     = OUTPUT_DIR / "training_curves.png"

# ─────────────────────────────────────────────
# PLOT
# ─────────────────────────────────────────────

def plot(csv_path: Path):
    if not csv_path.exists():
        raise FileNotFoundError(f"results.csv not found: {csv_path}")

    # Load and clean column names
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    epochs = df["epoch"]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("YOLO11m Training Results — SDNET2018", fontsize=14, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # ── Plot 1: Training Losses ──────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(epochs, df["train/box_loss"], label="Box loss",  color="#e74c3c")
    ax1.plot(epochs, df["train/cls_loss"], label="Cls loss",  color="#3498db")
    ax1.plot(epochs, df["train/dfl_loss"], label="DFL loss",  color="#2ecc71")
    ax1.set_title("Training Losses")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # ── Plot 2: Validation Losses ────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(epochs, df["val/box_loss"], label="Box loss",  color="#e74c3c")
    ax2.plot(epochs, df["val/cls_loss"], label="Cls loss",  color="#3498db")
    ax2.plot(epochs, df["val/dfl_loss"], label="DFL loss",  color="#2ecc71")
    ax2.set_title("Validation Losses")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # ── Plot 3: mAP ──────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(epochs, df["metrics/mAP50(B)"],      label="mAP50",      color="#9b59b6", linewidth=2)
    ax3.plot(epochs, df["metrics/mAP50-95(B)"],   label="mAP50-95",   color="#e67e22", linewidth=2)
    ax3.set_title("mAP Metrics")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("mAP")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # ── Plot 4: Precision & Recall ───────────
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(epochs, df["metrics/precision(B)"],  label="Precision",  color="#1abc9c", linewidth=2)
    ax4.plot(epochs, df["metrics/recall(B)"],     label="Recall",     color="#e74c3c", linewidth=2)
    ax4.set_title("Precision & Recall")
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("Score")
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    # ── Plot 5: Train vs Val Box Loss ────────
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(epochs, df["train/box_loss"], label="Train", color="#3498db", linewidth=2)
    ax5.plot(epochs, df["val/box_loss"],   label="Val",   color="#e74c3c", linewidth=2, linestyle="--")
    ax5.set_title("Train vs Val Box Loss")
    ax5.set_xlabel("Epoch")
    ax5.set_ylabel("Box Loss")
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)

    # ── Plot 6: Train vs Val Cls Loss ────────
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.plot(epochs, df["train/cls_loss"], label="Train", color="#3498db", linewidth=2)
    ax6.plot(epochs, df["val/cls_loss"],   label="Val",   color="#e74c3c", linewidth=2, linestyle="--")
    ax6.set_title("Train vs Val Cls Loss")
    ax6.set_xlabel("Epoch")
    ax6.set_ylabel("Cls Loss")
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3)

    # Save
    plt.savefig(OUTPUT_FILE, dpi=150, bbox_inches="tight")
    print(f"\n[✓] Training curves saved to: {OUTPUT_FILE}\n")
    plt.show()


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot YOLO training metrics")
    parser.add_argument(
        "--csv",
        type=str,
        default=str(DEFAULT_CSV),
        help=f"Path to results.csv (default: {DEFAULT_CSV})"
    )
    args = parser.parse_args()
    plot(Path(args.csv))