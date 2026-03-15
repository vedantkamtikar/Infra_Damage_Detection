"""
train.py

Fine-tunes YOLO11m on the prepared SDNET2018 dataset for crack detection.

Classes:
    0: deck_crack
    1: pavement_crack
    2: wall_crack
    3: no_crack

Usage:
    python scripts/train.py
"""

import torch
from ultralytics import YOLO
from pathlib import Path

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

ROOT_DIR    = Path(__file__).resolve().parent.parent
YAML_PATH   = ROOT_DIR / "data" / "sdnet.yaml"
MODELS_DIR  = ROOT_DIR / "models"
RUNS_DIR    = ROOT_DIR / "runs"

# Training hyperparameters
MODEL       = "yolo11m.pt"   # pretrained YOLO11m weights (auto-downloaded)
EPOCHS      = 20             # number of training epochs
IMAGE_SIZE  = 640            # input image size
BATCH_SIZE  = 8    
WORKERS     = 4              # dataloader workers
PATIENCE    = 15             # early stopping patience (epochs without improvement)
PROJECT     = str(RUNS_DIR)  # where YOLO saves training runs
RUN_NAME    = "sdnet_yolo11m"

# ─────────────────────────────────────────────
# VERIFY ENVIRONMENT
# ─────────────────────────────────────────────

def verify_environment():
    print("\n" + "─" * 45)
    print("  Environment Check")
    print("─" * 45)
    print(f"  PyTorch version : {torch.__version__}")
    print(f"  CUDA available  : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU             : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM            : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("  [!] WARNING: CUDA not available. Training will be very slow on CPU.")
    print(f"  Dataset config  : {YAML_PATH}")
    print(f"  Dataset exists  : {YAML_PATH.exists()}")
    print("─" * 45 + "\n")

    if not YAML_PATH.exists():
        raise FileNotFoundError(f"Dataset config not found: {YAML_PATH}\nRun prepare_dataset.py first.")

# ─────────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────────

def train():
    verify_environment()

    # Load pretrained YOLO11m
    print(f"[→] Loading model: {MODEL}")
    model = YOLO(MODEL)

    print(f"[→] Starting training for {EPOCHS} epochs...")
    print(f"    Batch size : {BATCH_SIZE}")
    print(f"    Image size : {IMAGE_SIZE}")
    print(f"    Patience   : {PATIENCE} epochs\n")

    # Train
    results = model.train(
        data        = str(YAML_PATH),
        epochs      = EPOCHS,
        imgsz       = IMAGE_SIZE,
        batch       = BATCH_SIZE,
        workers     = WORKERS,
        patience    = PATIENCE,
        project     = PROJECT,
        name        = RUN_NAME,
        device      = 0,            # GPU 0 (RTX 4050)
        pretrained  = True,
        optimizer   = "AdamW",
        lr0         = 0.001,        # initial learning rate
        weight_decay= 0.0005,
        augment     = True,         # enable data augmentation
        verbose     = True,
        amp = False,
    )

    # ─────────────────────────────────────────
    # SAVE BEST WEIGHTS
    # ─────────────────────────────────────────
    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    dest_weights = MODELS_DIR / "best.pt"

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if best_weights.exists():
        import shutil
        shutil.copy2(best_weights, dest_weights)
        print(f"\n[✓] Best weights saved to: {dest_weights}")
    else:
        print(f"\n[!] Could not find best.pt at: {best_weights}")

    # ─────────────────────────────────────────
    # TRAINING SUMMARY
    # ─────────────────────────────────────────
    print("\n" + "─" * 45)
    print("  Training Complete")
    print("─" * 45)
    print(f"  Results saved to : {results.save_dir}")
    print(f"  Best weights     : {dest_weights}")
    print("─" * 45 + "\n")

    return results


if __name__ == "__main__":
    train()