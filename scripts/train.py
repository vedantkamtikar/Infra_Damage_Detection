"""
train.py

Fine-tunes YOLO11m on the ENIM crack detection dataset.

Classes:
    0: crack

Usage:
    Fresh run  : python scripts/train.py
    Resume run : set RESUME = True, then python scripts/train.py
"""

import shutil
import torch
from ultralytics import YOLO
from pathlib import Path

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

ROOT_DIR    = Path(__file__).resolve().parent.parent
YAML_PATH   = ROOT_DIR / "datasets" / "CRACKS" / "data.yaml"
MODELS_DIR  = ROOT_DIR / "models"
RUNS_DIR    = ROOT_DIR / "runs"

# Training hyperparameters
MODEL       = "yolo11m.pt"   # pretrained YOLO11m weights (auto-downloaded)
EPOCHS      = 300             # number of training epochs
IMAGE_SIZE  = 800           # input image size
BATCH_SIZE  = 6              # safe for 6.4GB VRAM
WORKERS     = 4              # dataloader workers
PATIENCE    = 30             # early stopping patience (epochs without improvement)
PROJECT     = str(RUNS_DIR)  # where YOLO saves training runs
RUN_NAME    = "Run8"

# Set to True to resume from last checkpoint instead of starting fresh
RESUME      = False

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
        print(f"  VRAM allocated  : {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        print(f"  VRAM reserved   : {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
    else:
        print("  [!] WARNING: CUDA not available. Training will be very slow on CPU.")
    print(f"  Dataset config  : {YAML_PATH}")
    print(f"  Dataset exists  : {YAML_PATH.exists()}")
    print(f"  Resume          : {RESUME}")
    print("─" * 45 + "\n")

    if not YAML_PATH.exists():
        raise FileNotFoundError(f"Dataset config not found: {YAML_PATH}")


# ─────────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────────

def train():
    verify_environment()

    # Load model — pretrained weights for fresh run, last.pt for resume
    if RESUME:
        last_weights = RUNS_DIR / RUN_NAME / "weights" / "last.pt"
        if not last_weights.exists():
            raise FileNotFoundError(
                f"Cannot resume: last.pt not found at {last_weights}\n"
                f"Set RESUME = False to start a fresh run."
            )
        print(f"[→] Resuming from: {last_weights}")
        model = YOLO(str(last_weights))
    else:
        print(f"[→] Loading model: {MODEL}")
        model = YOLO(MODEL)

    print(f"[→] Starting training for {EPOCHS} epochs...")
    print(f"    Batch size : {BATCH_SIZE}")
    print(f"    Image size : {IMAGE_SIZE}")
    print(f"    Patience   : {PATIENCE} epochs\n")

    # Train
    results = model.train(
        data         = str(YAML_PATH),
        epochs       = EPOCHS,
        imgsz        = IMAGE_SIZE,
        batch        = BATCH_SIZE,
        workers      = WORKERS,
        patience     = PATIENCE,
        project      = PROJECT,
        name         = RUN_NAME,
        device       = 0,
        pretrained   = True,
        optimizer    = "AdamW",

        lr0          = 0.0001,
        lrf          = 0.01,
        weight_decay = 0.0005,
        warmup_epochs= 3,

        freeze       = 0,
        augment      = True,
        verbose      = True,
        amp          = False,

        resume       = RESUME,
        exist_ok     = RESUME,
    )

    # ─────────────────────────────────────────
    # SAVE BEST WEIGHTS
    # ─────────────────────────────────────────
    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    dest_weights = MODELS_DIR / "best_run8.pt"   

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if best_weights.exists():
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