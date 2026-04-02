"""
evaluate.py

Evaluates the trained YOLO11m model on the CRACKS crack detection val set.
Prints per-class metrics, mAP50, mAP50-95, and saves results.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --weights path/to/custom.pt
    python scripts/evaluate.py --weights path/to/custom.pt --yaml path/to/data.yaml
"""

import argparse
import torch
from ultralytics import YOLO
from pathlib import Path

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

ROOT_DIR        = Path(__file__).resolve().parent.parent
YAML_PATH       = ROOT_DIR / "datasets" / "CRACKS" / "data.yaml"
MODELS_DIR      = ROOT_DIR / "models"
RUNS_DIR        = ROOT_DIR / "runs"

DEFAULT_WEIGHTS = MODELS_DIR / "best_run9.pt"

CLASS_NAMES     = ["crack"]

# ─────────────────────────────────────────────
# VERIFY ENVIRONMENT
# ─────────────────────────────────────────────

def verify_environment(weights_path: Path, yaml_path: Path):
    print("\n" + "─" * 45)
    print("  Evaluation Setup")
    print("─" * 45)
    print(f"  PyTorch version : {torch.__version__}")
    print(f"  CUDA available  : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU             : {torch.cuda.get_device_name(0)}")
    print(f"  Weights         : {weights_path}")
    print(f"  Weights exist   : {weights_path.exists()}")
    print(f"  Dataset config  : {yaml_path}")
    print(f"  Dataset exists  : {yaml_path.exists()}")
    print("─" * 45 + "\n")

    if not weights_path.exists():
        raise FileNotFoundError(
            f"Model weights not found: {weights_path}\n"
            f"Run train.py first, or pass --weights path/to/best.pt"
        )
    if not yaml_path.exists():
        raise FileNotFoundError(f"Dataset config not found: {yaml_path}")


# ─────────────────────────────────────────────
# EVALUATE
# ─────────────────────────────────────────────

def evaluate(weights_path: Path, yaml_path: Path):
    verify_environment(weights_path, yaml_path)

    print(f"[→] Loading model: {weights_path}")
    model = YOLO(str(weights_path))

    print(f"[→] Running evaluation on val split...\n")

    results = model.val(
        data      = str(yaml_path),
        split     = "val",            # val split (test split has no labels)
        imgsz     = 640,              # CRACKS dataset is 640x640 natively
        batch     = 8,
        conf      = 0.2,            # low threshold for proper mAP computation
        iou       = 0.3,              # standard IoU threshold
        device    = 0 if torch.cuda.is_available() else "cpu",
        verbose   = True,
        amp       = False,            # keep consistent with training
        project   = str(ROOT_DIR / "results"),
        name      = f"{Path(weights_path).stem}_eval",
        plots     = True,             # saves confusion matrix, PR curve etc.
        save_json = False,
        exist_ok  = True,             # overwrite previous eval results
    )

    # ─────────────────────────────────────────
    # PRINT SUMMARY
    # ─────────────────────────────────────────
    print("\n" + "─" * 45)
    print("  Evaluation Results (Val Set)")
    print("─" * 45)

    if hasattr(results, 'box'):
        box = results.box
        print(f"\n  {'Class':<20} {'Precision':>10} {'Recall':>10} {'mAP50':>10} {'mAP50-95':>10}")
        print(f"  {'─'*20} {'─'*10} {'─'*10} {'─'*10} {'─'*10}")

        # Overall
        print(f"  {'all':<20} {box.mp:>10.3f} {box.mr:>10.3f} {box.map50:>10.3f} {box.map:>10.3f}")

        # Per class
        if box.ap_class_index is not None:
            for i, cls_idx in enumerate(box.ap_class_index):
                cls_name = CLASS_NAMES[cls_idx] if cls_idx < len(CLASS_NAMES) else f"class_{cls_idx}"
                p     = box.p[i]    if box.p    is not None else 0
                r     = box.r[i]    if box.r    is not None else 0
                a50   = box.ap50[i] if box.ap50 is not None else 0
                a5095 = box.ap[i]   if box.ap   is not None else 0
                print(f"  {cls_name:<20} {p:>10.3f} {r:>10.3f} {a50:>10.3f} {a5095:>10.3f}")

    print(f"\n  Overall mAP50    : {results.box.map50:.4f}")
    print(f"  Overall mAP50-95 : {results.box.map:.4f}")
    print(f"\n  Plots saved to   : {results.save_dir}")
    print("─" * 45 + "\n")

    return results


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate YOLO11m on CRACKS val set")
    parser.add_argument(
        "--weights",
        type=str,
        default=str(DEFAULT_WEIGHTS),
        help=f"Path to model weights (default: {DEFAULT_WEIGHTS})"
    )
    parser.add_argument(
        "--yaml",
        type=str,
        default=str(YAML_PATH),
        help=f"Path to dataset yaml (default: {YAML_PATH})"
    )
    args = parser.parse_args()

    evaluate(Path(args.weights), Path(args.yaml))