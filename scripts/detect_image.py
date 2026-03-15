"""
detect_image.py

Runs YOLO11m inference on a single image or a folder of images.
Draws bounding boxes with class labels and confidence scores.
Saves annotated output to results/detections/.

Usage:
    python scripts/detect_image.py --source path/to/image.jpg
    python scripts/detect_image.py --source path/to/folder/
    python scripts/detect_image.py --source path/to/image.jpg --weights path/to/best.pt
    python scripts/detect_image.py --source path/to/image.jpg --conf 0.3
"""

import argparse
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from pathlib import Path

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

ROOT_DIR        = Path(__file__).resolve().parent.parent
MODELS_DIR      = ROOT_DIR / "models"
RESULTS_DIR     = ROOT_DIR / "results" / "detections"

DEFAULT_WEIGHTS = MODELS_DIR / "best.pt"
DEFAULT_CONF    = 0.25      # confidence threshold
DEFAULT_IOU     = 0.45      # NMS IoU threshold

CLASS_NAMES = ["deck_crack", "pavement_crack", "wall_crack", "no_crack"]

# Colours per class (BGR)
CLASS_COLORS = {
    "deck_crack"      : (0,   0,   255),   # red
    "pavement_crack"  : (0,   165, 255),   # orange
    "wall_crack"      : (0,   255, 255),   # yellow
    "no_crack"        : (0,   200, 0  ),   # green
}

# ─────────────────────────────────────────────
# DRAW DETECTIONS
# ─────────────────────────────────────────────

def draw_detections(image: np.ndarray, result) -> np.ndarray:
    """Draw bounding boxes and labels on image."""
    annotated = image.copy()

    if result.boxes is None or len(result.boxes) == 0:
        cv2.putText(annotated, "No detections", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (128, 128, 128), 2)
        return annotated

    for box in result.boxes:
        cls_idx = int(box.cls[0])
        conf    = float(box.conf[0])
        cls_name = CLASS_NAMES[cls_idx] if cls_idx < len(CLASS_NAMES) else f"class_{cls_idx}"
        color   = CLASS_COLORS.get(cls_name, (255, 255, 255))

        # Bounding box coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Draw box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # Label background
        label = f"{cls_name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)

        # Label text
        cv2.putText(annotated, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

    return annotated


# ─────────────────────────────────────────────
# DETECT
# ─────────────────────────────────────────────

def detect(source: Path, weights_path: Path, conf: float, iou: float):

    # Validate inputs
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Weights not found: {weights_path}\n"
            f"Run train.py first, or pass --weights path/to/best.pt"
        )
    if not source.exists():
        raise FileNotFoundError(f"Source not found: {source}")

    # Collect image paths
    if source.is_dir():
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        image_paths = [p for p in source.iterdir() if p.suffix.lower() in exts]
        if not image_paths:
            raise ValueError(f"No images found in: {source}")
    else:
        image_paths = [source]

    print("\n" + "─" * 45)
    print("  Crack Detection")
    print("─" * 45)
    print(f"  Weights   : {weights_path}")
    print(f"  Source    : {source}")
    print(f"  Images    : {len(image_paths)}")
    print(f"  Conf thr  : {conf}")
    print(f"  IoU thr   : {iou}")
    print(f"  Device    : {'GPU - ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print("─" * 45 + "\n")

    # Load model
    model = YOLO(str(weights_path))
    device = 0 if torch.cuda.is_available() else "cpu"

    # Output directory
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    total_detections = 0

    for img_path in image_paths:
        # Read image
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"  [!] Could not read: {img_path.name}")
            continue

        # Run inference
        results = model.predict(
            source  = str(img_path),
            conf    = conf,
            iou     = iou,
            device  = device,
            verbose = False,
            amp     = False,
        )

        result = results[0]
        n_det  = len(result.boxes) if result.boxes else 0
        total_detections += n_det

        # Print per-image summary
        print(f"  {img_path.name:<40} {n_det} detection(s)")
        if result.boxes and len(result.boxes) > 0:
            for box in result.boxes:
                cls_idx  = int(box.cls[0])
                conf_val = float(box.conf[0])
                cls_name = CLASS_NAMES[cls_idx] if cls_idx < len(CLASS_NAMES) else f"class_{cls_idx}"
                print(f"    → {cls_name:<20} conf: {conf_val:.3f}")

        # Draw and save annotated image
        annotated = draw_detections(image, result)
        out_path  = RESULTS_DIR / f"detected_{img_path.name}"
        cv2.imwrite(str(out_path), annotated)

    # ─────────────────────────────────────────
    # SUMMARY
    # ─────────────────────────────────────────
    print("\n" + "─" * 45)
    print("  Detection Complete")
    print("─" * 45)
    print(f"  Images processed : {len(image_paths)}")
    print(f"  Total detections : {total_detections}")
    print(f"  Saved to         : {RESULTS_DIR}")
    print("─" * 45 + "\n")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run crack detection on image(s)")
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to image file or folder of images"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=str(DEFAULT_WEIGHTS),
        help=f"Path to model weights (default: {DEFAULT_WEIGHTS})"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=DEFAULT_CONF,
        help=f"Confidence threshold (default: {DEFAULT_CONF})"
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=DEFAULT_IOU,
        help=f"NMS IoU threshold (default: {DEFAULT_IOU})"
    )
    args = parser.parse_args()

    detect(
        source       = Path(args.source),
        weights_path = Path(args.weights),
        conf         = args.conf,
        iou          = args.iou,
    )