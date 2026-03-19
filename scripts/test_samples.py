"""
test_samples.py

Runs inference on sample images using the trained YOLO model
and saves annotated results to results/samples/.

Usage:
    python scripts/test_samples.py
"""

import cv2
from pathlib import Path
from ultralytics import YOLO

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

ROOT_DIR     = Path(__file__).resolve().parent.parent
WEIGHTS      = ROOT_DIR / "models" / "best_run6.pt"
SAMPLES_DIR  = ROOT_DIR / "samples"          # put your images here
OUTPUT_DIR   = ROOT_DIR / "results" / "samples"
CONF         = 0.25                           # confidence threshold for display

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("\n" + "─" * 45)
    print("  Sample Image Detection Test")
    print("─" * 45)
    print(f"  Weights    : {WEIGHTS}")
    print(f"  Samples    : {SAMPLES_DIR}")
    print(f"  Output     : {OUTPUT_DIR}")
    print(f"  Conf thr   : {CONF}")
    print("─" * 45 + "\n")

    if not WEIGHTS.exists():
        raise FileNotFoundError(f"Weights not found: {WEIGHTS}")

    if not SAMPLES_DIR.exists():
        SAMPLES_DIR.mkdir(parents=True)
        print(f"[!] Created samples/ folder — add your images there and rerun.")
        return

    images = list(SAMPLES_DIR.glob("*.jpg")) + \
             list(SAMPLES_DIR.glob("*.jpeg")) + \
             list(SAMPLES_DIR.glob("*.png"))

    if not images:
        print(f"[!] No images found in {SAMPLES_DIR}")
        print("    Add .jpg / .jpeg / .png files and rerun.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(WEIGHTS))
    print(f"[✓] Model loaded — running on {len(images)} image(s)\n")

    for img_path in sorted(images):
        results = model.predict(str(img_path), conf=CONF, verbose=False)
        r = results[0]

        # Annotate
        annotated = r.plot()

        # Save
        out_path = OUTPUT_DIR / f"detected_{img_path.name}"
        cv2.imwrite(str(out_path), annotated)

        # Console summary
        dets = len(r.boxes)
        if dets:
            confs = [round(float(b.conf[0]), 3) for b in r.boxes]
            print(f"  {img_path.name:<40} {dets} detection(s)  conf: {confs}")
        else:
            print(f"  {img_path.name:<40} no detections")

    print(f"\n[✓] Annotated images saved to: {OUTPUT_DIR}\n")


if __name__ == "__main__":
    main()
