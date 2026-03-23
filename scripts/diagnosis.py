import os
import cv2
import numpy as np
from pathlib import Path

def main():
    ROOT_DIR = Path(__file__).resolve().parent.parent
    img_dir  = ROOT_DIR / "datasets" / "CRACKS" / "train" / "images"
    lbl_dir  = ROOT_DIR / "datasets" / "CRACKS" / "train" / "labels"
    out_dir  = ROOT_DIR / "results" / "label_check"
    out_dir.mkdir(parents=True, exist_ok=True)

    checked = 0
    for fname in sorted(os.listdir(img_dir))[:20]:  # check first 20
        img_path = img_dir / fname
        lbl_path = lbl_dir / fname.replace(".jpg", ".txt").replace(".png", ".txt")

        if not lbl_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        h, w = img.shape[:2]

        with open(lbl_path) as f:
            lines = [l.strip() for l in f if l.strip()]

        if not lines:
            continue

        for line in lines:
            parts = line.split()
            if len(parts) != 5:
                continue
            cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            x1 = int((cx - bw/2) * w)
            y1 = int((cy - bh/2) * h)
            x2 = int((cx + bw/2) * w)
            y2 = int((cy + bh/2) * h)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"{bw:.2f}x{bh:.2f}", (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        out_path = out_dir / f"check_{fname}"
        cv2.imwrite(str(out_path), img)
        checked += 1

    print(f"Saved {checked} annotated images to: {out_dir}")

if __name__ == "__main__":
    main()