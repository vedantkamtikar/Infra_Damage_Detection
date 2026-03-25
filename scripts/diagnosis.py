import os
from pathlib import Path
from ultralytics import YOLO

def main():
    ROOT_DIR = Path(__file__).resolve().parent.parent
    lbl_dir  = ROOT_DIR / "datasets" / "CRACKS" / "valid" / "labels"
    img_dir  = ROOT_DIR / "datasets" / "CRACKS" / "valid" / "images"

    model = YOLO(str(ROOT_DIR / "models" / "best_run8.pt"))

    for fname in sorted(os.listdir(img_dir)):
        lbl_file = lbl_dir / fname.replace(".jpg", ".txt")
        if not lbl_file.exists():
            continue
        with open(lbl_file) as f:
            content = f.read().strip()
        if not content:
            continue

        print(f"Image: {fname}")
        print("Ground truth:")
        for line in content.splitlines():
            print(" ", line)

        results = model.predict(str(img_dir / fname), conf=0.001, verbose=False)
        r = results[0]
        print(f"Top 5 predictions ({len(r.boxes)} total):")
        for box in r.boxes[:5]:
            print(f"  conf={box.conf.item():.3f}  xywhn={[round(v,4) for v in box.xywhn[0].tolist()]}")
        print("Image size:", r.orig_shape)
        break

if __name__ == "__main__":
    main()