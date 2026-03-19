"""
filter_dataset.py

Filters ENIM dataset annotations by removing labels where bounding box height > 0.9
(full-image-height sloppy annotations) and creates a cleaned copy of the dataset.

The original dataset is NOT modified — a new folder ENIM_filtered is created.

Usage:
    python scripts/filter_dataset.py
"""

import os
import shutil
from pathlib import Path

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

ROOT_DIR        = Path(__file__).resolve().parent.parent
SRC_DIR         = ROOT_DIR / "datasets" / "ENIM"
DST_DIR         = ROOT_DIR / "datasets" / "ENIM_filtered"

HEIGHT_THRESHOLD = 0.9      # boxes with height > this are considered sloppy
SPLITS           = ["train", "valid", "test"]

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def filter_label_file(src_lbl: Path, dst_lbl: Path) -> tuple[int, int]:
    """
    Reads a YOLO label file, removes lines where box height > threshold.
    Returns (original_count, kept_count).
    If no lines remain, writes an empty file (YOLO treats as background).
    """
    with open(src_lbl, "r") as f:
        lines = [l.strip() for l in f if l.strip()]

    kept = []
    for line in lines:
        parts = line.split()
        if len(parts) != 5:
            kept.append(line)   # malformed line — keep as-is
            continue
        h = float(parts[4])
        if h <= HEIGHT_THRESHOLD:
            kept.append(line)

    dst_lbl.parent.mkdir(parents=True, exist_ok=True)
    with open(dst_lbl, "w") as f:
        f.write("\n".join(kept) + ("\n" if kept else ""))

    return len(lines), len(kept)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("\n" + "─" * 50)
    print("  ENIM Dataset Filter")
    print("─" * 50)
    print(f"  Source      : {SRC_DIR}")
    print(f"  Destination : {DST_DIR}")
    print(f"  Threshold   : box height > {HEIGHT_THRESHOLD} → removed")
    print("─" * 50 + "\n")

    if DST_DIR.exists():
        print(f"[!] Destination already exists. Removing: {DST_DIR}")
        shutil.rmtree(DST_DIR)

    total_boxes_before = 0
    total_boxes_after  = 0
    total_images       = 0
    images_fully_empty = 0   # images where ALL boxes were removed

    for split in SPLITS:
        img_src = SRC_DIR  / split / "images"
        lbl_src = SRC_DIR  / split / "labels"
        img_dst = DST_DIR  / split / "images"
        lbl_dst = DST_DIR  / split / "labels"

        if not img_src.exists():
            print(f"[~] Skipping {split} — no images folder found")
            continue

        # Copy images as-is
        print(f"[→] Copying {split} images...")
        shutil.copytree(img_src, img_dst)

        if not lbl_src.exists():
            print(f"[~] No labels folder for {split} — skipping label filter")
            continue

        # Filter labels
        print(f"[→] Filtering {split} labels...")
        split_before = 0
        split_after  = 0
        split_empty  = 0

        for lbl_file in sorted(lbl_src.iterdir()):
            if lbl_file.suffix != ".txt":
                continue
            dst_file = lbl_dst / lbl_file.name
            before, after = filter_label_file(lbl_file, dst_file)
            split_before += before
            split_after  += after
            total_images += 1
            if before > 0 and after == 0:
                split_empty += 1

        total_boxes_before += split_before
        total_boxes_after  += split_after
        images_fully_empty += split_empty

        removed = split_before - split_after
        print(f"    {split:<8} : {split_before} boxes → {split_after} kept "
              f"({removed} removed, {split_empty} images now background-only)")

    # Copy data.yaml and update paths
    src_yaml = SRC_DIR / "data.yaml"
    dst_yaml = DST_DIR / "data.yaml"
    if src_yaml.exists():
        content = src_yaml.read_text()
        # Replace ENIM path references with ENIM_filtered
        content = content.replace(str(SRC_DIR).replace("\\", "/"),
                                  str(DST_DIR).replace("\\", "/"))
        content = content.replace("ENIM/train", "ENIM_filtered/train")
        content = content.replace("ENIM/valid", "ENIM_filtered/valid")
        content = content.replace("ENIM/test",  "ENIM_filtered/test")
        dst_yaml.write_text(content)
        print(f"\n[✓] data.yaml copied and updated → {dst_yaml}")

    # Summary
    total_removed = total_boxes_before - total_boxes_after
    print("\n" + "─" * 50)
    print("  Filter Complete")
    print("─" * 50)
    print(f"  Total boxes before : {total_boxes_before}")
    print(f"  Total boxes after  : {total_boxes_after}")
    print(f"  Boxes removed      : {total_removed} "
          f"({100*total_removed/total_boxes_before:.1f}%)")
    print(f"  Images now bg-only : {images_fully_empty}")
    print(f"  Filtered dataset   : {DST_DIR}")
    print("─" * 50 + "\n")
    print("Next step: update YAML_PATH in train.py to point to:")
    print(f"  datasets/ENIM_filtered/data.yaml\n")


if __name__ == "__main__":
    main()