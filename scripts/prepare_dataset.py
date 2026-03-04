"""
prepare_dataset.py

Converts SDNET2018 dataset into YOLO detection format.
- Cracked images get a whole-image bounding box label
- Non-cracked images get an empty label file
- Dataset is split into train/val/test (80/10/10)

Classes:
    0: deck_crack
    1: pavement_crack
    2: wall_crack
    3: no_crack

Usage:
    python scripts/prepare_dataset.py
"""

import os
import shutil
import random
from pathlib import Path

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

# Paths
ROOT_DIR        = Path(__file__).resolve().parent.parent
DATASET_DIR     = ROOT_DIR / "datasets" / "SDNET"
OUTPUT_DIR      = ROOT_DIR / "datasets" / "SDNET_YOLO"
YAML_OUTPUT     = ROOT_DIR / "data" / "sdnet.yaml"

# Dataset split ratios
TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10
TEST_RATIO  = 0.10

# Random seed for reproducibility
RANDOM_SEED = 42

# Class mapping
# Each entry: (source_folder, cracked_subfolder, non_cracked_subfolder, class_id)
CATEGORY_MAP = [
    ("Decks",     "Cracked", "Non-cracked", 0),   # deck_crack
    ("Pavements", "Cracked", "Non-cracked", 1),   # pavement_crack
    ("Walls",     "Cracked", "Non-cracked", 2),   # wall_crack
]

CLASS_NAMES = ["deck_crack", "pavement_crack", "wall_crack", "no_crack"]
NO_CRACK_CLASS_ID = 3

# Supported image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def make_dirs():
    """Create output directory structure."""
    for split in ["train", "val", "test"]:
        (OUTPUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)
    print(f"[✓] Output directories created at: {OUTPUT_DIR}")


def get_yolo_label(class_id: int, cracked: bool) -> str:
    """
    Returns YOLO label string.
    Cracked   → whole-image bounding box
    No crack  → empty string (no detection)
    """
    if cracked:
        # YOLO format: class_id x_center y_center width height (all normalized)
        return f"{class_id} 0.5 0.5 1.0 1.0\n"
    else:
        return ""  # empty label = background / no object


def split_list(items: list, train_r: float, val_r: float):
    """Split a list into train, val, test subsets."""
    random.shuffle(items)
    n = len(items)
    train_end = int(n * train_r)
    val_end   = int(n * (train_r + val_r))
    return items[:train_end], items[train_end:val_end], items[val_end:]


def copy_and_label(image_path: Path, split: str, class_id: int, cracked: bool):
    """Copy image and write corresponding label to the output directory."""
    # Unique filename to avoid collisions across categories
    unique_name = f"{image_path.parent.parent.name}_{image_path.parent.name}_{image_path.name}"
    stem        = Path(unique_name).stem
    suffix      = image_path.suffix.lower()

    dest_image = OUTPUT_DIR / "images" / split / f"{stem}{suffix}"
    dest_label = OUTPUT_DIR / "labels" / split / f"{stem}.txt"

    # Copy image
    shutil.copy2(image_path, dest_image)

    # Write label
    label_content = get_yolo_label(class_id, cracked)
    with open(dest_label, "w") as f:
        f.write(label_content)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def prepare():
    random.seed(RANDOM_SEED)
    make_dirs()

    total_train = total_val = total_test = 0

    for folder_name, cracked_sub, non_cracked_sub, class_id in CATEGORY_MAP:
        category_path = DATASET_DIR / folder_name

        if not category_path.exists():
            print(f"[!] Folder not found, skipping: {category_path}")
            continue

        print(f"\n[→] Processing: {folder_name}")

        # ── Cracked images ──
        cracked_path = category_path / cracked_sub
        cracked_images = [p for p in cracked_path.iterdir() if is_image(p)] \
                         if cracked_path.exists() else []

        # ── Non-cracked images ──
        non_cracked_path = category_path / non_cracked_sub
        non_cracked_images = [p for p in non_cracked_path.iterdir() if is_image(p)] \
                              if non_cracked_path.exists() else []

        print(f"    Cracked:     {len(cracked_images)} images")
        print(f"    Non-cracked: {len(non_cracked_images)} images")

        # Split cracked
        c_train, c_val, c_test = split_list(cracked_images, TRAIN_RATIO, VAL_RATIO)

        # Split non-cracked
        n_train, n_val, n_test = split_list(non_cracked_images, TRAIN_RATIO, VAL_RATIO)

        # Copy cracked
        for split, subset in [("train", c_train), ("val", c_val), ("test", c_test)]:
            for img in subset:
                copy_and_label(img, split, class_id, cracked=True)

        # Copy non-cracked
        for split, subset in [("train", n_train), ("val", n_val), ("test", n_test)]:
            for img in subset:
                copy_and_label(img, split, NO_CRACK_CLASS_ID, cracked=False)

        cat_train = len(c_train) + len(n_train)
        cat_val   = len(c_val)   + len(n_val)
        cat_test  = len(c_test)  + len(n_test)

        print(f"    Split → Train: {cat_train}  Val: {cat_val}  Test: {cat_test}")

        total_train += cat_train
        total_val   += cat_val
        total_test  += cat_test

    # ─────────────────────────────────────────
    # SUMMARY
    # ─────────────────────────────────────────
    total = total_train + total_val + total_test
    print(f"\n{'─'*45}")
    print(f"[✓] Dataset preparation complete")
    print(f"    Total images : {total}")
    print(f"    Train        : {total_train}")
    print(f"    Val          : {total_val}")
    print(f"    Test         : {total_test}")
    print(f"    Output       : {OUTPUT_DIR}")
    print(f"{'─'*45}\n")

    # ─────────────────────────────────────────
    # WRITE YAML
    # ─────────────────────────────────────────
    YAML_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    yaml_content = f"""# SDNET2018 — YOLO Detection Dataset Config
# Auto-generated by prepare_dataset.py

path: {OUTPUT_DIR.as_posix()}
train: images/train
val:   images/val
test:  images/test

nc: {len(CLASS_NAMES)}
names:
"""
    for i, name in enumerate(CLASS_NAMES):
        yaml_content += f"  {i}: {name}\n"

    with open(YAML_OUTPUT, "w") as f:
        f.write(yaml_content)

    print(f"[✓] YAML config written to: {YAML_OUTPUT}\n")


if __name__ == "__main__":
    prepare()