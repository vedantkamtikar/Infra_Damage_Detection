"""
prepare_dataset.py

Converts SDNET2018 dataset into YOLO detection format.
- Cracked images get a whole-image bounding box label with class 0 (crack)
- Non-cracked images are skipped (no label file = background, handled natively by YOLO)
- Dataset is split into train/val/test (80/10/10)

Classes:
    0: crack
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

# Single class
CLASS_NAMES  = ["crack"]
CRACK_CLASS_ID = 0

# Source folders in SDNET
CATEGORY_MAP = [
    ("Decks",     "Cracked", "Non-cracked"),
    ("Pavements", "Cracked", "Non-cracked"),
    ("Walls",     "Cracked", "Non-cracked"),
]

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


def split_list(items: list, train_r: float, val_r: float):
    """Split a list into train, val, test subsets."""
    random.shuffle(items)
    n = len(items)
    train_end = int(n * train_r)
    val_end   = int(n * (train_r + val_r))
    return items[:train_end], items[train_end:val_end], items[val_end:]


def copy_and_label(image_path: Path, split: str):
    """Copy cracked image and write whole-image bounding box label."""
    unique_name = f"{image_path.parent.parent.name}_{image_path.parent.name}_{image_path.name}"
    stem        = Path(unique_name).stem
    suffix      = image_path.suffix.lower()

    dest_image = OUTPUT_DIR / "images" / split / f"{stem}{suffix}"
    dest_label = OUTPUT_DIR / "labels" / split / f"{stem}.txt"

    shutil.copy2(image_path, dest_image)

    # Whole-image bounding box for single crack class
    with open(dest_label, "w") as f:
        f.write(f"{CRACK_CLASS_ID} 0.5 0.5 1.0 1.0\n")


def copy_no_label(image_path: Path, split: str):
    """Copy non-cracked image with empty label (YOLO background handling)."""
    unique_name = f"{image_path.parent.parent.name}_{image_path.parent.name}_{image_path.name}"
    stem        = Path(unique_name).stem
    suffix      = image_path.suffix.lower()

    dest_image = OUTPUT_DIR / "images" / split / f"{stem}{suffix}"
    dest_label = OUTPUT_DIR / "labels" / split / f"{stem}.txt"

    shutil.copy2(image_path, dest_image)

    # Empty label = no objects = background image
    # YOLO handles this natively, no class needed
    open(dest_label, "w").close()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def prepare():
    random.seed(RANDOM_SEED)

    # Wipe and recreate output directory for clean run
    if OUTPUT_DIR.exists():
        print(f"[!] Removing existing output: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)

    make_dirs()

    total_train = total_val = total_test = 0
    crack_train = crack_val = crack_test = 0
    nocrack_train = nocrack_val = nocrack_test = 0

    for folder_name, cracked_sub, non_cracked_sub in CATEGORY_MAP:
        category_path = DATASET_DIR / folder_name

        if not category_path.exists():
            print(f"[!] Folder not found, skipping: {category_path}")
            continue

        print(f"\n[→] Processing: {folder_name}")

        # ── Cracked images ──────────────────
        cracked_path   = category_path / cracked_sub
        cracked_images = [p for p in cracked_path.iterdir() if is_image(p)] \
                         if cracked_path.exists() else []

        # ── Non-cracked images ──────────────
        non_cracked_path   = category_path / non_cracked_sub
        non_cracked_images = [p for p in non_cracked_path.iterdir() if is_image(p)] \
                             if non_cracked_path.exists() else []

        print(f"    Cracked     : {len(cracked_images)} images")
        print(f"    Non-cracked : {len(non_cracked_images)} images")

        # Split
        c_train, c_val, c_test = split_list(cracked_images,     TRAIN_RATIO, VAL_RATIO)
        n_train, n_val, n_test = split_list(non_cracked_images, TRAIN_RATIO, VAL_RATIO)

        # Copy cracked with label
        for split, subset in [("train", c_train), ("val", c_val), ("test", c_test)]:
            for img in subset:
                copy_and_label(img, split)

        # Copy non-cracked with empty label
        for split, subset in [("train", n_train), ("val", n_val), ("test", n_test)]:
            for img in subset:
                copy_no_label(img, split)

        print(f"    Split → Train: {len(c_train)+len(n_train)}  Val: {len(c_val)+len(n_val)}  Test: {len(c_test)+len(n_test)}")

        crack_train   += len(c_train); crack_val   += len(c_val);   crack_test   += len(c_test)
        nocrack_train += len(n_train); nocrack_val += len(n_val);   nocrack_test += len(n_test)

        total_train += len(c_train) + len(n_train)
        total_val   += len(c_val)   + len(n_val)
        total_test  += len(c_test)  + len(n_test)

    # ─────────────────────────────────────────
    # SUMMARY
    # ─────────────────────────────────────────
    total = total_train + total_val + total_test
    print(f"\n{'─'*45}")
    print(f"[✓] Dataset preparation complete")
    print(f"    Total images  : {total}")
    print(f"    Train         : {total_train}  (crack: {crack_train}, no-crack: {nocrack_train})")
    print(f"    Val           : {total_val}  (crack: {crack_val}, no-crack: {nocrack_val})")
    print(f"    Test          : {total_test}  (crack: {crack_test}, no-crack: {nocrack_test})")
    print(f"    Output        : {OUTPUT_DIR}")
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