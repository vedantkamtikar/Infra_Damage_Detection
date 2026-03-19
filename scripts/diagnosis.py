from pathlib import Path

def main():
    ROOT_DIR = Path(__file__).resolve().parent.parent
    cracks_dir = ROOT_DIR / "datasets" / "CRACKS"
    yaml_path  = cracks_dir / "data.yaml"

    train = (cracks_dir / "train" / "images").as_posix()
    val   = (cracks_dir / "valid" / "images").as_posix()
    test  = (cracks_dir / "test"  / "images").as_posix()

    content = f"""names:
- crack
nc: 1
train: {train}
val: {val}
test: {test}
"""
    yaml_path.write_text(content)
    print("data.yaml updated successfully:")
    print(content)

if __name__ == "__main__":
    main()