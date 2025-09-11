# -----------------------------------------------------------
# Train YOLOv8/YOLO11 segmentation on SKINPAN
# with a FRESH validation split generated every run
# -----------------------------------------------------------
import os
import time
import random
from pathlib import Path
from glob import glob

import yaml
from ultralytics import YOLO
from ultralytics.utils.checks import check_yaml

# -----------------------
# User config
# -----------------------
GPU = 0
MODEL_NAME = "yolo11x-seg.pt"        # any Ultralytics seg model (*.pt)
BASE_DATA_YAML = "./SKINPAN/yolo-dataset.yaml"
DATASET_ROOT = "./SKINPAN"
VAL_FRACTION = 0.2                    # 20% of train images -> validation (random each run)
IMG_SIZE = 640
EPOCHS = 100
BATCH = 16
PROJECT = "runs/segment"              # default Ultralytics project dir


# -----------------------
# Helpers: ephemeral split
# -----------------------
def list_images(dir_path: Path):
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff", "*.webp")
    files = []
    for e in exts:
        files.extend(glob(str(dir_path / e)))
    # convert to absolute paths to avoid YAML 'path' prefixing issues
    return sorted([str(Path(p).resolve()) for p in files])

def make_random_train_val_lists(train_img_dir: Path, val_fraction: float, seed=None):
    imgs = list_images(train_img_dir)
    if not imgs:
        raise RuntimeError(f"No images found in {train_img_dir}")
    if seed is not None:
        random.seed(seed)
    random.shuffle(imgs)
    n_val = max(1, int(len(imgs) * val_fraction))
    val_imgs = set(imgs[:n_val])
    trn_imgs = [p for p in imgs if p not in val_imgs]
    return trn_imgs, list(val_imgs)

def write_lines(path: Path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for ln in lines:
            f.write(str(ln).strip() + "\n")

def build_ephemeral_yaml(base_yaml_path: Path,
                         train_list_txt: Path,
                         val_list_txt: Path,
                         out_yaml_path: Path):
    with open(base_yaml_path, "r") as f:
        base = yaml.safe_load(f)

    # Ensure we don't allow Ultralytics to prefix with 'path'
    base.pop("path", None)

    # Point to our absolute-path list files
    base["train"] = str(train_list_txt.resolve())
    base["val"]   = str(val_list_txt.resolve())

    out_yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_yaml_path, "w") as f:
        yaml.safe_dump(base, f, sort_keys=False)
    return out_yaml_path


# -----------------------
# Build fresh split & YAML
# -----------------------
RUN_ID = time.strftime("%Y%m%d-%H%M%S")
DATASET_ROOT = Path(DATASET_ROOT).resolve()
SPLIT_DIR = DATASET_ROOT / "annotations" / f"yolo_run_split_{RUN_ID}"
TRAIN_LIST = SPLIT_DIR / "train_run.txt"
VAL_LIST   = SPLIT_DIR / "val_run.txt"
EPH_YAML   = SPLIT_DIR / "dataset_run.yaml"

TRAIN_IMG_DIR = DATASET_ROOT / "images" / "train"

# Fresh random split each run
trn_imgs, val_imgs = make_random_train_val_lists(TRAIN_IMG_DIR, VAL_FRACTION, seed=None)
write_lines(TRAIN_LIST, trn_imgs)
write_lines(VAL_LIST, val_imgs)
EPH_YAML_PATH = build_ephemeral_yaml(Path(BASE_DATA_YAML), TRAIN_LIST, VAL_LIST, EPH_YAML)

# Validate the ephemeral yaml (raises if bad)
check_yaml(str(EPH_YAML_PATH))


# -----------------------
# Train + validate
# -----------------------
def train_yolo():
    model = YOLO(MODEL_NAME)

    run_name = f"skinpan_{MODEL_NAME.replace('.pt','')}_{RUN_ID}"
    print(f"\nSplit: {len(trn_imgs)} train / {len(val_imgs)} val images (fresh @ {RUN_ID})")
    print(f"Ephemeral data yaml: {EPH_YAML_PATH}")

    print("\nStarting training...")
    results = model.train(
        data=str(EPH_YAML_PATH),
        imgsz=IMG_SIZE,
        epochs=EPOCHS,
        batch=BATCH,
        device=f"cuda:{GPU}",
        val=True,
        save=True,
        save_period=-1,
        exist_ok=True,
        verbose=True,
        task="segment",
        project=PROJECT,
        name=run_name,
    )

    print("\nStarting post-training validation...")
    val_results = model.val(
        data=str(EPH_YAML_PATH),
        device=f"cuda:{GPU}",
        save_json=True,
        save_hybrid=True,
        save_txt=True,
        plots=True,
        task="segment",
        project=PROJECT,
        name=run_name,  # writes under the same run dir
    )

    return results, val_results


if __name__ == "__main__":
    train_results, val_results = train_yolo()

    # Report directories
    print(f"\nTraining results saved to: {train_results.save_dir}")
    if val_results:
        print(f"Validation results saved to: {val_results.save_dir}")

    # Show the split files that were used
    print(f"\nEphemeral split files:\n  Train list: {TRAIN_LIST}\n  Val list:   {VAL_LIST}")