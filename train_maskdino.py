# -----------------------------------------------------------
# Train MaskDINO segmentation on SKINPAN
# with a FRESH validation split generated every run
# -----------------------------------------------------------
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
sys.path.append("MaskDINO")

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import json
import time
import random
from pathlib import Path

import torch

from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine.hooks import BestCheckpointer
from detectron2.evaluation import COCOEvaluator
from detectron2.projects.deeplab import add_deeplab_config

from maskdino.config import add_maskdino_config
from train_net import Trainer


# -------------------------------
# Custom Trainer (keep your evaluator and best-checkpoint logic)
# -------------------------------
class MyTrainer(Trainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, False, output_folder)

    def build_hooks(self):
        hooks = super().build_hooks()
        # remove periodic saver
        hooks = [h for h in hooks if h.__class__.__name__ != "PeriodicCheckpointer"]
        # add best saver
        best_checkpointer = BestCheckpointer(
            cfg.TEST.EVAL_PERIOD,
            self.checkpointer,
            "bbox/AP",   # keep bbox/AP (MaskDINO reports both bbox/segm; change if desired)
            "max",
            file_prefix="best_model"
        )
        hooks.insert(-1, best_checkpointer)  # before eval hook
        return hooks


# -------------------------------
# Config (edit to taste)
# -------------------------------
NUM_EPOCHS = 50
EPOCHS_PER_STEP = 5
IMS_PER_BATCH = 4
NUM_WORKERS = 4
VAL_FRACTION = 0.2   # 20% of *train* images -> validation (random every run)

CONFIG_PATH = "MaskDINO/configs/coco/instance-segmentation/swin/maskdino_R50_bs16_50ep_4s_dowsample1_2048.yaml"
PRETRAINED_WEIGHTS = "maskdino_swinl_50ep_300q_hid2048_3sd1_instance_maskenhanced_mask52.3ap_box59.0ap.pth"

DATASET_PATH = "./SKINPAN"  # expects SKINPAN/annotations/instances_train.json and images/train/

torch.cuda.empty_cache()


# -------------------------------
# Helper: create a fresh train/valid split (COCO JSON) on each run
# -------------------------------
def make_random_val_split(coco_train_json: str, out_dir: str, val_fraction: float = 0.2):
    """
    Load COCO training JSON, randomly split *images* into train/valid,
    and write two new COCO JSONs to out_dir:
      - instances_train_run.json
      - instances_valid_run.json
    Returns: (train_json_path, valid_json_path)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(coco_train_json, "r") as f:
        coco = json.load(f)

    images = coco.get("images", [])
    anns = coco.get("annotations", [])
    cats = coco.get("categories", [])
    info = coco.get("info", {})
    lic = coco.get("licenses", [])

    img_ids = [im["id"] for im in images]
    random.shuffle(img_ids)  # << fresh split every run
    n_val = max(1, int(len(img_ids) * val_fraction))
    val_ids = set(img_ids[:n_val])
    trn_ids = set(img_ids[n_val:])

    def subset(idset):
        sub_images = [im for im in images if im["id"] in idset]
        sub_anns = [a for a in anns   if a["image_id"] in idset]
        return {
            "info": info, "licenses": lic,
            "images": sub_images,
            "annotations": sub_anns,
            "categories": cats
        }

    split_train = subset(trn_ids)
    split_valid = subset(val_ids)

    train_json_path = out_dir / "instances_train_run.json"
    valid_json_path = out_dir / "instances_valid_run.json"
    with open(train_json_path, "w") as f:
        json.dump(split_train, f)
    with open(valid_json_path, "w") as f:
        json.dump(split_valid, f)

    return str(train_json_path), str(valid_json_path)


# -------------------------------
# Build an ephemeral split dir with timestamp
# -------------------------------
RUN_ID = time.strftime("%Y%m%d-%H%M%S")
ANNOT_DIR = Path(DATASET_PATH) / "annotations"
SPLIT_DIR = ANNOT_DIR / f"run_split_{RUN_ID}"

TRAIN_JSON_SRC = ANNOT_DIR / "instances_train.json"
TRAIN_JSON_EPH, VALID_JSON_EPH = make_random_val_split(
    coco_train_json=str(TRAIN_JSON_SRC),
    out_dir=str(SPLIT_DIR),
    val_fraction=VAL_FRACTION
)

# -------------------------------
# Register the ephemeral train/valid datasets
# -------------------------------
register_coco_instances(
    name="dataset_train",
    metadata={},
    json_file=TRAIN_JSON_EPH,
    image_root=f"{DATASET_PATH}/images/train"
)
register_coco_instances(
    name="dataset_valid",
    metadata={},
    json_file=VALID_JSON_EPH,
    image_root=f"{DATASET_PATH}/images/train"
)


# -------------------------------
# Setup Detectron2 + MaskDINO Config
# -------------------------------
cfg = get_cfg()
add_deeplab_config(cfg)
add_maskdino_config(cfg)
cfg.merge_from_file(CONFIG_PATH)

cfg.MODEL.WEIGHTS = PRETRAINED_WEIGHTS
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 1

cfg.DATASETS.TRAIN = ("dataset_train",)
cfg.DATASETS.TEST = ("dataset_valid",)
cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False
cfg.DATALOADER.NUM_WORKERS = NUM_WORKERS

# (optional) use a run-specific OUTPUT_DIR to avoid mixing artifacts
# cfg.OUTPUT_DIR = f"./output/skinpan_{RUN_ID}"

cfg.SOLVER.IMS_PER_BATCH = IMS_PER_BATCH

# Common input sizes (adjust to your setup)
cfg.INPUT.IMAGE_SIZE = 640
cfg.INPUT.MIN_SIZE_TEST = 640
cfg.INPUT.MAX_SIZE_TEST = 640

# -------------------------------
# Auto-calc iterations (based on the *current* random split)
# -------------------------------
dataset_dicts = DatasetCatalog.get(cfg.DATASETS.TRAIN[0])
dataset_size = len(dataset_dicts)
if dataset_size == 0:
    raise RuntimeError("Ephemeral train split is empty. Check filtering or annotations.")

iters_per_epoch = max(1, dataset_size // IMS_PER_BATCH)
cfg.SOLVER.MAX_ITER = NUM_EPOCHS * iters_per_epoch
cfg.TEST.EVAL_PERIOD = iters_per_epoch
cfg.SOLVER.CHECKPOINT_PERIOD = iters_per_epoch

# LR schedule steps every EPOCHS_PER_STEP epochs
cfg.SOLVER.STEPS = [
    step for step in range(
        iters_per_epoch * EPOCHS_PER_STEP,
        cfg.SOLVER.MAX_ITER,
        iters_per_epoch * EPOCHS_PER_STEP
    )
]
cfg.SOLVER.GAMMA = 0.5

# Ensure output directory exists
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

# -------------------------------
# Train
# -------------------------------
trainer = MyTrainer(cfg)
trainer.resume_or_load(resume=True)
trainer.train()