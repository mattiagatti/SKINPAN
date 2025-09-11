import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
import torch
import warnings
from pathlib import Path

# Silence warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Add MaskDINO and Detectron2 to path
sys.path.append("MaskDINO")

from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.projects.deeplab import add_deeplab_config
from maskdino.config import add_maskdino_config

# -------------------------------
# Configuration
# -------------------------------
CONFIG_PATH = "MaskDINO/configs/coco/instance-segmentation/swin/maskdino_R50_bs16_50ep_4s_dowsample1_2048.yaml"
BEST_MODEL_PATH = "./output/best_model.pth"
DATASET_PATH = "./SKINPAN/"
OUTPUT_DIR = "./output/test"
BATCH_SIZE = 4
NUM_WORKERS = 4
DATASET_NAME = "dataset_test"


class MyTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, False, output_folder)

# -------------------------------
# Dataset Registration
# -------------------------------
register_coco_instances(
    name=DATASET_NAME,
    metadata={},
    json_file=f"{DATASET_PATH}/annotations/instances_test.json",
    image_root=f"{DATASET_PATH}/test"
)

# -------------------------------
# Load Config
# -------------------------------
cfg = get_cfg()
add_deeplab_config(cfg)
add_maskdino_config(cfg)
cfg.merge_from_file(CONFIG_PATH)

cfg.SOLVER.CLIP_GRADIENTS.ENABLED = False
cfg.MODEL.WEIGHTS = BEST_MODEL_PATH
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 1
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

cfg.DATASETS.TRAIN = ("dataset_test",)
cfg.DATASETS.TEST = (DATASET_NAME,)
cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False
cfg.DATALOADER.NUM_WORKERS = NUM_WORKERS
cfg.SOLVER.IMS_PER_BATCH = BATCH_SIZE
cfg.OUTPUT_DIR = OUTPUT_DIR
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

# -------------------------------
# Run Evaluation
# -------------------------------
if __name__ == "__main__":
    print("Running evaluation on best model...")

    trainer = MyTrainer(cfg)
    trainer.resume_or_load(resume=False)

    trainer.test(cfg, trainer.model)