import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
sys.path.append("MaskDINO")

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import torch

from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.data.transforms import AugInput
from detectron2.engine import DefaultTrainer, HookBase
from detectron2.engine.hooks import BestCheckpointer
from detectron2.evaluation import COCOEvaluator
from detectron2.projects.deeplab import add_deeplab_config

from maskdino.config import add_maskdino_config
from train_net import Trainer


class MyTrainer(Trainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, False, output_folder)

    def build_hooks(self):
        hooks = super().build_hooks()

        # Remove PeriodicCheckpointer (saves every N iters)
        hooks = [h for h in hooks if h.__class__.__name__ != "PeriodicCheckpointer"]
        
        # Add BestCheckpointer
        best_checkpointer = BestCheckpointer(
            cfg.TEST.EVAL_PERIOD,
            self.checkpointer,
            "bbox/AP",
            "max",
            file_prefix="best_model"
        )

        hooks.insert(-1, best_checkpointer)  # Insert before evaluation hook

        # PeriodicCheckpointer will still save 'last_checkpoint' every N iters
        # cfg.SOLVER.CHECKPOINT_PERIOD is used for that

        return hooks


# -------------------------------
# Configuration Parameters
# -------------------------------
NUM_EPOCHS = 50
EPOCHS_PER_STEP = 5
IMS_PER_BATCH = 4
NUM_WORKERS = 4
CONFIG_PATH = "MaskDINO/configs/coco/instance-segmentation/swin/maskdino_R50_bs16_50ep_4s_dowsample1_2048.yaml"
PRETRAINED_WEIGHTS = "maskdino_swinl_50ep_300q_hid2048_3sd1_instance_maskenhanced_mask52.3ap_box59.0ap.pth"
DATASET_PATH = "/home/jovyan/nfs/mgatti/datasets/SKINPAN/"

torch.cuda.empty_cache()

# -------------------------------
# Register COCO-Style Dataset
# -------------------------------
for split in ["train", "valid"]:
    register_coco_instances(
        name=f"dataset_{split}",
        metadata={},
        json_file=f"{DATASET_PATH}/annotations/instances_{split}.json",
        image_root=f"{DATASET_PATH}/train"
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
cfg.SOLVER.IMS_PER_BATCH = IMS_PER_BATCH
cfg.INPUT.IMAGE_SIZE = 640
cfg.INPUT.MIN_SIZE_TEST = 640
cfg.INPUT.MAX_SIZE_TEST = 640
# cfg.INPUT.MAX_SIZE_TEST = 1138

# -------------------------------
# Auto-calculate Training Iterations
# -------------------------------
dataset_dicts = DatasetCatalog.get(cfg.DATASETS.TRAIN[0])
dataset_size = len(dataset_dicts)
iters_per_epoch = dataset_size // IMS_PER_BATCH
cfg.SOLVER.MAX_ITER = NUM_EPOCHS * iters_per_epoch
cfg.TEST.EVAL_PERIOD = iters_per_epoch
cfg.SOLVER.CHECKPOINT_PERIOD = iters_per_epoch

# Learning rate step schedule
cfg.SOLVER.STEPS = [
    step for step in range(
        iters_per_epoch * EPOCHS_PER_STEP,
        cfg.SOLVER.MAX_ITER,
        iters_per_epoch * EPOCHS_PER_STEP
    )
]

cfg.SOLVER.GAMMA = 0.5  # Learning rate decay factor

# Ensure output directory exists
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

# -------------------------------
# Start Training
# -------------------------------
trainer = MyTrainer(cfg)
trainer.resume_or_load(resume=True)
trainer.train()