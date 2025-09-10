import argparse
import cv2
import numpy as np
import os
import supervision as sv
import sys
import yaml
import warnings

from tqdm import tqdm
from pathlib import Path

# Suppress warnings early
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# YOLO
from ultralytics import YOLO

# MaskDINO / Detectron2
sys.path.append("MaskDINO")
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from maskdino.config import add_maskdino_config
from detectron2.projects.deeplab import add_deeplab_config


def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    boxBArea = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)


def visualize_predictions(model_type, model_config, output_dir, iou_threshold=0.5,  annotation_type="bbox"):
    output_dir = Path(output_dir)
    correct_dir = output_dir / "correct_detections"
    partial_dir = output_dir / "partially_correct_detections"
    incorrect_dir = output_dir / "incorrect_detections"
    correct_dir.mkdir(parents=True, exist_ok=True)
    partial_dir.mkdir(parents=True, exist_ok=True)
    incorrect_dir.mkdir(parents=True, exist_ok=True)

    count_correct, count_partial, count_incorrect = 0, 0, 0

    if model_type == "maskdino":
        predictor = model_config["predictor"]
        dataset_name = model_config["dataset_name"]
        dataset_dicts = DatasetCatalog.get(dataset_name)
        image_data = [(Path(d["file_name"]), d["annotations"]) for d in dataset_dicts]

    elif model_type == "yolo":
        model = YOLO(model_config["model_path"])
        with open(model_config["yaml_path"]) as f:
            data_cfg = yaml.safe_load(f)

        test_image_dir = Path(data_cfg["path"]) / data_cfg["test"]
        label_dir = Path(data_cfg["path"]) / "labels/test"
        image_paths = list(test_image_dir.glob("*.jpg")) + list(test_image_dir.glob("*.png"))
        image_data = [(img_path, label_dir / (img_path.stem + ".txt")) for img_path in image_paths]

    for item in tqdm(image_data, desc="Processing test images"):
        if model_type == "maskdino":
            image_path, annotations = item
        else:
            image_path, label_path = item

        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Could not load image {image_path}")
            continue

        gt_boxes, gt_classes = [], []
        if model_type == "maskdino":
            for ann in annotations:
                x, y, w, h = ann["bbox"]
                gt_boxes.append([x, y, x+w, y+h])
                gt_classes.append(ann["category_id"])
        else:
            if not label_path.exists():
                continue
            with open(label_path) as f:
                for line in f:
                    cls, x, y, w, h = map(float, line.strip().split())
                    H, W = image.shape[:2]
                    x1 = (x - w/2) * W
                    y1 = (y - h/2) * H
                    x2 = (x + w/2) * W
                    y2 = (y + h/2) * H
                    gt_boxes.append([x1, y1, x2, y2])
                    gt_classes.append(int(cls))

        if model_type == "maskdino":
            outputs = predictor(image)
            instances = outputs["instances"].to("cpu")
            instances = outputs["instances"][(outputs["instances"].scores > 0.5).nonzero().squeeze(1)].to("cpu")
            pred_boxes = instances.pred_boxes.tensor.numpy()
            pred_classes = instances.pred_classes.numpy()
            pred_scores = instances.scores.numpy()
        else:
            results = model.predict(str(image_path), conf=0.5)
            pred_boxes, pred_classes, pred_scores = [], [], []
            for r in results:
                for box in r.boxes:
                    pred_boxes.append(box.xyxy[0].tolist())
                    pred_classes.append(int(box.cls))
                    pred_scores.append(float(box.conf))

        vis_image = image.copy()
        matched_gt_indices = set()

        # Draw GT bboxes
        gt_detections = sv.Detections(xyxy=np.array(gt_boxes), class_id=np.array(gt_classes))
        gt_annotator = sv.BoxAnnotator(color=sv.Color.GREEN)
        vis_image = gt_annotator.annotate(vis_image, gt_detections)

        for i, pred_box in enumerate(pred_boxes):
            pred_box = np.array(pred_box)
            cls_id = pred_classes[i]
            conf = pred_scores[i]

            max_iou, best_match_idx = 0, -1
            for j, gt_box in enumerate(gt_boxes):
                if j in matched_gt_indices:
                    continue
                iou = calculate_iou(pred_box, gt_box)
                if iou > max_iou:
                    max_iou = iou
                    best_match_idx = j

            color = sv.Color.RED

            if max_iou >= iou_threshold:
                if cls_id == gt_classes[best_match_idx]:
                    color = sv.Color.YELLOW
                    matched_gt_indices.add(best_match_idx)
                else:
                    color = sv.Color.BLUE

            det = sv.Detections(
                xyxy=np.array([pred_box]),
                class_id=np.array([cls_id]),
                confidence=np.array([conf])
            )

            annotator = sv.BoxAnnotator(color=color)
            vis_image = annotator.annotate(vis_image, det)

        num_gt = len(gt_boxes)
        num_matched = len(matched_gt_indices)

        if num_matched == num_gt:
            save_path = correct_dir / image_path.name
            count_correct += 1
        elif num_matched > 0:
            save_path = partial_dir / image_path.name
            count_partial += 1
        else:
            save_path = incorrect_dir / image_path.name
            count_incorrect += 1

        cv2.imwrite(str(save_path), vis_image)

    total = count_correct + count_partial + count_incorrect
    print("\nVisualizations saved to:")
    print(f"   Correct:   {correct_dir} ({count_correct})")
    print(f"   Partial:   {partial_dir} ({count_partial})")
    print(f"   Incorrect: {incorrect_dir} ({count_incorrect})")
    print(f"\nSummary: {total} images processed\n")


# === Entry Point ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize predictions using MaskDINO or YOLO.")
    parser.add_argument("--model", choices=["yolo", "maskdino"], required=True, help="Model to use for prediction")
    parser.add_argument("--annotation", choices=["bbox", "segm"], default="bbox", help="Object detection or instance segmentation")

    args = parser.parse_args()
    annotation_type = args.annotation
    model_type = args.model

    if model_type == "maskdino":
        dataset_name = "lesion_test"
        register_coco_instances(
            name=dataset_name,
            metadata={},
            json_file="/home/jovyan/nfs/mgatti/datasets/SKINPAN/annotations/instances_test.json",
            image_root="/home/jovyan/nfs/mgatti/datasets/SKINPAN/test/"
        )

        cfg = get_cfg()
        add_deeplab_config(cfg)
        add_maskdino_config(cfg)
        cfg.merge_from_file("MaskDINO/configs/coco/instance-segmentation/swin/maskdino_R50_bs16_50ep_4s_dowsample1_2048.yaml")
        cfg.MODEL.WEIGHTS = "output/best_model.pth"
        cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 1
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        cfg.DATASETS.TEST = (dataset_name,)
        predictor = DefaultPredictor(cfg)

        visualize_predictions(
            model_type="maskdino",
            model_config={"predictor": predictor, "dataset_name": dataset_name},
            output_dir="predictions_visualization/maskdino",
            annotation_type=annotation_type
        )

    elif model_type == "yolo":
        visualize_predictions(
            model_type="yolo",
            model_config={
                "model_path": "runs/detect/skin_lesion_yolo8x-seg/weights/best.pt",
                "yaml_path": "/home/jovyan/nfs/mgatti/datasets/SKINPAN/yolo/skin_lesion.yaml"
            },
            output_dir="predictions_visualization/yolo",
            annotation_type=annotation_type
        )