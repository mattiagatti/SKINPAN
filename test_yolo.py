from pathlib import Path
from ultralytics import YOLO
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


# === CONFIGURATION ===
GPU = 0
model_path = Path("./runs/segment/skin_lesion_yolo11x-seg/weights/best.pt")
data_yaml = Path("/home/jovyan/nfs/mgatti/datasets/SKINPAN/yolo/skin_lesion.yaml")
gt_json = Path("/home/jovyan/nfs/mgatti/datasets/SKINPAN/annotations/instances_test.json")
pred_json = Path("./runs/segment/val/predictions.json")


# === VALIDATE PATHS ===
required_paths = {
    "YOLO model": model_path,
    "Data YAML": data_yaml,
    "COCO ground truth": gt_json,
}

for name, path in required_paths.items():
    if not path.exists():
        raise FileNotFoundError(f"{name} not found: {path}")

# === EVALUATE YOLO11 ===
print("Evaluating YOLO11 model...")
model = YOLO(str(model_path))

results = model.val(
    data=str(data_yaml),
    split="test",
    save_json=True,
    save_txt=True,
    plots=True,
    device=f"cuda:{GPU}"
)

print("\nYOLO11 Raw Results:")
for k, v in results.results_dict.items():
    print(f"  {k}: {v:.6f}")

# === VERIFY predictions.json ===
if not pred_json.exists():
    raise FileNotFoundError(f"Predicted results not found: {pred_json}")

# === COCO-STYLE EVALUATION ===
print("\nRunning COCO-style evaluation...")
coco_gt = COCO(str(gt_json))
coco_dt = coco_gt.loadRes(str(pred_json))

for iou_type in ['bbox', 'segm']:
    print(f"\n--- COCO Evaluation ({iou_type.upper()}) ---")
    coco_eval = COCOeval(coco_gt, coco_dt, iouType=iou_type)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()