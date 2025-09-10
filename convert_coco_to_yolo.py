from pathlib import Path
import shutil
import yaml
from tqdm import tqdm
from pycocotools.coco import COCO


def convert_coco_to_yolo(coco_json_path, image_dir, output_label_dir, output_img_dir):
    # Load COCO annotations
    coco = COCO(coco_json_path)
    
    img_ids = coco.getImgIds()
    class_id_map = {1: 0}  # Map category_id 1 to YOLO class 0
    
    output_label_dir.mkdir(parents=True, exist_ok=True)
    output_img_dir.mkdir(parents=True, exist_ok=True)
    
    for img_id in tqdm(img_ids, desc="Processing images"):
        img_info = coco.loadImgs(img_id)[0]
        img_width = img_info['width']
        img_height = img_info['height']
        img_filename = img_info['file_name']
        
        src_img_path = image_dir / img_filename
        dst_img_path = output_img_dir / img_filename
        
        if src_img_path.exists():
            shutil.copy2(src_img_path, dst_img_path)
        else:
            print(f"Warning: Image not found at {src_img_path}")
            continue
        
        ann_ids = coco.getAnnIds(imgIds=img_id)
        annotations = coco.loadAnns(ann_ids)

        yolo_lines = []
        for ann in annotations:
            class_id = class_id_map.get(ann['category_id'], 0)

            # COCO segmentation can be RLE or polygons. We only handle polygons.
            if not isinstance(ann["segmentation"], list):
                continue  # skip RLE masks

            for seg in ann["segmentation"]:
                if len(seg) < 6:
                    continue  # skip invalid polygons

                # Normalize coordinates
                normalized_points = []
                for i in range(0, len(seg), 2):
                    x = seg[i] / img_width
                    y = seg[i + 1] / img_height
                    normalized_points.extend([f"{x:.6f}", f"{y:.6f}"])

                yolo_line = f"{class_id} " + " ".join(normalized_points)
                yolo_lines.append(yolo_line)
    
        # for ann in annotations:
            # x_min, y_min, width, height = ann['bbox']
            # center_x = (x_min + width / 2) / img_width
            # center_y = (y_min + height / 2) / img_height
            # norm_width = width / img_width
            # norm_height = height / img_height
            
            # class_id = class_id_map.get(ann['category_id'], 0)
            # yolo_lines.append(f"{class_id} {center_x:.6f} {center_y:.6f} {norm_width:.6f} {norm_height:.6f}")
        
        label_filename = img_filename.rsplit('.', 1)[0] + '.txt'
        label_path = output_label_dir / label_filename
        label_path.write_text('\n'.join(yolo_lines))


# Define paths
coco_root = Path("/home/jovyan/nfs/mgatti/datasets/SKINPAN")
yolo_root = coco_root / "yolo"

# Create directory structure
(yolo_root / "images/train").mkdir(parents=True, exist_ok=True)
(yolo_root / "images/val").mkdir(parents=True, exist_ok=True)
(yolo_root / "labels/train").mkdir(parents=True, exist_ok=True)
(yolo_root / "labels/val").mkdir(parents=True, exist_ok=True)

# Convert train set
convert_coco_to_yolo(
    coco_json_path=coco_root / "annotations/instances_train.json",
    image_dir=coco_root / "train",
    output_label_dir=yolo_root / "labels/train",
    output_img_dir=yolo_root / "images/train"
)

# Convert validation set
convert_coco_to_yolo(
    coco_json_path=coco_root / "annotations/instances_valid.json",
    image_dir=coco_root / "train",
    output_label_dir=yolo_root / "labels/val",
    output_img_dir=yolo_root / "images/val"
)

# Convert test set
convert_coco_to_yolo(
    coco_json_path=coco_root / "annotations/instances_test.json",
    image_dir=coco_root / "test",
    output_label_dir=yolo_root / "labels/test",
    output_img_dir=yolo_root / "images/test"
)

# Save YOLO dataset YAML config
yolo_yaml = {
    "names": ["lesion"],
    "nc": 1,
    "path": str(yolo_root),
    "train": "images/train",
    "val": "images/val",
    "test": "images/test",
}

yaml_path = root / "yolo/skin_lesion.yaml"
with open(yaml_path, "w") as f:
    yaml.dump(yolo_yaml, f, default_flow_style=False)

print(f"YOLO config YAML saved to: {yaml_path}")
print("Conversion complete! All images copied (no symbolic links used).")