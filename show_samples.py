import argparse
import cv2
import json
import numpy as np
import os
import random

from pathlib import Path
from pycocotools.coco import COCO
from pycocotools import mask as mask_utils


def save_coco_ground_truth_samples(annotation_path, image_dir, output_dir, num_samples=5, seed=42, specific_id=None, specific_filename=None, annotation_count_filter=None, localization_filter=None,
    metadata_path=None):
    os.makedirs(output_dir, exist_ok=True)

    coco = COCO(annotation_path)
    image_ids = coco.getImgIds()

    selected_ids = []

    # Load metadata (only if needed)
    metadata = {}
    if localization_filter and metadata_path:
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
    
    if specific_id is not None:
        selected_ids = [specific_id]
    elif specific_filename is not None:
        selected_ids = [img['id'] for img in coco.loadImgs(image_ids) if img['file_name'] == specific_filename]
    elif metadata:
        filtered_ids = []
        for img_id, meta in metadata.items():
            if localization_filter and meta.get("localization_general") != localization_filter:
                continue
            if annotation_count_filter:
                count = meta.get("annotation_count", 0)
                min_count, max_count = annotation_count_filter
                if not (min_count <= count <= max_count):
                    continue
            filtered_ids.append(int(img_id))

        image_ids = list(set(image_ids) & set(filtered_ids))
        random.seed(seed)
        if num_samples is None:
            selected_ids = image_ids  # all filtered
        else:
            selected_ids = random.sample(image_ids, min(num_samples, len(image_ids)))
    else:
        random.seed(seed)
        if num_samples is None:
            selected_ids = image_ids  # all
        else:
            selected_ids = random.sample(image_ids, min(num_samples, len(image_ids)))

    for img_id in selected_ids:
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(image_dir, img_info["file_name"])
        image = cv2.imread(img_path)

        if image is None:
            print(f"Warning: could not read image: {img_path}")
            continue

        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        height, width = image.shape[:2]

        for ann in anns:
            if isinstance(ann["segmentation"], (list, dict)):
                rle = (
                    mask_utils.frPyObjects(ann["segmentation"], height, width)
                    if isinstance(ann["segmentation"], list)
                    else ann["segmentation"]
                )
                rle = mask_utils.merge(rle) if isinstance(rle, list) else rle
                mask = mask_utils.decode(rle)

                mask_uint8 = (mask * 255).astype(np.uint8)
                kernel = np.ones((5, 5), np.uint8)
                dilated_mask = cv2.dilate(mask_uint8, kernel, iterations=1)

                contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(image, contours, -1, (0, 255, 0), thickness=2)

        base_filename = os.path.splitext(img_info["file_name"])[0]
        output_filename = f"{base_filename}_gt.jpg"
        output_path = os.path.join(output_dir, output_filename)
        cv2.imwrite(output_path, image)
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-path", type=str, required=True)
    parser.add_argument("--image-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="./output/gt_samples")
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--specific-id", type=int, default=None)
    parser.add_argument("--specific-filename", type=str, default=None)
    parser.add_argument("--metadata-path", type=str, default=None)
    parser.add_argument("--localization", type=str, default=None)
    parser.add_argument("--min-count", type=int, default=None)
    parser.add_argument("--max-count", type=int, default=None)

    args = parser.parse_args()

    annotation_count_filter = None
    if args.min_count is not None and args.max_count is not None:
        annotation_count_filter = (args.min_count, args.max_count)

    save_coco_ground_truth_samples(
        annotation_path=Path(args.annotation_path),
        image_dir=Path(args.image_dir),
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        seed=args.seed,
        specific_id=args.specific_id,
        specific_filename=args.specific_filename,
        metadata_path=args.metadata_path,
        localization_filter=args.localization,
        annotation_count_filter=annotation_count_filter
    )

"""
python show_samples.py \
  --annotation-path /home/jovyan/nfs/mgatti/datasets/SKINPAN/coco/annotations/instances_train.json \
  --image-dir /home/jovyan/nfs/mgatti/datasets/SKINPAN/coco/train \
  --output-dir ./output/gt_samples/ \
  --specific-id 1068 \
  --seed 42 \
  --metadata-path /home/jovyan/nfs/mgatti/datasets/SKINPAN/metadata.json

python show_samples.py \
  --annotation-path /home/jovyan/nfs/mgatti/datasets/SKINPAN/coco/annotations/instances_train.json \
  --image-dir /home/jovyan/nfs/mgatti/datasets/SKINPAN/coco/train \
  --output-dir ./output/gt_samples/back \
  --seed 42 \
  --localization back \
  --min-count 2 \
  --metadata-path /home/jovyan/nfs/mgatti/datasets/SKINPAN/metadata.json

python show_samples.py \
  --annotation-path /home/jovyan/nfs/mgatti/datasets/SKINPAN/coco/annotations/instances_train.json \
  --image-dir /home/jovyan/nfs/mgatti/datasets/SKINPAN/coco/train \
  --output-dir ./output/gt_samples/chest \
  --seed 42 \
  --localization chest \
  --min-count 2 \
  --metadata-path /home/jovyan/nfs/mgatti/datasets/SKINPAN/metadata.json

python show_samples.py \
  --annotation-path /home/jovyan/nfs/mgatti/datasets/SKINPAN/coco/annotations/instances_train.json \
  --image-dir /home/jovyan/nfs/mgatti/datasets/SKINPAN/coco/train \
  --output-dir ./output/gt_samples/arm \
  --seed 42 \
  --localization arm \
  --metadata-path /home/jovyan/nfs/mgatti/datasets/SKINPAN/metadata.json
"""