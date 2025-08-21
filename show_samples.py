import argparse
import cv2
import json
import numpy as np
import os
import random
from pathlib import Path
from typing import Iterable, List, Optional

from pycocotools.coco import COCO
from pycocotools import mask as mask_utils

# ─────────────────────────────────────────────
# Mask outline helpers (inside | outside | center)
# ─────────────────────────────────────────────

def _mask_outline(binary_mask: np.ndarray, thickness: int = 2, mode: str = "outside") -> np.ndarray:
    """Return a boolean outline of the binary mask with placement control.
    - mode='inside': outline is entirely inside the original mask
    - mode='outside': outline is entirely outside the original mask
    - mode='center': outline straddles boundary (half in, half out)
    """
    if thickness < 1:
        thickness = 1
    m = (binary_mask > 0).astype(np.uint8)

    # Use a simple square kernel; thickness controls iteration count
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    if mode == "inside":
        eroded = cv2.erode(m, k, iterations=thickness)
        ring = cv2.subtract(m, eroded)  # inside rim
        return ring.astype(bool)

    if mode == "outside":
        dilated = cv2.dilate(m, k, iterations=thickness)
        ring = cv2.subtract(dilated, m)  # outside rim
        return ring.astype(bool)

    # center: combine half outside + half inside
    out_iters = int(np.ceil(thickness / 2))
    in_iters = int(np.floor(thickness / 2))

    if out_iters > 0:
        dil = cv2.dilate(m, k, iterations=out_iters)
        rim_out = cv2.subtract(dil, m)
    else:
        rim_out = np.zeros_like(m)

    if in_iters > 0:
        ero = cv2.erode(m, k, iterations=in_iters)
        rim_in = cv2.subtract(m, ero)
    else:
        rim_in = np.zeros_like(m)

    ring = cv2.bitwise_or(rim_in, rim_out)
    return ring.astype(bool)


def draw_bbox_crisp(img, x1, y1, x2, y2, color=(0, 0, 255), t=3, mode="center"):
    """
    Draws a crisp (pixel-perfect) rectangle without anti-aliasing.
    mode: "center" | "inside" | "outside"
    """
    h, w = img.shape[:2]

    h, w = img.shape[:2]
    x1 = max(0, min(x1, w-1)); y1 = max(0, min(y1, h-1))
    x2 = max(0, min(x2, w-1)); y2 = max(0, min(y2, h-1))
    if x2 <= x1 or y2 <= y1:
        return img

    if mode == "outside":
        x1, y1 = max(0, x1 - t), max(0, y1 - t)
        x2, y2 = min(w - 1, x2 + t), min(h - 1, y2 + t)
    elif mode == "inside":
        x1, y1 = x1 + t, y1 + t
        x2, y2 = x2 - t, y2 - t
        if x2 <= x1 or y2 <= y1:
            # fallback: draw center-style thin
            mode = "center"
            t = 1

    # center (default) -> no coordinate adjustment

    # draw solid strips
    img[y1:y1+t, x1:x2] = color  # top
    img[y2-t:y2, x1:x2] = color  # bottom
    img[y1:y2, x1:x1+t] = color  # left
    img[y1:y2, x2-t:x2] = color  # right
    return img



def _parse_specific_ids(values: Optional[Iterable[str]]) -> List[int]:
    """Parse a list of ids from CLI. Accepts forms like:
    --specific-ids 12 34 56
    --specific-ids 12,34,56
    --specific-ids 12 34,56 78
    Returns a deduplicated list of ints preserving order.
    """
    if not values:
        return []
    seen = set()
    out: List[int] = []
    for v in values:
        for tok in str(v).replace(";", ",").split(","):
            tok = tok.strip()
            if not tok:
                continue
            try:
                i = int(tok)
            except ValueError:
                continue
            if i not in seen:
                seen.add(i)
                out.append(i)
    return out


def save_coco_ground_truth_samples(
    annotation_path,
    image_dir,
    output_dir,
    num_samples: Optional[int] = 5,
    seed: int = 42,
    specific_id: Optional[int] = None,
    specific_ids: Optional[List[int]] = None,
    specific_filename: Optional[str] = None,
    annotation_count_filter=None,
    localization_filter: Optional[str] = None,
    metadata_path: Optional[str] = None,
    show_bbox: bool = False,
    thickness: int = 3,
    mode: str = "outside",  # center|outside|inside
    bbox_style: str = "crisp",
):
    os.makedirs(output_dir, exist_ok=True)

    coco = COCO(str(annotation_path))
    image_ids = coco.getImgIds()

    selected_ids: List[int] = []

    # Load metadata (for filters and inpaint fallback)
    metadata = {}
    if metadata_path:
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

    if specific_ids:
        # user supplied a list of ids
        selected_ids = [i for i in specific_ids if i in image_ids]
        if not selected_ids:
            print("[Warning] None of the provided --specific-ids were found in COCO index.")
    elif specific_id is not None:
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
            try:
                filtered_ids.append(int(img_id))
            except Exception:
                continue

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
        imgs = coco.loadImgs(img_id)
        if not imgs:
            print(f"[Warning] Image id not found in annotations: {img_id}")
            continue
        img_info = imgs[0]
        img_path = os.path.join(str(image_dir), img_info["file_name"])
        image = cv2.imread(img_path)

        if image is None:
            print(f"[Warning] could not read image: {img_path}")
            continue

        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        # Fallback: if no annotations for this image, but metadata has 'inpainted_from',
        # use the annotations from that source image id instead.
        if (not anns) and metadata:
            meta_entry = metadata.get(str(img_id)) or metadata.get(int(img_id))
            if isinstance(meta_entry, dict) and "inpainted_from" in meta_entry:
                try:
                    src_id = int(meta_entry["inpainted_from"])
                    ann_ids_src = coco.getAnnIds(imgIds=src_id)
                    anns_src = coco.loadAnns(ann_ids_src)
                    if anns_src:
                        print(f"[Info] No anns for id {img_id}; using inpainted_from={src_id} annotations.")
                        anns = anns_src
                    else:
                        print(f"[Info] inpainted_from={src_id} also has no anns.")
                except Exception as _e:
                    print(f"[Warning] Invalid inpainted_from for id {img_id}: {meta_entry.get('inpainted_from')}")

        height, width = image.shape[:2]

        for ann in anns:
            if show_bbox:
                # COCO bbox: [x, y, w, h] (floats)
                x, y, w, h = ann.get("bbox", [0, 0, 0, 0])
                # robust rounding to cover full region
                x1 = int(np.floor(x)); y1 = int(np.floor(y))
                x2 = int(np.ceil(x + w)); y2 = int(np.ceil(y + h))
                # clamp to image bounds
                x1 = max(0, min(x1, width - 1)); y1 = max(0, min(y1, height - 1))
                x2 = max(0, min(x2, width - 1)); y2 = max(0, min(y2, height - 1))

                t = int(max(1, thickness))
                mode = (mode or "center").lower()
                if bbox_style == "cv2":
                    if mode == "outside":
                        x1o, y1o = x1 - t, y1 - t
                        x2o, y2o = x2 + t, y2 + t
                        cv2.rectangle(image, (x1o, y1o), (x2o, y2o),
                                      (0, 0, 255), thickness=2*t, lineType=cv2.LINE_8)
                    elif mode == "inside":
                        x1i, y1i = x1 + t, y1 + t
                        x2i, y2i = x2 - t, y2 - t
                        if x2i <= x1i or y2i <= y1i:
                            cv2.rectangle(image, (x1, y1), (x2, y2),
                                          (0, 0, 255), thickness=1, lineType=cv2.LINE_8)
                        else:
                            cv2.rectangle(image, (x1i, y1i), (x2i, y2i),
                                          (0, 0, 255), thickness=2*t, lineType=cv2.LINE_8)
                    else:  # center
                        cv2.rectangle(image, (x1, y1), (x2, y2),
                                      (0, 0, 255), thickness=t, lineType=cv2.LINE_8)
                else:  # crisp
                    image = draw_bbox_crisp(image, x1, y1, x2, y2, color=(0, 0, 255), t=t, mode=mode)
            else:
                if isinstance(ann.get("segmentation"), (list, dict)):
                    rle = (
                        mask_utils.frPyObjects(ann["segmentation"], height, width)
                        if isinstance(ann["segmentation"], list)
                        else ann["segmentation"]
                    )
                    rle = mask_utils.merge(rle) if isinstance(rle, list) else rle
                    mask = mask_utils.decode(rle)  # 0/1 uint8

                    # Build an outline ring with the same placement logic as bboxes
                    mode = (mode or "center").lower()
                    t = int(max(1, thickness))
                    ring = _mask_outline(mask, thickness=t, mode=mode)

                    # Paint ring with crisp pixels (no anti-aliasing)
                    image[ring] = (0, 255, 0)

        base_filename = os.path.splitext(img_info["file_name"])[0]
        suffix = "bbox" if show_bbox else "gt"
        output_filename = f"{base_filename}_{suffix}.jpg"
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
    parser.add_argument("--specific-ids", nargs="*", help="List of image ids (space/comma separated)")
    parser.add_argument("--specific-filename", type=str, default=None)
    parser.add_argument("--metadata-path", type=str, default=None)
    parser.add_argument("--localization", type=str, default=None)
    parser.add_argument("--min-ann-count", type=int, default=None)
    parser.add_argument("--max-ann-count", type=int, default=None)
    parser.add_argument("--show-bbox", action="store_true")
    parser.add_argument("--thickness", type=int, default=6, help="BBox line thickness in px")
    parser.add_argument("--bbox-style", type=str, default="crisp", choices=["cv2", "crisp"], help="Rendering style for bounding boxes")
    parser.add_argument("--mode", type=str, default="outside", choices=["center","outside","inside"], help="Stroke placement: center|outside|inside")

    args = parser.parse_args()

    annotation_count_filter = None
    if args.min_ann_count is not None and args.max_ann_count is not None:
        annotation_count_filter = (args.min_ann_count, args.max_ann_count)

    save_coco_ground_truth_samples(
        annotation_path=Path(args.annotation_path),
        image_dir=Path(args.image_dir),
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        seed=args.seed,
        specific_ids=_parse_specific_ids(args.specific_ids),
        specific_filename=args.specific_filename,
        metadata_path=args.metadata_path,
        localization_filter=args.localization,
        annotation_count_filter=annotation_count_filter,
        show_bbox=args.show_bbox,
        thickness=args.thickness,
        mode=args.mode,
        bbox_style=args.bbox_style
    )

"""
PAPER SAMPLES:
python show_samples.py \
  --annotation-path /home/jovyan/nfs/mgatti/datasets/SKINPAN/coco/annotations/instances_train.json \
  --image-dir /home/jovyan/nfs/mgatti/datasets/SKINPAN/coco/train \
  --output-dir ./output/gt_samples/ \
  --specific-ids 832,2660,5936,6359,7132,7469 \
  --metadata-path /home/jovyan/nfs/mgatti/datasets/SKINPAN/metadata.json

PAPER INPAINTING EXAMPLE:
python show_samples.py \
  --annotation-path /home/jovyan/nfs/mgatti/datasets/SKINPAN/coco/annotations/instances_train.json  \
  --image-dir /home/jovyan/nfs/mgatti/datasets/SKINPAN/coco/train \
  --output-dir ./output/gt_samples_bbox/ \
  --specific-ids 660,1351 \
  --metadata-path /home/jovyan/nfs/mgatti/datasets/SKINPAN/metadata.json \
  --show-bbox

SPECIFIC PART:
python show_samples.py \
  --annotation-path /home/jovyan/nfs/mgatti/datasets/SKINPAN/coco/annotations/instances_train.json \
  --image-dir /home/jovyan/nfs/mgatti/datasets/SKINPAN/coco/train \
  --output-dir ./output/gt_samples/chest \
  --seed 42 \
  --localization chest \
  --min-ann-count 2 \
  --metadata-path /home/jovyan/nfs/mgatti/datasets/SKINPAN/metadata.json
"""