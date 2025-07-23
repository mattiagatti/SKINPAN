import os
from ultralytics import YOLO
from ultralytics.utils.checks import check_yaml

# Configuration
GPU = 0
MODEL_NAME = "yolo11x-seg.pt"
DATA_YAML = "/home/jovyan/nfs/mgatti/datasets/skin_lesion_dataset/yolo/skin_lesion.yaml"


def train_yolo():
    # Load YOLO11x segmentation model
    model = YOLO(MODEL_NAME)

    print("\nStarting training...")
    results = model.train(
        data=DATA_YAML,
        imgsz=640,
        epochs=100,
        batch=16,
        device=f"cuda:{GPU}",
        val=True,
        save=True,
        save_period=-1,
        exist_ok=True,
        verbose=True,
        task='segment',
        name=f"skin_lesion_{MODEL_NAME.replace('.pt', '')}"
    )

    print("\nStarting post-training validation...")
    val_results = model.val(
        data=DATA_YAML,
        device=f"cuda:{GPU}",
        save_json=True,
        save_hybrid=True,
        save_txt=True,
        plots=True
    )

    return results, val_results


if __name__ == "__main__":
    # Validate the dataset config YAML
    check_yaml(DATA_YAML)

    # Run training and validation
    train_results, val_results = train_yolo()

    # Report results directory
    print(f"\nTraining results saved to: {train_results.save_dir}")
    if val_results:
        print(f"Validation results saved to: {val_results.save_dir}")

    # Check for validation image outputs
    val_img_dir = os.path.join(train_results.save_dir, 'val')
    if os.path.exists(val_img_dir):
        print(f"\nFound validation images in: {val_img_dir}")
        print("Contents:", os.listdir(val_img_dir))
    else:
        print("\nWarning: Validation images directory not found!")