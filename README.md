# 🧠 Training Workflow for Mask DINO & YOLO

This repository provides code for training and inference on the SKINPAN dataset using:

- [**Mask DINO**](https://github.com/IDEA-Research/MaskDINO): A state-of-the-art transformer-based model for instance segmentation, built on [Detectron2](https://github.com/facebookresearch/detectron2).
- [**YOLO (Ultralytics)**](https://www.ultralytics.com/yolo): A real-time object detection model, using the [Ultralytics](https://github.com/ultralytics/ultralytics) framework.

---

## 🚀 Environment Setup

Follow the steps below to set up your training environment.

### 1. Create and Activate a Conda Environment

```bash
conda create -n dino310 python=3.10 -y
conda activate dino310
```

### 2. Install Python Dependencies

Install PyTorch and other required packages:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install ninja opencv-python supervision ultralytics
```

> 💡 Make sure this is run inside the `dino310` Conda environment.

### 3. Install System Dependencies

Ensure build tools like `gcc` and `g++` (≥5.4) are installed:

```bash
sudo apt-get update
sudo apt-get install -y build-essential gcc g++ python3-dev
```

### 4. Install CUDA Toolkit

Install a CUDA version compatible with your GPU and PyTorch version. Example:

```bash
conda install -c nvidia/label/cuda-12.1.1 cuda-toolkit
```

---

## 📦 Project Setup

### 5. Install Detectron2

```bash
git clone https://github.com/facebookresearch/detectron2.git
cd detectron2
pip install . --no-build-isolation
cd ..
```

### 6. Clone & Build MaskDINO

```bash
git clone https://github.com/IDEA-Research/MaskDINO.git
cd MaskDINO
pip install -r requirements.txt

# Compile CUDA extensions
cd maskdino/modeling/pixel_decoder/ops
sh make.sh
cd ../../../../..
```

---

## 📥 Pre-trained Weights

Download pre-trained weights from Kaggle:

```bash
pip install kaggle

# Ensure ~/.kaggle/kaggle.json contains your API key
kaggle datasets download -d resimasss/maskdino-weights --unzip
```

---

## 🏋️ Training

To plot some ground truths for inspection you can use:

```bash
python show_samples.py \
  --annotation-path /path/to/SKINPAN/coco/annotations/instances_train.json \
  --image-dir /path/to/SKINPAN/coco/train \
  --output-dir ./output/gt_samples \
  --num-samples 5 \
  --seed 42 \
  --localization back \
  --min-count 2 \
  --max-count 5 \
  --metadata-path /path/to/SKINPAN/metadata.json
```

### Mask DINO

```bash
python train_maskdino.py
```

### YOLO

```bash
python train_yolo.py
```

---

## 🔍 Inference

### Mask DINO

```bash
python visualize.py --model maskdino
```

### YOLO

```bash
python visualize.py --model yolo
```

---

## 📝 Notes

- You may need to update dataset registration paths in the scripts.
- For custom datasets, refer to [Detectron2's dataset registration guide](https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html).
- Verify that your GPU drivers, PyTorch, and CUDA versions are compatible.

---

## 📄 License

This project is licensed under the **Apache 2.0 License**, inherited from:

- [Mask DINO](https://github.com/IDEA-Research/MaskDINO)
- [Detectron2](https://github.com/facebookresearch/detectron2)
