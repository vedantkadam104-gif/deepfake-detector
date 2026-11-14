# DeepShield AI — Image & Video Deepfake Detection System

An end-to-end Deepfake Forensic Detection Studio built with **PyTorch**, **EfficientNet-B0 + 2D Radial FFT Frequency Fusion**, **4-View Test-Time Augmentation (TTA)**, **Grad-CAM / ELA / FFT Visual Forensics**, and a **FastAPI** web interface.

---

## Verified Benchmark Results

Trained on **10,312** images (`1,289` validation, `1,289` held-out test) on an **NVIDIA GeForce RTX 3050 (6 GB Laptop GPU)** using FP16 Mixed Precision and validated decision threshold calibration (`0.885`):

| Metric | Single-Crop | 4x Test-Time Augmentation (TTA) |
| :--- | :---: | :---: |
| **Test Accuracy** | **96.20%** | **96.28%** (`1,241 / 1,289`) |
| **ROC-AUC** | **0.9929** | **0.9947** |
| **Real Recall / Precision** | `95%` / `96%` | `95%` / `97%` |
| **Fake Recall / Precision** | `97%` / `96%` | `97%` / `96%` |
| **Confusion Matrix** `[Real, Fake]` | `[[561, 28], [21, 679]]` | `[[560, 29], [19, 681]]` |

---

## Architecture Highlights

1. **Automated Face + Blending Seam Extraction (`src/face_pipeline.py`)**
   - Detects faces via OpenCV Haar Cascades and crops with a **+22% margin** around the forehead, cheeks, and jawline so face-swap blending artifacts are preserved (automatically skips cropping if an image is already a tightly cropped portrait).
2. **Spatial + Frequency Hybrid Model (`src/model.py`)**
   - **`EfficientNet-B0` Spatial Backbone (`1,280` dims):** Captures skin textures, asymmetrical facial structures, and edge defects.
   - **`RadialFFTExtractor` (`16` dims):** Computes 2D Fast Fourier Transform (`torch.fft.fft2` in `float32`) radial profiles on the GPU to spot invisible GAN/diffusion upsampling grids.
3. **Robust Preprocessing & Compression Immunity (`src/transforms.py`, `src/dataset.py`)**
   - Explicit locked label mapping (`real: 0`, `fake: 1` — preventing alphabetical `ImageFolder` inversion).
   - `RandomJPEGCompression(quality=55..95)` during training for resilience against WhatsApp/social media recompression.
   - 4-view TTA (normal, horizontal flip, 92% tight crop, tight crop + flip).
4. **Validation-Only Threshold Calibration (`tune_threshold.py`)**
   - Sweeps decision thresholds exclusively on the validation split (`0.885` saved in `checkpoints/threshold.json`) and piecewise-calibrates output probabilities around `0.50` in `src/detector.py`.
5. **Visual Forensic Studio (`app.py`, `src/gradcam.py`, `src/forensics.py`, `static/`)**
   - Supports **Images** and **Videos** (16-frame timeline sampling + top-k suspicious frame aggregation).
   - Generates 4 visual panels: **Face Bounding Box**, **Grad-CAM Heatmap**, **2D FFT Spectrum**, and **JPEG Error Level Analysis (ELA)**.

---

## Quick Start (Web Studio)

```powershell
# 1. Create virtual environment & install dependencies
uv venv --python 3.12 .venv
uv pip install --python .\.venv\Scripts\python.exe torch torchvision --index-url https://download.pytorch.org/whl/cu124
uv pip install --python .\.venv\Scripts\python.exe -r requirements.txt

# 2. Launch the FastAPI server (uses included checkpoints/best_model.pt)
.\.venv\Scripts\uvicorn.exe app:app --host 127.0.0.1 --port 8000
```
Open **`http://127.0.0.1:8000`** in your browser.

---

## Train & Evaluate on Your Own Dataset

```powershell
# 1. Inspect dataset folders, class balance, and image integrity
.\.venv\Scripts\python.exe check_dataset.py --data-dir "PATH_TO_DATASET"

# 2. Two-phase training (2 warmup + 10 fine-tuning epochs, FP16 + EMA)
.\.venv\Scripts\python.exe train.py --data-dir "PATH_TO_DATASET" --epochs 12 --batch-size 32 --grad-accum 2

# 3. Calibrate decision threshold on validation split
.\.venv\Scripts\python.exe tune_threshold.py --data-dir "PATH_TO_DATASET"

# 4. Evaluate on held-out test set
.\.venv\Scripts\python.exe evaluate.py --data-dir "PATH_TO_DATASET" --split test
```

---

## Interview Guide
Includes **[DeepShield_AI_Interview_Guide.pdf](./DeepShield_AI_Interview_Guide.pdf)** — a plain-English architecture walkthrough, difficulties-faced summary, terminology cheat sheet, and 15 recruiter & technical Q&As.
