## Installation

**Python 3.9+ and a CUDA-capable GPU are recommended.**

```bash
# 1. Install PyTorch for your CUDA version (example: CUDA 12.9)
pip install torch --index-url https://download.pytorch.org/whl/cu129

# 2. Install remaining dependencies
pip install -r requirements.txt
```

---

## Running Inference

### Option A — Shell script (quickest)

```bash
# Run on the bundled example image
bash src/stanford2d3d_predict.sh

# Run on your own image
IMG=/path/to/your/panorama.png \
OUTDIR=/path/to/output/ \
bash src/stanford2d3d_predict.sh
```

### Option B — Python directly

```bash
python src/predict_seg.py \
  --image_path /path/to/your/panorama.png \
  --output_dir predictions/ \
  --checkpoint src/saved_weights/model_best.pth
```

---

## Outputs

Two files are written to `--output_dir` for each input `<stem>.png`:

| File | Content |
|---|---|
| `<stem>_pred_labels.png` | Semantic segmentation map (RGBA, class colours) |
| `<stem>_overlay.png` | Original panorama blended with segmentation (RGBA) |

---

## Key Arguments

| Argument | Default | Description |
|---|---|---|
| `--image_path` | *(required)* | Path to input ERP panorama |
| `--output_dir` | *(required)* | Directory for output images |
| `--checkpoint` | `saved_weights/model_best.pth` | Model weights |
| `--device` | `auto` | `auto` \| `cuda` \| `cpu` |
| `--img_width` | `512` | ERP width (height = width / 2) |
| `--gt_path` | `None` | Optional GT path for ignore-region masking |
