#!/usr/bin/env bash
set -e

# GPU selection (override by exporting GPU_ID or CUDA_VISIBLE_DEVICES)
GPU_ID=${GPU_ID:-1}
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU_ID}"

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
PYTHON=${PYTHON:-python}

IMG=${IMG:-"$REPO_DIR/example_data/original/rgb/camera_5153d9f273054dd99cae0f83f8457604_office_26_frame_equirectangular_domain_rgb.png"}
CKPT=${CKPT:-"$SCRIPT_DIR/saved_weights/model_best.pth"}
OUTDIR=${OUTDIR:-"$REPO_DIR/predictions/"}
W=512
NUM_SCALES=4
ENC="2 4 8 16"

DEVICE="auto"
CHUNK=8192

if [ -z "$IMG" ]; then
  echo "Set IMG to an input ERP image path." >&2
  exit 1
fi

mkdir -p "$OUTDIR"

cmd=("$PYTHON" "$SCRIPT_DIR/predict_seg.py"
  --image_path "$IMG"
  --output_dir "$OUTDIR"
  --checkpoint "$CKPT"
  --img_width "$W"
  --num_scales "$NUM_SCALES"
  --enc_num_heads $ENC
  --device "$DEVICE"
  --nn_chunk "$CHUNK"
)

"${cmd[@]}"
