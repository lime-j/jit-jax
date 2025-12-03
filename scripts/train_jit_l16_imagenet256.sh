#!/usr/bin/env bash
set -euo pipefail

# Training script for JiT-L/16 on ImageNet 256x256 for 600 epochs (TPU pod ready).
# Batch size is per device; adjust if you change device count.

TFDS_DATA_DIR="${TFDS_DATA_DIR:-gs://trc-2/}"
SAVE_DIR="${SAVE_DIR:-./jit_jax_ckpts/jit_l16_imagenet256}"

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
export TFDS_DATA_DIR

python train.py \
  --model JiT-L/16 \
  --img_size 256 \
  --batch_size 2 \
  --epochs 200 \
  --warmup_epochs 5 \
  --blr 5e-5 \
  --lr_schedule cosine \
  --proj_dropout 0.0 \
  --P_mean -0.8 --P_std 0.8 \
  --noise_scale 1.0 \
  --sampling_method heun --num_sampling_steps 50 \
  --cfg 2.9 --interval_min 0.1 --interval_max 1.0 \
  --save_dir "${SAVE_DIR}"
