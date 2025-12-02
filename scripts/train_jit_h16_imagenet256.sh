#!/usr/bin/env bash
set -euo pipefail

# Training script for JiT-H/16 on ImageNet 256x256 for 600 epochs (TPU pod ready).
# Batch size is per device; adjust if you change device count.

TFDS_DATA_DIR="${TFDS_DATA_DIR:-/path/to/tfds}"
SAVE_DIR="${SAVE_DIR:-./jit_jax_ckpts/jit_h16_imagenet256}"

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
export TFDS_DATA_DIR

python -m jit_jax.train \
  --model JiT-H/16 \
  --img_size 256 \
  --batch_size 128 \
  --epochs 600 \
  --warmup_epochs 5 \
  --blr 5e-5 \
  --lr_schedule cosine \
  --proj_dropout 0.2 \
  --P_mean -0.8 --P_std 0.8 \
  --noise_scale 1.0 \
  --sampling_method heun --num_sampling_steps 50 \
  --cfg 2.2 --interval_min 0.1 --interval_max 1.0 \
  --save_dir "${SAVE_DIR}"
