# JiT-JAX (Flax + Optax)

A JAX/Flax re-implementation of **JiT** (Just image Transformer) with the same architecture, v-prediction objective, EMA tracking, and sampling schedule as the PyTorch version. It targets TPU training and uses TensorFlow Datasets (TFDS) ImageNet input.

## Highlights
- Flax/Linen model matches the PyTorch blocks: rotary attention, AdaLN modulation, in-context tokens, and zero-initialized output/ada layers.
- V-prediction loss with logistic time sampling (`P_mean`, `P_std`), label dropout for classifier-free guidance, dual EMA tracks, and Heun/Euler ODE sampler with configurable CFG interval.
- TFDS ImageNet pipeline matches the PyTorch logic: repeated half-resolution downsample (area) until <2x target, bicubic resize + center-crop, random flip, and `[-1, 1]` normalization. Batch size is **per device**; effective batch = `batch_size * device_count`.
- TPU flash attention (Pallas) is enabled by default and will raise if unavailable. Disable with `--no-use_flash` to fall back.

## Quick start (TPU)
```
cd JiT-jax
python -m jit_jax.train \
  --model JiT-B/16 \
  --img_size 256 \
  --batch_size 128 \            # per device
  --epochs 600 \
  --blr 5e-5 \
  --warmup_epochs 5 \
  --lr_schedule cosine \
  --P_mean -0.8 --P_std 0.8 \
  --noise_scale 1.0 \
  --sampling_method heun --num_sampling_steps 50 \
  --cfg 2.9 --interval_min 0.1 --interval_max 1.0 \
  --save_dir /path/to/ckpts
```

- Set `TFDS_DATA_DIR` to your ImageNet location or rely on the default TFDS cache.
- Learning rate scales like the original (`base_lr = blr * effective_batch / 256` unless `--lr` is set).
- `--eval_samples N` optionally writes a `.npy` of generated images (using EMA1) at each epoch in `save_dir`.
- `--resume /path/to/ckpts` will restore the latest Flax checkpoint in that folder.
- `--steps_per_epoch` lets you override the default ImageNet-1k length (`1,281,167 // effective_batch`).
- Validation loss is computed every epoch on ImageNet validation (host-sharded) and printed alongside train loss.
- For sampling-only, use `python sample.py --checkpoint <path> --num_images ...` (host 0 only). For end-to-end FID/IS matching the PyTorch flow, use `python eval_fid.py --checkpoint <path> --fid_stats fid_stats/jit_in256_stats.npz --num_images 50000 --batch_size 128 --sampling_method heun --num_sampling_steps 50 --cfg 2.2 --interval_min 0.1 --interval_max 1.0`.

## Notes
- Training uses `jax.pmap` across all devices; the TFDS loader shards the split by host (`train[process_index/process_count]`) and disables auto-sharding to avoid overlap. Batch size is per device; global batch = `batch_size * jax.device_count()`.
- Checkpoints are Flax-native (`flax.training.checkpoints`) with EMA params; they are saved under `save_dir`.
- Reference script for JiT-H/16 on ImageNet 256 is at `scripts/train_jit_h16_imagenet256.sh`.
- Sampling is decoupled from training; use `python sample.py --checkpoint <path> --num_images ...` to generate PNGs (host 0 only).
- End-to-end evaluation: `python eval_fid.py --checkpoint <path> --fid_stats fid_stats/jit_in256_stats.npz --num_images 50000 --batch_size 128 --sampling_method heun --num_sampling_steps 50 --cfg 2.2 --interval_min 0.1 --interval_max 1.0`. This generates class-balanced samples with EMA params, then runs torch-fidelity with the same stats as the PyTorch repo. Use `--cuda` if you want torch-fidelity on GPU.

## Reported vs. this implementation
| Model (256px) | Paper FID / IS | This repo FID / IS |
| --- | --- | --- |
| JiT-B/16 | (paper) / (paper) | training in progress |
| JiT-L/16 | (paper) / (paper) | training in progress |
| JiT-H/16 | (paper) / (paper) | training in progress |
