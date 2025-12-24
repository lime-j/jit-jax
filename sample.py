#!/usr/bin/env python
"""Sampling script for JiT-JAX using EMA weights."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import jax_utils
from flax.training import checkpoints
from PIL import Image

from train import TrainConfig, TrainState, create_state, make_sampler, shard_prng_key


def save_images(images: np.ndarray, start_idx: int, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(images):
        idx = start_idx + i
        arr = np.clip((img + 1.0) * 127.5, 0, 255).astype(np.uint8)
        Image.fromarray(arr).save(out_dir / f"{idx:05d}.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample images from a JiT-JAX checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint directory or file.")
    parser.add_argument("--output_dir", type=str, default="./samples", help="Where to write PNG samples.")
    parser.add_argument("--model", type=str, default="JiT-B/16")
    parser.add_argument("--model_backend", type=str, default="jax", choices=["jax", "torchax"])
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--class_num", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=128, help="Per-device batch size for sampling.")
    parser.add_argument("--num_images", type=int, default=50000)
    parser.add_argument("--sampling_method", type=str, default="heun")
    parser.add_argument("--num_sampling_steps", type=int, default=50)
    parser.add_argument("--cfg", type=float, default=1.0)
    parser.add_argument("--interval_min", type=float, default=0.0)
    parser.add_argument("--interval_max", type=float, default=1.0)
    parser.add_argument("--noise_scale", type=float, default=1.0)
    parser.add_argument("--params_key", type=str, default="ema1", choices=["ema1", "ema2", "params"])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if jax.process_count() > 1 and jax.process_index() != 0:
        print("Sampling runs on host 0 only; other hosts exit early.")
        return

    devices = jax.local_device_count()
    per_device = args.batch_size
    global_batch = per_device * devices

    config = TrainConfig(
        model=args.model,
        model_backend=args.model_backend,
        img_size=args.img_size,
        batch_size=per_device,
        sampling_method=args.sampling_method,
        num_sampling_steps=args.num_sampling_steps,
        cfg=args.cfg,
        interval_min=args.interval_min,
        interval_max=args.interval_max,
        noise_scale=args.noise_scale,
        class_num=args.class_num,
    )

    rng = jax.random.PRNGKey(args.seed)
    rng = jax.random.fold_in(rng, jax.process_index())
    rng, init_rng = jax.random.split(rng)

    state = create_state(init_rng, config, steps_per_epoch=1)
    ckpt_path = checkpoints.latest_checkpoint(args.checkpoint)
    if ckpt_path is None:
        last_dir = Path(args.checkpoint) / "last"
        best_dir = Path(args.checkpoint) / "best"
        if last_dir.is_dir():
            ckpt_path = str(last_dir)
        elif best_dir.is_dir():
            ckpt_path = str(best_dir)
        else:
            ckpt_path = args.checkpoint
    state = checkpoints.restore_checkpoint(ckpt_path, target=state)
    state = jax_utils.replicate(state)

    sampler = make_sampler(config, params_key=args.params_key)
    p_sampler = jax.pmap(sampler, axis_name="batch")

    out_dir = Path(args.output_dir)
    labels_all = np.arange(args.class_num).repeat(math.ceil(args.num_images / args.class_num))[: args.num_images]

    total_steps = math.ceil(args.num_images / global_batch)
    idx = 0
    for step in range(total_steps):
        rng, step_rng = jax.random.split(rng)
        step_rng = shard_prng_key(jax.random.fold_in(step_rng, step))

        start = step * global_batch
        end = min((step + 1) * global_batch, args.num_images)
        labels_step = labels_all[start:end]
        pad = global_batch - labels_step.shape[0]
        if pad > 0:
            labels_step = np.concatenate([labels_step, np.zeros(pad, dtype=labels_step.dtype)])

        labels_step = labels_step.reshape((devices, per_device))
        samples = p_sampler(state, jnp.asarray(labels_step), step_rng)
        samples = np.asarray(samples).reshape((-1, args.img_size, args.img_size, 3))
        samples = samples[: end - start]

        if jax.process_index() == 0:
            save_images(samples, idx, out_dir)
        idx += samples.shape[0]

    if jax.process_index() == 0:
        print(f"Saved {idx} samples to {out_dir}")


if __name__ == "__main__":
    main()
