#!/usr/bin/env python
"""End-to-end sampling + FID/IS evaluation using torch-fidelity (CPU/GPU PyTorch)."""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import jax_utils
from flax.training import checkpoints

from jit_jax.train import TrainConfig, create_state, make_sampler, shard_prng_key


def run_torch_fidelity(folder: Path, fid_stats: Path, use_cuda: bool) -> dict:
    cmd = [
        "python",
        "-m",
        "torch_fidelity",
        "--input1",
        str(folder),
        "--fid-statistics",
        str(fid_stats),
        "--fid",
        "--isc",
    ]
    if use_cuda:
        cmd.append("--cuda")
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    print(result.stdout)
    # torch-fidelity does not emit JSON by default; for now we just stream stdout.
    return {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample and evaluate JiT-JAX with torch-fidelity.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint directory or file.")
    parser.add_argument("--fid_stats", type=str, required=True, help="Path to fid_stats npz (same as PyTorch).")
    parser.add_argument("--model", type=str, default="JiT-B/16")
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
    parser.add_argument(
        "--use_flash",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use TPU flash attention; defaults to on and will error if unavailable.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default=None, help="Optional directory to save samples (else temp).")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA for torch-fidelity if available.")
    args = parser.parse_args()

    if jax.process_count() > 1 and jax.process_index() != 0:
        print("Evaluation runs on host 0 only; other hosts exit early.")
        return

    devices = jax.local_device_count()
    per_device = args.batch_size
    global_batch = per_device * devices

    config = TrainConfig(
        model=args.model,
        img_size=args.img_size,
        batch_size=per_device,
        sampling_method=args.sampling_method,
        num_sampling_steps=args.num_sampling_steps,
        cfg=args.cfg,
        interval_min=args.interval_min,
        interval_max=args.interval_max,
        noise_scale=args.noise_scale,
        class_num=args.class_num,
        use_flash=args.use_flash,
    )

    rng = jax.random.PRNGKey(args.seed)
    rng = jax.random.fold_in(rng, jax.process_index())
    rng, init_rng = jax.random.split(rng)

    state = create_state(init_rng, config, steps_per_epoch=1)
    ckpt_path = checkpoints.latest_checkpoint(args.checkpoint) or args.checkpoint
    state = checkpoints.restore_checkpoint(ckpt_path, target=state)
    state = jax_utils.replicate(state)

    sampler = make_sampler(config, params_key=args.params_key)
    p_sampler = jax.pmap(sampler, axis_name="batch")

    out_dir = Path(args.output_dir) if args.output_dir else Path(tempfile.mkdtemp(prefix="jit_jax_samples_"))
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
            # Save as PNGs
            for i, img in enumerate(samples):
                arr = np.clip((img + 1.0) * 127.5, 0, 255).astype(np.uint8)
                img_id = idx + i
                out_path = out_dir / f"{img_id:05d}.png"
                out_dir.mkdir(parents=True, exist_ok=True)
                from PIL import Image

                Image.fromarray(arr).save(out_path)
        idx += samples.shape[0]

    if jax.process_index() == 0:
        print(f"Saved {idx} samples to {out_dir}")
        run_torch_fidelity(out_dir, Path(args.fid_stats), use_cuda=args.cuda)


if __name__ == "__main__":
    main()
