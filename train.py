"""Training and sampling entrypoints for JiT-JAX (Flax + Optax)."""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from typing import Callable, Iterable

import jax
import jax.numpy as jnp
import optax
from flax import jax_utils
from flax import struct
from flax.training import checkpoints, train_state

from jit_jax.datasets import DataPipeline
from jit_jax.model import JiT_models


DEFAULT_IMAGENET_TRAIN_EXAMPLES = 1_281_167  # ImageNet-1k train split size.
DEFAULT_IMAGENET_VAL_EXAMPLES = 50_000


def shard_prng_key(key: jax.Array) -> jax.Array:
    return jax.random.split(key, jax.local_device_count())


def sample_t(rng: jax.Array, n: int, p_mean: float, p_std: float) -> jax.Array:
    z = jax.random.normal(rng, (n,)) * p_std + p_mean
    return jax.nn.sigmoid(z)


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h{m:02d}m{s:02d}s"
    if m > 0:
        return f"{m}m{s:02d}s"
    return f"{s}s"


@dataclass
class TrainConfig:
    model: str = "JiT-B/16"
    img_size: int = 256
    batch_size: int = 128
    epochs: int = 200
    warmup_epochs: int = 5
    lr: float | None = None
    blr: float = 5e-5
    min_lr: float = 0.0
    lr_schedule: str = "constant"
    weight_decay: float = 0.0
    ema_decay1: float = 0.9999
    ema_decay2: float = 0.9996
    P_mean: float = -0.8
    P_std: float = 0.8
    noise_scale: float = 1.0
    t_eps: float = 5e-2
    label_drop_prob: float = 0.1
    sampling_method: str = "heun"
    num_sampling_steps: int = 50
    cfg: float = 1.0
    interval_min: float = 0.0
    interval_max: float = 1.0
    class_num: int = 1000
    seed: int = 0
    log_every: int = 100
    save_dir: str = "./jit_jax_ckpts"
    steps_per_epoch: int | None = None
    attn_dropout: float = 0.0
    proj_dropout: float = 0.0
    resume: str | None = None
    use_flash: bool = True
    val_steps: int | None = None


class TrainState(train_state.TrainState):
    ema1: struct.PyTreeNode
    ema2: struct.PyTreeNode
    apply_fn: Callable = struct.field(pytree_node=False)


def make_lr_schedule(config: TrainConfig, steps_per_epoch: int) -> Callable[[int], float]:
    eff_batch = config.batch_size * jax.device_count()
    base_lr = config.lr or config.blr * eff_batch / 256.0
    warmup_steps = config.warmup_epochs * steps_per_epoch
    total_steps = config.epochs * steps_per_epoch
    min_lr = config.min_lr

    def schedule(step: int) -> float:
        step = jnp.asarray(step, dtype=jnp.float32)
        warmup = jnp.asarray(warmup_steps, dtype=jnp.float32)
        total = jnp.asarray(total_steps, dtype=jnp.float32)

        if config.lr_schedule == "cosine":
            progress = (step - warmup) / jnp.maximum(total - warmup, 1.0)
            cosine_lr = min_lr + (base_lr - min_lr) * 0.5 * (1.0 + jnp.cos(jnp.pi * progress))
            lr = jnp.where(warmup > 0, jnp.where(step < warmup, base_lr * (step / warmup), cosine_lr), cosine_lr)
        elif config.lr_schedule == "constant":
            lr = jnp.where(warmup > 0, jnp.where(step < warmup, base_lr * (step / warmup), base_lr), base_lr)
        else:
            raise ValueError(f"Unknown lr_schedule {config.lr_schedule}")
        return lr

    return schedule


def create_state(rng: jax.Array, config: TrainConfig, steps_per_epoch: int) -> TrainState:
    model_def = JiT_models[config.model](
        input_size=config.img_size,
        in_channels=3,
        num_classes=config.class_num,
        attn_drop=config.attn_dropout,
        proj_drop=config.proj_dropout,
        use_flash=config.use_flash,
    )
    per_device_batch = config.batch_size
    variables = model_def.init(
        rng,
        jnp.ones((per_device_batch, config.img_size, config.img_size, 3), jnp.float32),
        jnp.zeros((per_device_batch,)),
        jnp.zeros((per_device_batch,), jnp.int32),
        train=True,
    )
    params = variables["params"]

    lr_schedule = make_lr_schedule(config, steps_per_epoch)
    tx = optax.adamw(learning_rate=lr_schedule, b1=0.9, b2=0.95, weight_decay=config.weight_decay)

    state = TrainState.create(apply_fn=model_def.apply, params=params, tx=tx, ema1=params, ema2=params)
    return state


def drop_labels(rng: jax.Array, labels: jax.Array, drop_prob: float, num_classes: int) -> jax.Array:
    drop = jax.random.bernoulli(rng, drop_prob, labels.shape)
    return jnp.where(drop, num_classes, labels)


def v_prediction(model_apply: Callable, params: dict, z: jax.Array, t: jax.Array, labels: jax.Array, *, config: TrainConfig, train: bool, rng: jax.Array | None) -> jax.Array:
    t_vec = jnp.broadcast_to(jnp.asarray(t), (z.shape[0],))
    denom = jnp.clip(1.0 - t_vec.reshape((z.shape[0],) + (1,) * (z.ndim - 1)), a_min=config.t_eps)
    rngs = {"dropout": rng} if rng is not None else None
    x_pred = model_apply({"params": params}, z, t_vec, labels, train=train, rngs=rngs)
    v_pred = (x_pred - z) / denom
    return v_pred


def make_train_step(config: TrainConfig) -> Callable:
    def train_step(state: TrainState, batch: dict, rng: jax.Array) -> tuple[TrainState, dict]:
        images = batch["images"]
        labels = batch["labels"]

        rng, rng_t, rng_noise, rng_drop, rng_label = jax.random.split(rng, 5)
        t = sample_t(rng_t, images.shape[0], config.P_mean, config.P_std)
        t_broadcast = t.reshape((images.shape[0],) + (1,) * (images.ndim - 1))
        noise = jax.random.normal(rng_noise, images.shape, dtype=images.dtype) * config.noise_scale
        z = t_broadcast * images + (1.0 - t_broadcast) * noise
        denom = jnp.clip(1.0 - t_broadcast, a_min=config.t_eps)
        v = (images - z) / denom
        labels_dropped = drop_labels(rng_label, labels, config.label_drop_prob, config.class_num)

        def loss_fn(params):
            x_pred = state.apply_fn(
                {"params": params},
                z,
                t,
                labels_dropped,
                train=True,
                rngs={"dropout": rng_drop},
            )
            v_pred = (x_pred - z) / denom
            loss = jnp.mean((v_pred - v) ** 2)
            return loss

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        loss = jax.lax.pmean(loss, axis_name="batch")
        grads = jax.lax.pmean(grads, axis_name="batch")
        t_mean = jax.lax.pmean(jnp.mean(t), axis_name="batch")

        state = state.apply_gradients(grads=grads)
        ema1 = optax.incremental_update(state.params, state.ema1, config.ema_decay1)
        ema2 = optax.incremental_update(state.params, state.ema2, config.ema_decay2)
        state = state.replace(ema1=ema1, ema2=ema2)
        metrics = {"loss": loss, "t_mean": t_mean}
        return state, metrics

    return train_step


def predict_cfg(model_apply: Callable, params: dict, z: jax.Array, t: jax.Array, labels: jax.Array, *, config: TrainConfig) -> jax.Array:
    v_cond = v_prediction(model_apply, params, z, t, labels, config=config, train=False, rng=None)
    v_uncond = v_prediction(
        model_apply, params, z, t, jnp.full_like(labels, config.class_num), config=config, train=False, rng=None
    )
    interval_mask = (t < config.interval_max) & ((config.interval_min == 0.0) | (t > config.interval_min))
    cfg_scale = jnp.where(interval_mask, config.cfg, 1.0)
    return v_uncond + cfg_scale * (v_cond - v_uncond)


def make_sampler(config: TrainConfig, params_key: str = "ema1") -> Callable:
    def sampler(state: TrainState, labels: jax.Array, rng: jax.Array) -> jax.Array:
        params = getattr(state, params_key)
        rng, noise_rng = jax.random.split(rng)
        z = config.noise_scale * jax.random.normal(noise_rng, (labels.shape[0], config.img_size, config.img_size, 3))
        timesteps = jnp.linspace(0.0, 1.0, config.num_sampling_steps + 1)

        def euler_step(z, t, t_next):
            v_pred = predict_cfg(state.apply_fn, params, z, t, labels, config=config)
            return z + (t_next - t) * v_pred

        def heun_step(z, t, t_next):
            v_t = predict_cfg(state.apply_fn, params, z, t, labels, config=config)
            z_euler = z + (t_next - t) * v_t
            v_t_next = predict_cfg(state.apply_fn, params, z_euler, t_next, labels, config=config)
            v = 0.5 * (v_t + v_t_next)
            return z + (t_next - t) * v

        step_fn = heun_step if config.sampling_method == "heun" else euler_step

        def body(i, carry):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            return step_fn(carry, t, t_next)

        z = jax.lax.fori_loop(0, config.num_sampling_steps - 1, body, z)
        z = euler_step(z, timesteps[-2], timesteps[-1])
        return z

    return sampler


def make_eval_step(config: TrainConfig) -> Callable:
    def eval_step(state: TrainState, batch: dict, rng: jax.Array) -> dict:
        images = batch["images"]
        labels = batch["labels"]

        rng, rng_t, rng_noise = jax.random.split(rng, 3)
        t = sample_t(rng_t, images.shape[0], config.P_mean, config.P_std)
        t_broadcast = t.reshape((images.shape[0],) + (1,) * (images.ndim - 1))
        noise = jax.random.normal(rng_noise, images.shape, dtype=images.dtype) * config.noise_scale
        z = t_broadcast * images + (1.0 - t_broadcast) * noise
        denom = jnp.clip(1.0 - t_broadcast, a_min=config.t_eps)
        v = (images - z) / denom

        x_pred = state.apply_fn({"params": state.ema1}, z, t, labels, train=False)
        v_pred = (x_pred - z) / denom
        loss = jnp.mean((v_pred - v) ** 2)
        loss = jax.lax.pmean(loss, axis_name="batch")
        return {"loss": loss}

    return eval_step


def save_checkpoint_if_needed(state: TrainState, save_dir: str, step: int) -> None:
    os.makedirs(save_dir, exist_ok=True)
    state_to_save = jax.device_get(jax_utils.unreplicate(state))
    checkpoints.save_checkpoint(save_dir, state_to_save, step=step, overwrite=True)


def train_and_maybe_sample(config: TrainConfig) -> None:
    eff_batch = config.batch_size * jax.device_count()
    steps_per_epoch = config.steps_per_epoch or max(1, DEFAULT_IMAGENET_TRAIN_EXAMPLES // eff_batch)
    val_steps = config.val_steps or max(1, DEFAULT_IMAGENET_VAL_EXAMPLES // eff_batch)
    total_steps = config.epochs * steps_per_epoch

    rng = jax.random.PRNGKey(config.seed)
    rng = jax.random.fold_in(rng, jax.process_index())
    rng, init_rng = jax.random.split(rng)
    state = create_state(init_rng, config, steps_per_epoch)
    if config.resume:
        ckpt_path = checkpoints.latest_checkpoint(config.resume)
        if ckpt_path:
            state = checkpoints.restore_checkpoint(ckpt_path, target=state)
            if jax.process_index() == 0:
                print(f"Restored checkpoint from {ckpt_path}")
    state = jax_utils.replicate(state)

    pipeline = DataPipeline(batch_size=config.batch_size, image_size=config.img_size, seed=config.seed)
    pipeline_val = DataPipeline(batch_size=config.batch_size, image_size=config.img_size, seed=config.seed)
    train_iter: Iterable[dict] = pipeline.build("train")

    p_train_step = jax.pmap(make_train_step(config), axis_name="batch")
    p_eval_step = jax.pmap(make_eval_step(config), axis_name="batch")

    start_time = time.time()
    last_log_time = start_time
    last_log_step = 0

    for epoch in range(config.epochs):
        train_loss_acc = 0.0
        train_batches = 0

        for step in range(steps_per_epoch):
            rng, step_rng = jax.random.split(rng)
            step_rng = shard_prng_key(jax.random.fold_in(step_rng, step + epoch * steps_per_epoch))
            batch = next(train_iter)
            state, metrics = p_train_step(state, batch, step_rng)
            if (step + epoch * steps_per_epoch) % config.log_every == 0 and jax.process_index() == 0:
                now = time.time()
                step_idx = step + epoch * steps_per_epoch
                steps_completed = step_idx + 1
                interval_steps = steps_completed - last_log_step
                elapsed_interval = max(now - last_log_time, 1e-6)
                imgs_per_sec = interval_steps * eff_batch / elapsed_interval
                elapsed_total = now - start_time
                remaining_steps = max(total_steps - steps_completed, 0)
                eta = (elapsed_total / steps_completed) * remaining_steps if steps_completed > 0 else 0.0
                loss_val = float(jax.device_get(metrics["loss"])[0])
                t_mean = float(jax.device_get(metrics["t_mean"])[0])
                print(
                    f"step {step_idx}: loss={loss_val:.4f} t_mean={t_mean:.4f} "
                    f"imgs/s={imgs_per_sec:.1f} eta={_format_duration(eta)}"
                )
                last_log_time = now
                last_log_step = steps_completed
            if jax.process_index() == 0:
                train_loss_acc += float(jax.device_get(metrics["loss"])[0])
                train_batches += 1

        train_loss_avg = train_loss_acc / max(train_batches, 1) if jax.process_index() == 0 else None

        # Validation loss
        val_iter = pipeline_val.build("validation", repeat=False)
        val_loss_acc = 0.0
        val_batches = 0
        for vstep in range(val_steps):
            try:
                batch = next(val_iter)
            except StopIteration:
                break
            rng, val_rng = jax.random.split(rng)
            val_rng = shard_prng_key(jax.random.fold_in(val_rng, vstep + epoch * val_steps))
            metrics = p_eval_step(state, batch, val_rng)
            if jax.process_index() == 0:
                val_loss_acc += float(jax.device_get(metrics["loss"])[0])
                val_batches += 1

        if jax.process_index() == 0:
            val_loss_avg = val_loss_acc / max(val_batches, 1)
            print(f"Epoch {epoch}: train_loss={train_loss_avg:.4f} val_loss={val_loss_avg:.4f}")

        if jax.process_index() == 0:
            save_checkpoint_if_needed(state, config.save_dir, epoch)


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train JiT in JAX (Flax + Optax).")
    parser.add_argument("--model", type=str, default="JiT-B/16")
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--blr", type=float, default=5e-5)
    parser.add_argument("--min_lr", type=float, default=0.0)
    parser.add_argument("--lr_schedule", type=str, default="constant")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--ema_decay1", type=float, default=0.9999)
    parser.add_argument("--ema_decay2", type=float, default=0.9996)
    parser.add_argument("--P_mean", type=float, default=-0.8)
    parser.add_argument("--P_std", type=float, default=0.8)
    parser.add_argument("--noise_scale", type=float, default=1.0)
    parser.add_argument("--t_eps", type=float, default=5e-2)
    parser.add_argument("--label_drop_prob", type=float, default=0.1)
    parser.add_argument("--sampling_method", type=str, default="heun")
    parser.add_argument("--num_sampling_steps", type=int, default=50)
    parser.add_argument("--cfg", type=float, default=1.0)
    parser.add_argument("--interval_min", type=float, default=0.0)
    parser.add_argument("--interval_max", type=float, default=1.0)
    parser.add_argument("--class_num", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--save_dir", type=str, default="./jit_jax_ckpts")
    parser.add_argument("--steps_per_epoch", type=int, default=None)
    parser.add_argument("--attn_dropout", type=float, default=0.0)
    parser.add_argument("--proj_dropout", type=float, default=0.0)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument(
        "--use_flash",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use TPU flash attention; defaults to on and will error if unavailable.",
    )
    parser.add_argument("--val_steps", type=int, default=None, help="Override validation steps per epoch.")
    args = parser.parse_args()
    return TrainConfig(**vars(args))


if __name__ == "__main__":
    config = parse_args()
    train_and_maybe_sample(config)
