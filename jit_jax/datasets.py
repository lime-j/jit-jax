"""tfds ImageNet input pipeline for JiT-JAX."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


def _center_crop_arr(image: tf.Tensor, image_size: int) -> tf.Tensor:
    """Match the PyTorch center-crop logic used in the original JiT repo."""
    image = tf.cast(image, tf.float32)
    h = tf.shape(image)[0]
    w = tf.shape(image)[1]

    def cond_fn(img, h_cur, w_cur):
        return tf.logical_and(h_cur >= 2 * image_size, w_cur >= 2 * image_size)

    def body_fn(img, h_cur, w_cur):
        new_h = tf.cast(tf.math.floordiv(h_cur, 2), tf.int32)
        new_w = tf.cast(tf.math.floordiv(w_cur, 2), tf.int32)
        img = tf.image.resize(img, (new_h, new_w), method=tf.image.ResizeMethod.AREA, antialias=True)
        return img, new_h, new_w

    image, h, w = tf.while_loop(
        cond_fn,
        body_fn,
        (image, h, w),
        shape_invariants=(
            tf.TensorShape([None, None, None]),
            tf.TensorShape([]),
            tf.TensorShape([]),
        ),
    )

    min_hw = tf.minimum(h, w)
    scale = tf.cast(image_size, tf.float32) / tf.cast(min_hw, tf.float32)
    new_hw = tf.cast(tf.round(tf.cast([h, w], tf.float32) * scale), tf.int32)
    image = tf.image.resize(image, new_hw, method=tf.image.ResizeMethod.BICUBIC, antialias=True)
    offset_h = tf.maximum((new_hw[0] - image_size) // 2, 0)
    offset_w = tf.maximum((new_hw[1] - image_size) // 2, 0)
    image = tf.image.crop_to_bounding_box(image, offset_h, offset_w, image_size, image_size)
    return image


def _preprocess(example: dict, image_size: int, train: bool, dtype: tf.dtypes.DType) -> Tuple[tf.Tensor, tf.Tensor]:
    image = _center_crop_arr(example["image"], image_size)
    if train:
        image = tf.image.random_flip_left_right(image)
    image = tf.clip_by_value(image, 0.0, 255.0)
    image = image / 127.5 - 1.0
    image = tf.cast(image, dtype)
    label = tf.cast(example["label"], tf.int32)
    return image, label


@dataclass
class DataPipeline:
    batch_size: int  # per-device batch size
    image_size: int
    dtype: tf.dtypes.DType = tf.float32
    shuffle_buffer: int = 50_000
    seed: int = 0

    def build(self, split: str, *, repeat: bool = True) -> Iterator[dict]:
        local_devices = jax.local_device_count()
        per_host_batch = self.batch_size * local_devices
        process_index = jax.process_index()
        process_count = jax.process_count()

        # Evenly shard the split across hosts to avoid overlap using TFDS helpers.
        split_shards = tfds.even_splits(split, n=process_count)
        split_spec = split_shards[process_index]

        train = split == "train"
        ds = tfds.load("imagenet2012", split=split_spec)
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
        ds = ds.with_options(options)
        if train:
            ds = ds.shuffle(self.shuffle_buffer, seed=self.seed + process_index)
        if repeat:
            ds = ds.repeat()
        ds = ds.map(lambda x: _preprocess(x, self.image_size, train=train, dtype=self.dtype), num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(per_host_batch, drop_remainder=True)
        ds = ds.prefetch(tf.data.AUTOTUNE)

        for images, labels in tfds.as_numpy(ds):
            images = np.asarray(images).reshape((local_devices, self.batch_size) + images.shape[1:])
            labels = np.asarray(labels).reshape((local_devices, self.batch_size))
            yield {"images": images, "labels": labels}
