"""Flax/Linen implementation of JiT (Just image Transformer)."""

from __future__ import annotations

import math
from typing import Callable, Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from jit_jax.flash_attention_stub import tpu_flash_attention, tpu_flash_available


def rotate_half(x: jnp.ndarray) -> jnp.ndarray:
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concatenate((-x2, x1), axis=-1)


class VisionRotaryEmbeddingFast(nn.Module):
    dim: int
    pt_seq_len: int = 16
    ft_seq_len: Optional[int] = None
    theta: float = 10000.0
    max_freq: float = 10.0
    freqs_for: str = "lang"
    num_cls_token: int = 0

    def setup(self) -> None:
        ft_seq_len = self.ft_seq_len or self.pt_seq_len
        if self.freqs_for == "lang":
            freqs = 1.0 / (self.theta ** (jnp.arange(0, self.dim, 2) / self.dim))
        elif self.freqs_for == "pixel":
            freqs = jnp.linspace(1.0, self.max_freq / 2, self.dim // 2) * math.pi
        else:
            raise ValueError(f"Unsupported freqs_for={self.freqs_for}")

        t = jnp.arange(ft_seq_len) / ft_seq_len * self.pt_seq_len
        freqs_single = jnp.einsum("n,f->nf", t, freqs)
        freqs_single = jnp.repeat(freqs_single, repeats=2, axis=-1)

        freqs_h = jnp.broadcast_to(freqs_single[:, None, :], (ft_seq_len, ft_seq_len, freqs_single.shape[-1]))
        freqs_w = jnp.broadcast_to(freqs_single[None, :, :], (ft_seq_len, ft_seq_len, freqs_single.shape[-1]))
        freqs = jnp.concatenate([freqs_h, freqs_w], axis=-1)
        freqs = freqs.reshape(-1, freqs.shape[-1])

        if self.num_cls_token > 0:
            cos_pad = jnp.ones((self.num_cls_token, freqs.shape[-1]))
            sin_pad = jnp.zeros((self.num_cls_token, freqs.shape[-1]))
            self.freqs_cos = jnp.concatenate([cos_pad, jnp.cos(freqs)], axis=0)
            self.freqs_sin = jnp.concatenate([sin_pad, jnp.sin(freqs)], axis=0)
        else:
            self.freqs_cos = jnp.cos(freqs)
            self.freqs_sin = jnp.sin(freqs)

    def __call__(self, t: jnp.ndarray) -> jnp.ndarray:
        # t: (B, heads, tokens, dim)
        cos = self.freqs_cos.astype(t.dtype)[None, None, :, :]
        sin = self.freqs_sin.astype(t.dtype)[None, None, :, :]
        return t * cos + rotate_half(t) * sin


class RMSNorm(nn.Module):
    hidden_size: int
    eps: float = 1e-6

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        dtype = x.dtype
        x = x.astype(jnp.float32)
        variance = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        x = x * jax.lax.rsqrt(variance + self.eps)
        weight = self.param("weight", nn.initializers.ones, (self.hidden_size,))
        return (weight * x).astype(dtype)


def modulate(x: jnp.ndarray, shift: jnp.ndarray, scale: jnp.ndarray) -> jnp.ndarray:
    return x * (1.0 + scale[:, None, :]) + shift[:, None, :]


class BottleneckPatchEmbed(nn.Module):
    img_size: int = 224
    patch_size: int = 16
    in_chans: int = 3
    pca_dim: int = 768
    embed_dim: int = 768
    bias: bool = True

    def setup(self) -> None:
        self.proj1 = nn.Conv(
            features=self.pca_dim,
            kernel_size=(self.patch_size, self.patch_size),
            strides=(self.patch_size, self.patch_size),
            padding="VALID",
            use_bias=False,
            kernel_init=nn.initializers.xavier_uniform(),
        )
        self.proj2 = nn.Conv(
            features=self.embed_dim,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="VALID",
            use_bias=self.bias,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.zeros,
        )
        self.num_patches = (self.img_size // self.patch_size) ** 2

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: (B, H, W, C)
        x = self.proj1(x)
        x = self.proj2(x)
        b, h, w, c = x.shape
        return x.reshape(b, h * w, c)


class TimestepEmbedder(nn.Module):
    hidden_size: int
    frequency_embedding_size: int = 256

    @staticmethod
    def timestep_embedding(t: jnp.ndarray, dim: int, max_period: int = 10000) -> jnp.ndarray:
        half = dim // 2
        freqs = jnp.exp(-math.log(max_period) * jnp.arange(half, dtype=jnp.float32) / half)
        args = t[:, None].astype(jnp.float32) * freqs[None]
        embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
        if dim % 2:
            embedding = jnp.concatenate([embedding, jnp.zeros_like(embedding[:, :1])], axis=-1)
        return embedding

    @nn.compact
    def __call__(self, t: jnp.ndarray) -> jnp.ndarray:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        x = nn.Dense(self.hidden_size, kernel_init=nn.initializers.normal(0.02))(t_freq)
        x = nn.silu(x)
        x = nn.Dense(self.hidden_size, kernel_init=nn.initializers.normal(0.02))(x)
        return x


class LabelEmbedder(nn.Module):
    num_classes: int
    hidden_size: int

    @nn.compact
    def __call__(self, labels: jnp.ndarray) -> jnp.ndarray:
        emb = nn.Embed(
            num_embeddings=self.num_classes + 1,
            features=self.hidden_size,
            embedding_init=nn.initializers.normal(stddev=0.02),
        )
        return emb(labels)


class Attention(nn.Module):
    dim: int
    num_heads: int = 8
    qkv_bias: bool = True
    qk_norm: bool = True
    attn_drop: float = 0.0
    proj_drop: float = 0.0
    use_flash: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray, rope: VisionRotaryEmbeddingFast, deterministic: bool) -> jnp.ndarray:
        head_dim = self.dim // self.num_heads
        qkv = nn.Dense(
            self.dim * 3,
            use_bias=self.qkv_bias,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.zeros,
        )(x)
        qkv = qkv.reshape(x.shape[0], x.shape[1], 3, self.num_heads, head_dim).transpose(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.qk_norm:
            q = RMSNorm(head_dim, eps=1e-6)(q)
            k = RMSNorm(head_dim, eps=1e-6)(k)

        q = rope(q)
        k = rope(k)

        rng = self.make_rng("dropout") if not deterministic else None
        if self.use_flash and not tpu_flash_available:
            raise RuntimeError("Flash attention requested but TPU flash kernel is unavailable in this JAX build.")

        if self.use_flash:
            out = tpu_flash_attention.flash_attention(
                q,
                k,
                v,
                dropout_rng=rng,
                dropout_rate=self.attn_drop if not deterministic else 0.0,
                causal=False,
                softmax_scale=1.0 / math.sqrt(head_dim),
            )
        else:
            scale = 1.0 / math.sqrt(head_dim)
            attn_logits = jnp.einsum("bhqd,bhkd->bhqk", q.astype(jnp.float32), k.astype(jnp.float32)) * scale
            attn = nn.softmax(attn_logits, axis=-1)
            attn = nn.Dropout(rate=self.attn_drop)(attn, deterministic=deterministic)
            out = jnp.einsum("bhqk,bhkd->bhqd", attn, v)
        out = out.transpose(0, 2, 1, 3).reshape(x.shape[0], x.shape[1], self.dim)

        out = nn.Dense(self.dim, kernel_init=nn.initializers.xavier_uniform())(out)
        out = nn.Dropout(rate=self.proj_drop)(out, deterministic=deterministic)
        return out


class SwiGLUFFN(nn.Module):
    dim: int
    hidden_dim: int
    drop: float = 0.0
    bias: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool) -> jnp.ndarray:
        hidden_dim = int(self.hidden_dim * 2 / 3)
        x12 = nn.Dense(2 * hidden_dim, use_bias=self.bias, kernel_init=nn.initializers.xavier_uniform())(x)
        x1, x2 = jnp.split(x12, 2, axis=-1)
        hidden = nn.silu(x1) * x2
        hidden = nn.Dropout(rate=self.drop)(hidden, deterministic=deterministic)
        return nn.Dense(self.dim, use_bias=self.bias, kernel_init=nn.initializers.xavier_uniform())(hidden)


class FinalLayer(nn.Module):
    hidden_size: int
    patch_size: int
    out_channels: int

    @nn.compact
    def __call__(self, x: jnp.ndarray, c: jnp.ndarray, deterministic: bool) -> jnp.ndarray:
        norm = RMSNorm(self.hidden_size)(x)
        shift, scale = jnp.split(
            nn.Sequential(
                [
                    nn.silu,
                    nn.Dense(2 * self.hidden_size, kernel_init=nn.initializers.zeros, bias_init=nn.initializers.zeros),
                ]
            )(c),
            2,
            axis=-1,
        )
        x = modulate(norm, shift, scale)
        x = nn.Dense(
            self.patch_size * self.patch_size * self.out_channels,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
        )(x)
        return x


class JiTBlock(nn.Module):
    hidden_size: int
    num_heads: int
    mlp_ratio: float = 4.0
    attn_drop: float = 0.0
    proj_drop: float = 0.0

    @nn.compact
    def __call__(self, x: jnp.ndarray, c: jnp.ndarray, rope: VisionRotaryEmbeddingFast, deterministic: bool) -> jnp.ndarray:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = jnp.split(
            nn.Sequential(
                [
                    nn.silu,
                    nn.Dense(6 * self.hidden_size, kernel_init=nn.initializers.zeros, bias_init=nn.initializers.zeros),
                ]
            )(c),
            6,
            axis=-1,
        )

        x = x + gate_msa[:, None, :] * Attention(
            dim=self.hidden_size,
            num_heads=self.num_heads,
            qkv_bias=True,
            qk_norm=True,
            attn_drop=self.attn_drop,
            proj_drop=self.proj_drop,
        )(modulate(RMSNorm(self.hidden_size, eps=1e-6)(x), shift_msa, scale_msa), rope=rope, deterministic=deterministic)

        mlp_hidden_dim = int(self.hidden_size * self.mlp_ratio)
        x = x + gate_mlp[:, None, :] * SwiGLUFFN(
            dim=self.hidden_size,
            hidden_dim=mlp_hidden_dim,
            drop=self.proj_drop,
        )(modulate(RMSNorm(self.hidden_size, eps=1e-6)(x), shift_mlp, scale_mlp), deterministic=deterministic)
        return x


def _get_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> np.ndarray:
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0).reshape([2, 1, grid_size, grid_size])

    emb_h = _get_1d_sincos_pos_embed(embed_dim // 2, grid[0])
    emb_w = _get_1d_sincos_pos_embed(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


def _get_1d_sincos_pos_embed(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / (10000**omega)

    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)

    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    return np.concatenate([emb_sin, emb_cos], axis=1)


class JiT(nn.Module):
    input_size: int = 256
    patch_size: int = 16
    in_channels: int = 3
    hidden_size: int = 1024
    depth: int = 24
    num_heads: int = 16
    mlp_ratio: float = 4.0
    attn_drop: float = 0.0
    proj_drop: float = 0.0
    num_classes: int = 1000
    bottleneck_dim: int = 128
    in_context_len: int = 32
    in_context_start: int = 8
    use_flash: bool = True

    def setup(self) -> None:
        self.t_embedder = TimestepEmbedder(self.hidden_size)
        self.y_embedder = LabelEmbedder(self.num_classes, self.hidden_size)
        self.x_embedder = BottleneckPatchEmbed(
            img_size=self.input_size,
            patch_size=self.patch_size,
            in_chans=self.in_channels,
            pca_dim=self.bottleneck_dim,
            embed_dim=self.hidden_size,
            bias=True,
        )

        num_patches = self.x_embedder.num_patches
        pos_embed = _get_2d_sincos_pos_embed(self.hidden_size, int(num_patches**0.5))
        self.pos_embed = jnp.asarray(pos_embed[None, ...])

        if self.in_context_len > 0:
            self.in_context_posemb = self.param(
                "in_context_posemb",
                lambda rng: jax.random.normal(rng, (1, self.in_context_len, self.hidden_size)) * 0.02,
            )
        else:
            self.in_context_posemb = None

        half_head_dim = self.hidden_size // self.num_heads // 2
        hw_seq_len = self.input_size // self.patch_size
        self.feat_rope = VisionRotaryEmbeddingFast(dim=half_head_dim, pt_seq_len=hw_seq_len, num_cls_token=0)
        self.feat_rope_incontext = VisionRotaryEmbeddingFast(
            dim=half_head_dim, pt_seq_len=hw_seq_len, num_cls_token=self.in_context_len
        )

        drops = [
            self.attn_drop if (self.depth // 4 <= i < self.depth // 4 * 3) else 0.0 for i in range(self.depth)
        ]
        self.blocks = [
            JiTBlock(
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                attn_drop=drops[i],
                proj_drop=self.proj_drop if drops[i] > 0 else 0.0,
                use_flash=self.use_flash,
            )
            for i in range(self.depth)
        ]

        self.final_layer = FinalLayer(self.hidden_size, self.patch_size, self.in_channels)

    def unpatchify(self, x: jnp.ndarray) -> jnp.ndarray:
        c = self.in_channels
        h = w = int(x.shape[1] ** 0.5)
        x = x.reshape(x.shape[0], h, w, self.patch_size, self.patch_size, c)
        x = x.transpose(0, 1, 3, 2, 4, 5).reshape(x.shape[0], h * self.patch_size, w * self.patch_size, c)
        return x

    def __call__(self, x: jnp.ndarray, t: jnp.ndarray, y: jnp.ndarray, *, train: bool) -> jnp.ndarray:
        t_emb = self.t_embedder(t)
        y_emb = self.y_embedder(y)
        c = t_emb + y_emb

        x = self.x_embedder(x)
        x = x + self.pos_embed

        for i, block in enumerate(self.blocks):
            if self.in_context_len > 0 and i == self.in_context_start:
                assert self.in_context_posemb is not None
                in_tokens = jnp.repeat(y_emb[:, None, :], self.in_context_len, axis=1) + self.in_context_posemb
                x = jnp.concatenate([in_tokens, x], axis=1)
            rope = self.feat_rope if i < self.in_context_start else self.feat_rope_incontext
            x = block(x, c, rope, deterministic=not train)

        x = x[:, self.in_context_len :]
        x = self.final_layer(x, c, deterministic=not train)
        return self.unpatchify(x)


def JiT_B_16(**kwargs) -> JiT:
    return JiT(depth=12, hidden_size=768, num_heads=12, bottleneck_dim=128, in_context_len=32, in_context_start=4, patch_size=16, **kwargs)


def JiT_B_32(**kwargs) -> JiT:
    return JiT(depth=12, hidden_size=768, num_heads=12, bottleneck_dim=128, in_context_len=32, in_context_start=4, patch_size=32, **kwargs)


def JiT_L_16(**kwargs) -> JiT:
    return JiT(depth=24, hidden_size=1024, num_heads=16, bottleneck_dim=128, in_context_len=32, in_context_start=8, patch_size=16, **kwargs)


def JiT_L_32(**kwargs) -> JiT:
    return JiT(depth=24, hidden_size=1024, num_heads=16, bottleneck_dim=128, in_context_len=32, in_context_start=8, patch_size=32, **kwargs)


def JiT_H_16(**kwargs) -> JiT:
    return JiT(depth=32, hidden_size=1280, num_heads=16, bottleneck_dim=256, in_context_len=32, in_context_start=10, patch_size=16, **kwargs)


def JiT_H_32(**kwargs) -> JiT:
    return JiT(depth=32, hidden_size=1280, num_heads=16, bottleneck_dim=256, in_context_len=32, in_context_start=10, patch_size=32, **kwargs)


JiT_models: dict[str, Callable[..., JiT]] = {
    "JiT-B/16": JiT_B_16,
    "JiT-B/32": JiT_B_32,
    "JiT-L/16": JiT_L_16,
    "JiT-L/32": JiT_L_32,
    "JiT-H/16": JiT_H_16,
    "JiT-H/32": JiT_H_32,
}
