"""
Helpers for flash attention dispatch.

We gate flash attention imports because some environments (e.g., older JAX) may
not ship TPU flash kernels. Callers should check `tpu_flash_available`.
"""

from __future__ import annotations

import inspect

try:
    from jax.experimental.pallas.ops.tpu import flash_attention as tpu_flash_attention  # type: ignore
    tpu_flash_available = True
    _flash_params = inspect.signature(tpu_flash_attention.flash_attention).parameters
    flash_supports_dropout = "dropout_rate" in _flash_params or "dropout_rng" in _flash_params
    _flash_supports_dropout_rate = "dropout_rate" in _flash_params
    _flash_supports_dropout_rng = "dropout_rng" in _flash_params
    _flash_scale_kwarg = "softmax_scale" if "softmax_scale" in _flash_params else ("sm_scale" if "sm_scale" in _flash_params else None)
except Exception:  # pragma: no cover - runtime availability guard
    tpu_flash_attention = None
    tpu_flash_available = False
    _flash_params = {}
    flash_supports_dropout = False
    _flash_supports_dropout_rate = False
    _flash_supports_dropout_rng = False
    _flash_scale_kwarg = None


def call_flash_attention(q, k, v, *, dropout_rng=None, dropout_rate=0.0, softmax_scale=1.0, causal=False):
    if not tpu_flash_available or tpu_flash_attention is None:
        raise RuntimeError("TPU flash attention is unavailable in this environment.")
    kwargs = {"causal": causal}
    if _flash_scale_kwarg:
        kwargs[_flash_scale_kwarg] = softmax_scale
    if _flash_supports_dropout_rate:
        kwargs["dropout_rate"] = dropout_rate
    if _flash_supports_dropout_rng:
        kwargs["dropout_rng"] = dropout_rng
    return tpu_flash_attention.flash_attention(q, k, v, **kwargs)


__all__ = [
    "tpu_flash_attention",
    "tpu_flash_available",
    "call_flash_attention",
    "flash_supports_dropout",
]
