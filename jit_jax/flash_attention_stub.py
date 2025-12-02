"""
Helpers for flash attention dispatch.

We gate flash attention imports because some environments (e.g., older JAX) may
not ship TPU flash kernels. Callers should check `tpu_flash_available`.
"""

from __future__ import annotations

try:
    from jax.experimental.pallas.ops.tpu import flash_attention as tpu_flash_attention  # type: ignore
    tpu_flash_available = True
except Exception:  # pragma: no cover - runtime availability guard
    tpu_flash_attention = None
    tpu_flash_available = False

__all__ = ["tpu_flash_attention", "tpu_flash_available"]
