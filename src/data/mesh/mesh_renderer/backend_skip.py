from __future__ import annotations

from logging import getLogger
from typing import Any, List, Optional, Union

import numpy as onp
import numpy.typing as onpt

LOG = getLogger("[mesh_renderer|skip]")

Tensor = Union[onpt.NDArray[Any], Any]

_g_v: Optional[onpt.NDArray[onp.float32]] = None
_g_f: Optional[onpt.NDArray[onp.int32]] = None
_default_size = (512, 512)


def _to_numpy(x: Tensor, dtype: Any = "float32") -> onpt.NDArray[Any]:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return onp.array(x, dtype=dtype)


def set_template(v: Tensor, f: Tensor, mode: str = "smooth"):
    global _g_f, _g_v
    if (_g_f is None) or (_g_f.shape[0] != f.shape[0]):
        _g_v = _to_numpy(v, "float32")
        _g_f = _to_numpy(f, "int32")
        LOG.info(f"Set template (wgpu disabled): v={_g_v.shape}, f={_g_f.shape}")


def set_image_size(w: int, h: int):
    global _default_size
    _default_size = (w, h)
    LOG.info(f"Set image size to {w}x{h} (wgpu disabled).")


def render(
    position: Tensor,
    mat_model: Optional[Tensor] = None,
    mat_view: Optional[Tensor] = None,
    mat_proj: Optional[Tensor] = None,
    vert_rgb: Optional[Tensor] = None,
    lighting: bool = True,
) -> onpt.NDArray[onp.uint8]:
    # Check initialized like wgpu backend
    if _g_v is None or _g_f is None:
        LOG.warning("Please call `set_template(v, f)` first!")
    
    # Check inputs
    if position.ndim != 2 or position.shape[-1] != 3:
        LOG.warning(f"Invalid shape: {position.shape}")

    w, h = _default_size
    # Return a black image with alpha channel
    return onp.zeros((h, w, 4), dtype=onp.uint8)
