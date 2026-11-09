"""Shared image preprocessing."""
from __future__ import annotations

import cv2
import numpy as np


def enhance_image(img: np.ndarray) -> np.ndarray:
    """Histogram-equalize the luminance (Y of YUV) of an RGB image, keeping its colour.

    Accepts either uint8 pixels in [0, 255] or floats in [0, 1] (i.e. after
    ``ImageDataGenerator(rescale=1./255)``) and returns the same convention: uint8 in,
    uint8 out; float in, float32 in [0, 1] out.

    The previous per-script version ran on rescaled floats, equalized Y into [0, 255] and
    wrote it back next to U/V channels still in [0, 1], so the output was ~0-255,
    luminance-only and on a different scale from the network's expected input.
    """
    arr = np.asarray(img)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"expected an HxWx3 RGB image, got shape {arr.shape}")

    is_float = np.issubdtype(arr.dtype, np.floating)
    if is_float:
        if arr.min() < -1e-3 or arr.max() > 1.0 + 1e-3:
            raise ValueError(
                f"float input must be in [0, 1] (got [{arr.min():.3f}, {arr.max():.3f}]); "
                "rescale by 1/255 first or pass uint8 pixels"
            )
        u8 = np.clip(np.rint(arr * 255.0), 0, 255).astype(np.uint8)
    else:
        u8 = np.clip(arr, 0, 255).astype(np.uint8)

    yuv = cv2.cvtColor(u8, cv2.COLOR_RGB2YUV)
    yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
    out = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
    return out.astype(np.float32) / 255.0 if is_float else out
