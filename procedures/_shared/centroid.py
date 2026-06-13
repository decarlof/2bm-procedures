"""Centroid-fit primitives.

Default is centre-of-mass on an above-threshold ROI — robust, fast,
no nonlinear-solver dependency, good enough for the linear-slope
use case the alignment procedures need. Gaussian-fit upgrade lands
here when a procedure asks for it.
"""

from __future__ import annotations

import numpy as np


def center_of_mass(frame: np.ndarray, threshold_fraction: float = 0.5) -> tuple[float, float] | None:
    """Centre of mass of pixels above ``threshold_fraction × max``.

    Returns ``(x_pix, y_pix)`` in pixel coordinates (``x`` = column,
    ``y`` = row), or ``None`` if no pixel is above threshold (no
    signal — caller should treat as a failed measurement).
    """
    frame = np.asarray(frame, dtype=float)
    threshold = float(frame.max()) * float(threshold_fraction)
    if threshold <= 0:
        return None
    mask = frame > threshold
    weights = (frame - threshold).clip(min=0.0) * mask
    total = float(weights.sum())
    if total <= 0:
        return None
    y_idx, x_idx = np.indices(frame.shape)
    x_c = float((weights * x_idx).sum() / total)
    y_c = float((weights * y_idx).sum() / total)
    return (x_c, y_c)


def pixels_to_object_um(pix: tuple[float, float],
                        camera_pixel_um: float = 3.45,
                        magnification: float = 1.1) -> tuple[float, float]:
    """Convert ``(x, y)`` from pixel units to object-side micrometres."""
    scale = camera_pixel_um / magnification
    return (pix[0] * scale, pix[1] * scale)
