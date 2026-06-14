"""Centroid-fit primitives.

Two algorithms:

- ``center_of_mass``: intensity-weighted COM above a fraction-of-max
  threshold. Best for clean Gaussian-like spots. Biased by internal
  intensity modulation (e.g. multilayer-monochromator stripes,
  saturated pixels) -- the centroid drifts toward whichever feature
  is brightest, not the geometric centre of the illuminated area.

- ``centroid_above_background``: geometric centroid of a binary mask
  built by thresholding against the background statistics. Robust to
  internal modulation -- the mask is "is this pixel above the
  background noise floor?", not "is this pixel one of the brightest".
  The mean of mask coordinates returns the geometric centre of the
  illuminated region regardless of which pixels inside are brightest.
  This is the algorithm the 2-BM detector_z_rail_alignment procedure
  uses, because the DMM produces a beam with strong horizontal
  multilayer-stripe modulation that biases an intensity-weighted COM.
"""

from __future__ import annotations

import numpy as np


def center_of_mass(frame: np.ndarray, threshold_fraction: float = 0.5) -> tuple[float, float] | None:
    """Centre of mass of pixels above ``threshold_fraction × max``.

    Returns ``(x_pix, y_pix)`` in pixel coordinates (``x`` = column,
    ``y`` = row), or ``None`` if no pixel is above threshold (no
    signal — caller should treat as a failed measurement).

    Intensity-weighted: pixel positions are weighted by
    ``(intensity - threshold)``. Biases toward bright features.
    Use ``centroid_above_background`` when the spot has internal
    structure (e.g. multilayer stripes) that would otherwise pull
    the centroid off the geometric centre.
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


def centroid_above_background(
    frame: np.ndarray,
    bg_corner_size: int = 100,
    bg_sigma_threshold: float = 5.0,
) -> tuple[float, float, dict] | None:
    """Geometric centroid of pixels above a background-noise threshold.

    Robust to internal intensity modulation in the spot (multilayer
    stripes, hot pixels, saturated regions) because pixel positions
    are unweighted -- every pixel that passes the threshold gets
    equal vote, so the centroid is the geometric centre of the
    illuminated region, not the intensity-weighted mean.

    Algorithm:

    1. Sample background from the four corners of the frame
       (each ``bg_corner_size × bg_corner_size`` pixels). Compute
       median and MAD (Median Absolute Deviation), then convert
       MAD to a sigma-equivalent (``sigma = 1.4826 * MAD`` for
       Gaussian noise). Median+MAD is robust to outliers --
       a bright feature spilling into one corner won't poison the
       threshold the way mean+std would.
    2. Threshold = ``bg_median + bg_sigma_threshold * sigma_from_mad``.
       Pixels above are classified as "beam"; below as "background".
    3. Centroid = unweighted mean of (x, y) coordinates of beam
       pixels.

    Returns ``(x_pix, y_pix, diag)`` where ``diag`` is a dict of
    diagnostic numbers (threshold, beam-pixel count, bg_median,
    bg_sigma). Returns ``None`` if no pixels are classified as beam
    (no signal -- caller should treat as failed measurement).

    Caveats:

    - If the spot is large enough to actually spill into all four
      corner samples, the threshold will be too high and beam-pixel
      count will drop. Reduce ``bg_corner_size`` (smaller boxes,
      less spot contamination).
    - Hot pixels far from the spot will be flagged as beam and
      pulled into the centroid average. Pre-filtering (median or
      morphological opening) would remove them; not implemented.
    """
    frame = np.asarray(frame, dtype=float)
    h, w = frame.shape
    cs = min(bg_corner_size, h // 4, w // 4)
    if cs < 4:
        return None  # frame too small for meaningful corner stats

    bg_pixels = np.concatenate([
        frame[:cs, :cs].ravel(),
        frame[:cs, w - cs:].ravel(),
        frame[h - cs:, :cs].ravel(),
        frame[h - cs:, w - cs:].ravel(),
    ])
    # Robust background statistics: median + MAD. For Gaussian noise
    # sigma == 1.4826 * MAD; the factor lets us use bg_sigma_threshold
    # in the same "N standard deviations above background" semantic
    # the caller expects, while staying robust to bright outliers
    # spilling into one of the corner samples.
    bg_median = float(np.median(bg_pixels))
    bg_mad = float(np.median(np.abs(bg_pixels - bg_median)))
    bg_sigma = 1.4826 * bg_mad
    threshold = bg_median + bg_sigma_threshold * bg_sigma

    mask = frame > threshold
    n_beam = int(mask.sum())
    if n_beam == 0:
        return None

    y_idx, x_idx = np.indices(frame.shape)
    x_c = float(x_idx[mask].mean())
    y_c = float(y_idx[mask].mean())

    diag = {
        "threshold": threshold,
        "bg_median": bg_median,
        "bg_sigma": bg_sigma,
        "n_beam_pix": n_beam,
        "frame_pix_fraction": n_beam / (h * w),
    }
    return (x_c, y_c, diag)


def pixels_to_object_um(pix: tuple[float, float],
                        camera_pixel_um: float = 3.45,
                        magnification: float = 1.1) -> tuple[float, float]:
    """Convert ``(x, y)`` from pixel units to object-side micrometres."""
    scale = camera_pixel_um / magnification
    return (pix[0] * scale, pix[1] * scale)
