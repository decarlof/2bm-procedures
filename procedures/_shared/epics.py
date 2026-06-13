"""Thin PyEpics helpers used across procedures.

Kept deliberately small. Each helper wraps one common pattern with a
timeout and a sensible error message so the procedure modules don't
each re-invent the boilerplate.
"""

from __future__ import annotations

import time
import logging

import numpy as np
from epics import PV, caget, caput


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Basic ca operations with timeout + value-readback verification
# ---------------------------------------------------------------------------

def caput_wait(pvname: str, value, dmov_pvname: str | None = None, timeout: float = 30.0):
    """caput + wait for completion.

    If ``dmov_pvname`` is given, wait until that bool PV reads 1
    (typical for motor records — pass ``<motor>.DMOV``). Otherwise
    fall back to ``ca_put(wait=True)`` semantics.
    """
    log.debug("caput %s = %s", pvname, value)
    caput(pvname, value, wait=(dmov_pvname is None), timeout=timeout)
    if dmov_pvname is None:
        return
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if int(caget(dmov_pvname) or 0) == 1:
            return
        time.sleep(0.05)
    raise TimeoutError(f"{dmov_pvname} did not reach DMOV=1 within {timeout} s")


def cawait_value(pvname: str, target, timeout: float = 30.0, tolerance: float = 0.0):
    """Block until ``pvname`` reads ``target`` (numeric within ``tolerance``,
    or exact match for bool / string), or raise ``TimeoutError``."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        val = caget(pvname)
        if val is None:
            time.sleep(0.05)
            continue
        try:
            if abs(float(val) - float(target)) <= tolerance:
                return
        except (TypeError, ValueError):
            if val == target:
                return
        time.sleep(0.05)
    raise TimeoutError(
        f"{pvname} did not reach {target!r} (±{tolerance}) within {timeout} s"
    )


# ---------------------------------------------------------------------------
# Motor convenience: move + wait
# ---------------------------------------------------------------------------

def move_motor(motor_prefix: str, position: float, timeout: float = 60.0):
    """Move a standard EPICS motor record to ``position`` and wait for
    ``<motor>.DMOV == 1``. ``motor_prefix`` is the motor base PV (e.g.
    ``2bmbAERO:m1``); ``.VAL`` and ``.DMOV`` are appended internally."""
    caput_wait(f"{motor_prefix}.VAL", position, dmov_pvname=f"{motor_prefix}.DMOV",
               timeout=timeout)


# ---------------------------------------------------------------------------
# areaDetector image fetch
# ---------------------------------------------------------------------------

def acquire_image(cam_prefix: str, image_prefix: str | None = None,
                  exposure_time: float | None = None, timeout: float = 30.0) -> np.ndarray:
    """Trigger one acquisition on an areaDetector camera and return the
    frame as a 2-D numpy array.

    Parameters
    ----------
    cam_prefix
        Camera areaDetector prefix without the trailing ``cam1:``
        (e.g. ``2bmSP1:`` for the Oryx 5MP). Internally addresses
        ``<cam_prefix>cam1:Acquire`` etc.
    image_prefix
        Image plugin prefix; defaults to ``<cam_prefix>image1:``.
    exposure_time
        If given, set ``cam1:AcquireTime`` before triggering. Otherwise
        leaves the existing value alone.
    """
    image_prefix = image_prefix or f"{cam_prefix}image1:"
    if exposure_time is not None:
        caput_wait(f"{cam_prefix}cam1:AcquireTime", exposure_time)
    caput(f"{cam_prefix}image1:EnableCallbacks", 1)
    caput(f"{cam_prefix}cam1:ImageMode", 0)        # Single
    caput(f"{cam_prefix}cam1:Acquire", 1)
    cawait_value(f"{cam_prefix}cam1:Acquire", 0, timeout=timeout)
    width = int(caget(f"{image_prefix}ArraySize0_RBV"))
    height = int(caget(f"{image_prefix}ArraySize1_RBV"))
    arr = caget(f"{image_prefix}ArrayData", count=width * height)
    if arr is None:
        raise RuntimeError(f"image fetch returned None from {image_prefix}ArrayData")
    return np.asarray(arr).reshape((height, width))
