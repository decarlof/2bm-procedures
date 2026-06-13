"""B-station slit composite helpers.

The four B-station slit blades are addressed individually at the
motor level (``2bma:m9``/``m10`` vertical pair, ``2bma:m11``/``m12``
horizontal pair); the procedures want the higher-level Size and
Centre handles.

Note the **horizontal-blade label flip** documented in
`2bm-docs/manual/item_020.rst` (B-station Slits block): the on-screen
"B slit Inb" / "B slit Outb" labels are mirrored with respect to
the physical inboard / outboard convention because the detector
image is left-right flipped. Helpers here follow the **physical**
convention, not the on-screen labels.

The exact composite-PV names for ``Size`` and ``Centre`` are
defined by the ``slit.db`` template; we expose them via
``set_aperture()`` which writes to the composite PV directly.
Operators set the prefixes through environment variables or pass
them in so the helper isn't pinned to one IOC naming.
"""

from __future__ import annotations

import os
import logging

from .epics import caput_wait


log = logging.getLogger(__name__)


# Defaults; override via env vars or explicit arguments at call site.
B_SLIT_H_SIZE_PV = os.environ.get("B_SLIT_H_SIZE_PV", "2bma:Slit2H:size")
B_SLIT_H_CENTER_PV = os.environ.get("B_SLIT_H_CENTER_PV", "2bma:Slit2H:center")
B_SLIT_V_SIZE_PV = os.environ.get("B_SLIT_V_SIZE_PV", "2bma:Slit2V:size")
B_SLIT_V_CENTER_PV = os.environ.get("B_SLIT_V_CENTER_PV", "2bma:Slit2V:center")


def set_horizontal_aperture(size_mm: float, center_mm: float | None = None,
                            size_pv: str | None = None,
                            center_pv: str | None = None,
                            timeout: float = 10.0) -> None:
    """Set the B-station horizontal slit aperture in mm; optionally
    re-centre. PV names default to ``B_SLIT_H_*`` module constants
    (env-var overridable)."""
    caput_wait(size_pv or B_SLIT_H_SIZE_PV, size_mm, timeout=timeout)
    if center_mm is not None:
        caput_wait(center_pv or B_SLIT_H_CENTER_PV, center_mm, timeout=timeout)


def set_vertical_aperture(size_mm: float, center_mm: float | None = None,
                          size_pv: str | None = None,
                          center_pv: str | None = None,
                          timeout: float = 10.0) -> None:
    """Set the B-station vertical slit aperture in mm; optionally
    re-centre."""
    caput_wait(size_pv or B_SLIT_V_SIZE_PV, size_mm, timeout=timeout)
    if center_mm is not None:
        caput_wait(center_pv or B_SLIT_V_CENTER_PV, center_mm, timeout=timeout)
