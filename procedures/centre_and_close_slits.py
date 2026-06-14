"""Centre an L3-style slit aperture on the detector, then close it
incrementally (cora slug: ``centre_and_close_slits``).

Two phases run sequentially:

1. **Centre.** Calibrate the 2x2 sensitivity matrix M_centre
   (centroid pixel shift per mm of slit centre move), then iterate
   the slit Hcenter / Vcenter virtual motors to drive the centroid
   to the geometric centre of the detector frame.

2. **Close.** Alternate H / V size reduction in small steps (default
   0.1 mm per step), gated so the operator confirms each one. Stops
   when both H and V reach the target size (default 0 mm, i.e.
   fully closed) or when the centroid fit fails (beam no longer
   visible, expected as the slits close past the beam envelope).

After the procedure ends the slits are left at the closed +
centred state. The operator typically follows up by rezeroing the
slit virtual motors (set Hcenter = Vcenter = Hsize = Vsize = 0)
to define a new origin -- that step is NOT done by this procedure
in v0.0.1.

Either of the 2-BM L3-style slits can be the target via
``--slit-station``: ``A`` (front-end at z=25225 mm) or ``B``
(2-BM-B entrance at z=50500 mm, default). The four virtual-motor
PVs share a common prefix; see :doc:`../manual/item_020`.

The full operator-facing spec is at
``2bm-docs/procedures/item_011.rst``.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from dataclasses import dataclass, field

import numpy as np
from epics import caget, caput

from ._shared.centroid import (
    center_of_mass,
    centroid_above_background,
    pixels_to_object_um,
)
from ._shared.cora_log import CoraProcedureLog
from ._shared.epics import (
    OperatorAbort,
    acquire_image,
    confirm_motion,
    move_table_axis,
    safe_restore,
)
from ._shared.log import setup_console_logger


# ---------------------------------------------------------------------------
# PV constants
# ---------------------------------------------------------------------------

# Slit-station prefixes. The four virtual-motor PVs for each station
# are <PREFIX>Hsize, <PREFIX>Hcenter, <PREFIX>Vsize, <PREFIX>Vcenter
# (all ao records on the IOC; no colon between H/V and size/center).
# Underlying blade motors aggregate the same calc records the virtual
# motors drive.
SLIT_STATIONS = {
    "A": {
        # Front-end L3 Slits at z = 25225 mm, per item_020.rst.
        "prefix": "2bma:Slit1",
        "blade_prefixes": ("2bma:m1", "2bma:m2", "2bma:m3", "2bma:m4"),
    },
    "B": {
        # B-station L3-style slits at z = 50500 mm.
        "prefix": "2bma:Slit2",
        "blade_prefixes": ("2bma:m9", "2bma:m10", "2bma:m11", "2bma:m12"),
    },
}

# MCTOptics IOC (operator-set; procedure reads only).
PV_LENS_SELECT = "2bm:MCTOptics:LensSelect"
PV_CAMERA_SELECT = "2bm:MCTOptics:CameraSelect"

CAMERA_PREFIXES_BY_INDEX = {
    0: "2bmSP1:",
    1: "2bmSP2:",
}
LENS_MAGNIFICATIONS_BY_INDEX = {
    0: 1.1,
    1: 5.0,
    2: 10.0,
}

RAIL_STRAIGHTNESS_FLOOR_PIX = 1.0  # placeholder; centring threshold is in pixels


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration + state
# ---------------------------------------------------------------------------

@dataclass
class Config:
    slit_station: str = "B"

    # Phase 1: centring
    centring_step_mm: float = 0.5      # perturbation for M calibration
    centring_threshold_pix: float = 5.0  # centroid within N pixels of centre
    centring_max_iterations: int = 5
    centring_damping: float = 0.5
    centring_max_correction_mm: float = 1.0  # clip per iteration per axis
    centring_min_sensitivity: float = 1.0    # min |det(M)| (pix/mm)^2

    # Phase 2: closing
    closing_step_mm: float = 0.1       # incremental size reduction
    target_h_size_mm: float = 0.0
    target_v_size_mm: float = 0.0

    # Image acquisition
    exposure_time: float = 0.2
    centroid_algorithm: str = "com"
    threshold_fraction: float = 0.5
    bg_corner_size: int = 100
    bg_sigma_threshold: float = 5.0
    frames_per_measurement: int = 1
    camera_pixel_um: float = 3.45

    # Operator UX
    dry_run: bool = False
    auto_yes: bool = False
    confirm_restore: bool = False
    enable_cora_log: bool = True


@dataclass
class CentringSensitivity:
    """2x2 centroid-pixel-shift per slit-centre-mm matrix.

        Δcentroid_x_pix = M_Hc_x · ΔHcenter_mm + M_Vc_x · ΔVcenter_mm
        Δcentroid_y_pix = M_Hc_y · ΔHcenter_mm + M_Vc_y · ΔVcenter_mm
    """
    M_Hc_x: float = 0.0
    M_Hc_y: float = 0.0
    M_Vc_x: float = 0.0
    M_Vc_y: float = 0.0

    def as_matrix(self) -> np.ndarray:
        return np.array([[self.M_Hc_x, self.M_Vc_x],
                         [self.M_Hc_y, self.M_Vc_y]])

    def determinant(self) -> float:
        return self.M_Hc_x * self.M_Vc_y - self.M_Vc_x * self.M_Hc_y


@dataclass
class IterationResult:
    iteration: int
    centroid_x_pix: float
    centroid_y_pix: float
    error_x_pix: float
    error_y_pix: float
    correction_Hc_mm: float = 0.0
    correction_Vc_mm: float = 0.0
    converged: bool = False


@dataclass
class _Snapshot:
    """Pre-procedure state for restore-on-exit.

    Slit Hcenter, Vcenter, Hsize, Vsize are restored on any
    non-success exit. On clean convergence they're left at the
    new values (procedure's deliberate output).
    """
    cam_prefix: str = ""
    cam_was_acquiring: bool = False
    cam_acquire_time: float = 0.0
    cam_num_images: int = 1
    cam_image_mode: str = ""
    cam_trigger_mode: str = ""
    cam_trigger_source: str = ""
    cam_trigger_overlap: str = ""
    cam_exposure_mode: str = ""
    cam_array_callbacks: str = ""
    slit_h_center_mm: float = 0.0
    slit_v_center_mm: float = 0.0
    slit_h_size_mm: float = 0.0
    slit_v_size_mm: float = 0.0
    slit_prefix: str = ""
    slit_blade_prefixes: tuple = ()

    @classmethod
    def capture(cls, cam_prefix: str, slit_prefix: str,
                slit_blade_prefixes: tuple) -> "_Snapshot":
        def s(field):
            return caget(f"{cam_prefix}cam1:{field}", as_string=True)
        return cls(
            cam_prefix=cam_prefix,
            cam_was_acquiring=bool(caget(f"{cam_prefix}cam1:Acquire")),
            cam_acquire_time=float(caget(f"{cam_prefix}cam1:AcquireTime")),
            cam_num_images=int(caget(f"{cam_prefix}cam1:NumImages")),
            cam_image_mode=s("ImageMode"),
            cam_trigger_mode=s("TriggerMode"),
            cam_trigger_source=s("TriggerSource"),
            cam_trigger_overlap=s("TriggerOverlap"),
            cam_exposure_mode=s("ExposureMode"),
            cam_array_callbacks=s("ArrayCallbacks"),
            slit_h_center_mm=float(caget(f"{slit_prefix}Hcenter")),
            slit_v_center_mm=float(caget(f"{slit_prefix}Vcenter")),
            slit_h_size_mm=float(caget(f"{slit_prefix}Hsize")),
            slit_v_size_mm=float(caget(f"{slit_prefix}Vsize")),
            slit_prefix=slit_prefix,
            slit_blade_prefixes=tuple(slit_blade_prefixes),
        )

    def restore_plan(self, restore_slits: bool = True) -> list[dict]:
        cp = self.cam_prefix
        plan: list[dict] = []
        if restore_slits:
            plan.extend([
                {"pv": f"{self.slit_prefix}Hcenter", "current": "?",
                 "target": self.slit_h_center_mm, "units": "mm"},
                {"pv": f"{self.slit_prefix}Vcenter", "current": "?",
                 "target": self.slit_v_center_mm, "units": "mm"},
                {"pv": f"{self.slit_prefix}Hsize", "current": "?",
                 "target": self.slit_h_size_mm, "units": "mm"},
                {"pv": f"{self.slit_prefix}Vsize", "current": "?",
                 "target": self.slit_v_size_mm, "units": "mm"},
            ])
        plan.extend([
            {"pv": f"{cp}cam1:Acquire", "current": "?", "target": 0,
             "units": "(stop if running)"},
            {"pv": f"{cp}cam1:TriggerMode", "current": "?",
             "target": self.cam_trigger_mode, "units": ""},
            {"pv": f"{cp}cam1:ImageMode", "current": "?",
             "target": self.cam_image_mode, "units": ""},
            {"pv": f"{cp}cam1:NumImages", "current": "?",
             "target": self.cam_num_images, "units": ""},
            {"pv": f"{cp}cam1:AcquireTime", "current": "?",
             "target": self.cam_acquire_time, "units": "s"},
            {"pv": f"{cp}cam1:TriggerSource", "current": "?",
             "target": self.cam_trigger_source, "units": ""},
            {"pv": f"{cp}cam1:TriggerOverlap", "current": "?",
             "target": self.cam_trigger_overlap, "units": ""},
            {"pv": f"{cp}cam1:ExposureMode", "current": "?",
             "target": self.cam_exposure_mode, "units": ""},
            {"pv": f"{cp}cam1:ArrayCallbacks", "current": "?",
             "target": self.cam_array_callbacks, "units": ""},
        ])
        if self.cam_was_acquiring:
            plan.append({"pv": f"{cp}cam1:Acquire", "current": 0,
                         "target": 1, "units": "(resume)"})
        return plan

    def restore(self, restore_slits: bool = True) -> None:
        """Restore slits first (highest stakes), then camera."""
        cp = self.cam_prefix
        actions = []
        if restore_slits:
            # Drive each composite back via the soft PV; the blade
            # motors track. Reuse move_table_axis (same algorithm:
            # write soft PV, poll underlying motors).
            actions.extend([
                (f"slit Hcenter -> {self.slit_h_center_mm}",
                 lambda: move_table_axis(
                     f"{self.slit_prefix}Hcenter",
                     self.slit_h_center_mm,
                     self.slit_blade_prefixes, timeout=30)),
                (f"slit Vcenter -> {self.slit_v_center_mm}",
                 lambda: move_table_axis(
                     f"{self.slit_prefix}Vcenter",
                     self.slit_v_center_mm,
                     self.slit_blade_prefixes, timeout=30)),
                (f"slit Hsize -> {self.slit_h_size_mm}",
                 lambda: move_table_axis(
                     f"{self.slit_prefix}Hsize",
                     self.slit_h_size_mm,
                     self.slit_blade_prefixes, timeout=30)),
                (f"slit Vsize -> {self.slit_v_size_mm}",
                 lambda: move_table_axis(
                     f"{self.slit_prefix}Vsize",
                     self.slit_v_size_mm,
                     self.slit_blade_prefixes, timeout=30)),
            ])
        actions.extend([
            ("stop in-progress acquire",
             lambda: caput(f"{cp}cam1:Acquire", 0, wait=True, timeout=5.0)),
            ("cam TriggerMode",
             lambda: caput(f"{cp}cam1:TriggerMode",
                           self.cam_trigger_mode, wait=True)),
            ("cam ImageMode",
             lambda: caput(f"{cp}cam1:ImageMode",
                           self.cam_image_mode, wait=True)),
            ("cam NumImages",
             lambda: caput(f"{cp}cam1:NumImages",
                           self.cam_num_images, wait=True)),
            ("cam AcquireTime",
             lambda: caput(f"{cp}cam1:AcquireTime",
                           self.cam_acquire_time, wait=True)),
            ("cam TriggerSource",
             lambda: caput(f"{cp}cam1:TriggerSource",
                           self.cam_trigger_source, wait=True)),
            ("cam TriggerOverlap",
             lambda: caput(f"{cp}cam1:TriggerOverlap",
                           self.cam_trigger_overlap, wait=True)),
            ("cam ExposureMode",
             lambda: caput(f"{cp}cam1:ExposureMode",
                           self.cam_exposure_mode, wait=True)),
            ("cam ArrayCallbacks",
             lambda: caput(f"{cp}cam1:ArrayCallbacks",
                           self.cam_array_callbacks, wait=True)),
        ])
        if self.cam_was_acquiring:
            actions.append(("resume continuous acquire",
                            lambda: caput(f"{cp}cam1:Acquire", 1)))
        safe_restore(actions)


# ---------------------------------------------------------------------------
# The procedure
# ---------------------------------------------------------------------------

class CentreAndCloseSlits:
    """Stateful executor for ``centre_and_close_slits``."""

    def __init__(self, config: Config) -> None:
        self.config = config
        if config.slit_station not in SLIT_STATIONS:
            raise ValueError(
                f"unknown slit_station {config.slit_station!r}; "
                f"expected one of {list(SLIT_STATIONS)}"
            )
        s = SLIT_STATIONS[config.slit_station]
        self.slit_prefix = s["prefix"]
        self.slit_blade_prefixes = s["blade_prefixes"]
        self.sensitivity = CentringSensitivity()
        self.history: list[IterationResult] = []
        self._snapshot: _Snapshot | None = None
        self._cam_prefix: str = ""
        self._magnification: float = 1.0
        self._pixel_um: float = config.camera_pixel_um
        self._frame_center_x: float = 0.0
        self._frame_center_y: float = 0.0
        self._centring_committed: bool = False

    # ---- detection -------------------------------------------------------

    def detect_camera_and_lens(self) -> None:
        cam_idx_raw = caget(PV_CAMERA_SELECT)
        lens_idx_raw = caget(PV_LENS_SELECT)
        if cam_idx_raw is None:
            raise RuntimeError(
                f"could not read {PV_CAMERA_SELECT} -- is MCTOptics IOC "
                "reachable from this host?"
            )
        if lens_idx_raw is None:
            raise RuntimeError(f"could not read {PV_LENS_SELECT}")
        cam_idx = int(cam_idx_raw)
        lens_idx = int(lens_idx_raw)
        cam_label = caget(PV_CAMERA_SELECT, as_string=True) or f"idx {cam_idx}"
        lens_label = caget(PV_LENS_SELECT, as_string=True) or f"idx {lens_idx}"
        if cam_idx not in CAMERA_PREFIXES_BY_INDEX:
            raise RuntimeError(f"unknown camera idx {cam_idx} ({cam_label!r})")
        if lens_idx not in LENS_MAGNIFICATIONS_BY_INDEX:
            raise RuntimeError(f"unknown lens idx {lens_idx} ({lens_label!r})")
        self._cam_prefix = CAMERA_PREFIXES_BY_INDEX[cam_idx]
        self._magnification = LENS_MAGNIFICATIONS_BY_INDEX[lens_idx]
        bin_x = int(caget(f"{self._cam_prefix}cam1:BinX_RBV") or 1)
        bin_y = int(caget(f"{self._cam_prefix}cam1:BinY_RBV") or 1)
        if bin_x != bin_y:
            log.warning("camera BinX=%d != BinY=%d", bin_x, bin_y)
        self._pixel_um = self.config.camera_pixel_um * bin_x
        # Compute frame centre in pixel coordinates (used as the
        # centring target).
        width = int(caget(f"{self._cam_prefix}cam1:SizeX_RBV"))
        height = int(caget(f"{self._cam_prefix}cam1:SizeY_RBV"))
        self._frame_center_x = width / 2.0
        self._frame_center_y = height / 2.0
        log.info("detected: camera=%s [idx %d] -> %s (%dx%d, bin %dx%d, "
                 "pitch %.2f um); lens=%s [idx %d] -> %.2fx; "
                 "frame centre = (%.1f, %.1f) pix; slit=%s -> %s",
                 cam_label, cam_idx, self._cam_prefix, width, height,
                 bin_x, bin_y, self._pixel_um, lens_label, lens_idx,
                 self._magnification, self._frame_center_x,
                 self._frame_center_y, self.config.slit_station,
                 self.slit_prefix)

    # ---- gated motion helpers --------------------------------------------

    def _gate(self, plan: list[dict], step_label: str) -> bool:
        return confirm_motion(
            plan, step_label=step_label,
            dry_run=self.config.dry_run, auto_yes=self.config.auto_yes,
        )

    def _gated_move_slit(self, suffix: str, target_mm: float,
                         step_label: str) -> None:
        pv = f"{self.slit_prefix}{suffix}"
        current = float(caget(pv))
        proceed = self._gate(
            [{"pv": pv, "current": current,
              "target": target_mm, "units": "mm"}],
            step_label=step_label,
        )
        if proceed:
            move_table_axis(pv, target_mm, self.slit_blade_prefixes,
                            timeout=30)

    def _gated_move_centre_pair(self, hc_target: float, vc_target: float,
                                step_label: str) -> None:
        hc_now = float(caget(f"{self.slit_prefix}Hcenter"))
        vc_now = float(caget(f"{self.slit_prefix}Vcenter"))
        proceed = self._gate(
            [
                {"pv": f"{self.slit_prefix}Hcenter", "current": hc_now,
                 "target": hc_target, "units": "mm"},
                {"pv": f"{self.slit_prefix}Vcenter", "current": vc_now,
                 "target": vc_target, "units": "mm"},
            ],
            step_label=step_label,
        )
        if proceed:
            move_table_axis(f"{self.slit_prefix}Hcenter", hc_target,
                            self.slit_blade_prefixes, timeout=30)
            move_table_axis(f"{self.slit_prefix}Vcenter", vc_target,
                            self.slit_blade_prefixes, timeout=30)

    def _measure_centroid(self) -> tuple[float, float] | None:
        """Acquire (and optionally average) frames; run centroid.

        Returns ``(px, py)`` in pixel coordinates, or ``None`` if the
        beam is no longer visible (algorithm returns no above-threshold
        pixels). The closing phase relies on the None case to detect
        when the slits have closed past the beam.
        """
        c = self.config
        n = max(1, int(c.frames_per_measurement))
        if n == 1:
            frame = acquire_image(self._cam_prefix,
                                  exposure_time=c.exposure_time)
        else:
            f0 = acquire_image(self._cam_prefix, exposure_time=c.exposure_time)
            stack = np.empty((n,) + f0.shape, dtype=np.float32)
            stack[0] = f0
            for i in range(1, n):
                stack[i] = acquire_image(self._cam_prefix,
                                         exposure_time=c.exposure_time)
            frame = np.mean(stack, axis=0)

        if c.centroid_algorithm == "com":
            com = center_of_mass(frame, c.threshold_fraction)
            if com is None:
                return None
            px, py = com
            diag = None
        elif c.centroid_algorithm == "binmask":
            result = centroid_above_background(
                frame, bg_corner_size=c.bg_corner_size,
                bg_sigma_threshold=c.bg_sigma_threshold,
            )
            if result is None:
                return None
            px, py, diag = result
        else:
            raise ValueError(f"unknown centroid_algorithm {c.centroid_algorithm!r}")

        h, w = frame.shape
        dx = px - w / 2.0
        dy = py - h / 2.0
        avg_tag = f"avg{n}" if n > 1 else "1f"
        if diag:
            log.info("centroid[%s,%s]: pix=(%.1f, %.1f); "
                     "offset-from-centre=(%+.1f, %+.1f) pix; "
                     "beam=%d pix (%.2f%%), threshold=%.0f",
                     c.centroid_algorithm, avg_tag, px, py, dx, dy,
                     diag["n_beam_pix"], 100 * diag["frame_pix_fraction"],
                     diag["threshold"])
        else:
            log.info("centroid[%s,%s]: pix=(%.1f, %.1f); "
                     "offset-from-centre=(%+.1f, %+.1f) pix",
                     c.centroid_algorithm, avg_tag, px, py, dx, dy)
        return (px, py)

    # ---- phase 1: centring -----------------------------------------------

    def calibrate_centring_sensitivity(self) -> None:
        """Perturb Hcenter then Vcenter; measure centroid shift each
        time; build the 2x2 sensitivity matrix M."""
        c = self.config
        delta = c.centring_step_mm
        log.info("calibrate centring sensitivity at slit=%s with "
                 "delta=%.3f mm", self.slit_prefix, delta)

        base = self._measure_centroid()
        if base is None:
            raise RuntimeError(self._centroid_failure_message(
                "no signal at baseline; cannot calibrate"))
        cx0, cy0 = base

        # Perturb Hcenter
        hc_baseline = self._snapshot.slit_h_center_mm
        self._gated_move_slit("Hcenter", hc_baseline + delta,
                              f"calibration: perturb Hcenter by +{delta:.3f} mm")
        after_hc = self._measure_centroid()
        self._gated_move_slit("Hcenter", hc_baseline,
                              "calibration: restore Hcenter")
        if after_hc is None:
            raise RuntimeError(self._centroid_failure_message(
                "no signal after Hcenter perturb"))
        cx_hc, cy_hc = after_hc

        # Perturb Vcenter
        vc_baseline = self._snapshot.slit_v_center_mm
        self._gated_move_slit("Vcenter", vc_baseline + delta,
                              f"calibration: perturb Vcenter by +{delta:.3f} mm")
        after_vc = self._measure_centroid()
        self._gated_move_slit("Vcenter", vc_baseline,
                              "calibration: restore Vcenter")
        if after_vc is None:
            raise RuntimeError(self._centroid_failure_message(
                "no signal after Vcenter perturb"))
        cx_vc, cy_vc = after_vc

        self.sensitivity = CentringSensitivity(
            M_Hc_x=(cx_hc - cx0) / delta,
            M_Hc_y=(cy_hc - cy0) / delta,
            M_Vc_x=(cx_vc - cx0) / delta,
            M_Vc_y=(cy_vc - cy0) / delta,
        )
        M = self.sensitivity
        det = M.determinant()
        log.info("centring sensitivity M (pix of centroid per mm of slit centre):")
        log.info("  d_cx = %+.2f * dHc + %+.2f * dVc", M.M_Hc_x, M.M_Vc_x)
        log.info("  d_cy = %+.2f * dHc + %+.2f * dVc", M.M_Hc_y, M.M_Vc_y)
        log.info("  det(M) = %+.4e", det)
        try:
            sv = np.linalg.svd(M.as_matrix(), compute_uv=False)
            cond = sv[0] / sv[1] if sv[1] > 0 else float("inf")
            log.info("  singular values: %.3e, %.3e   cond: %.1f",
                     sv[0], sv[1], cond)
            if cond > 10:
                log.warning("  elevated condition number %.1f -- the "
                            "calibration may be marginal", cond)
        except Exception:
            pass
        if not c.dry_run and abs(det) < c.centring_min_sensitivity:
            raise RuntimeError(
                f"centring sensitivity near-singular (|det|={abs(det):.4e} "
                f"< min={c.centring_min_sensitivity}). Try larger "
                "--centring-step-mm.")

    def centre_iterate(self) -> bool:
        """Drive (Hcenter, Vcenter) so the centroid lands at the frame
        centre. Returns True on convergence within threshold."""
        c = self.config
        M = self.sensitivity.as_matrix()
        try:
            M_inv = np.linalg.inv(M)
        except np.linalg.LinAlgError as exc:
            raise RuntimeError(f"M not invertible: {exc}") from exc
        log.info("centre iterate (max=%d, threshold=%.1f pix, "
                 "damping=%.2f)", c.centring_max_iterations,
                 c.centring_threshold_pix, c.centring_damping)

        prev_err = None
        best_err = float("inf")
        best_Hc = self._snapshot.slit_h_center_mm
        best_Vc = self._snapshot.slit_v_center_mm

        for i in range(1, c.centring_max_iterations + 1):
            now = self._measure_centroid()
            if now is None:
                raise RuntimeError(self._centroid_failure_message(
                    f"no signal at centring iter {i}"))
            cx, cy = now
            err_x = cx - self._frame_center_x
            err_y = cy - self._frame_center_y
            err_mag = math.hypot(err_x, err_y)
            converged = (abs(err_x) <= c.centring_threshold_pix
                         and abs(err_y) <= c.centring_threshold_pix)

            current_Hc = float(caget(f"{self.slit_prefix}Hcenter"))
            current_Vc = float(caget(f"{self.slit_prefix}Vcenter"))
            if err_mag < best_err:
                best_err = err_mag
                best_Hc = current_Hc
                best_Vc = current_Vc

            log.info("  iter %d: centroid=(%.1f, %.1f); error=(%+.1f, "
                     "%+.1f) pix; |err|=%.1f", i, cx, cy, err_x, err_y, err_mag)

            if converged:
                self.history.append(IterationResult(
                    iteration=i, centroid_x_pix=cx, centroid_y_pix=cy,
                    error_x_pix=err_x, error_y_pix=err_y, converged=True,
                ))
                log.info("  iter %d: CONVERGED", i)
                return True

            # Simple divergence guard
            if prev_err is not None and err_mag > prev_err * 1.5 and not c.dry_run:
                raise RuntimeError(
                    f"centring diverging at iter {i}: |err|={err_mag:.1f} > "
                    f"1.5x prev {prev_err:.1f}; M may be wrong."
                )
            prev_err = err_mag

            # Compute correction. M @ d = -error -> d = M_inv @ (-error).
            d = M_inv @ np.array([-err_x, -err_y])
            cap = c.centring_max_correction_mm
            d_Hc = max(-cap, min(cap, float(d[0]) * c.centring_damping))
            d_Vc = max(-cap, min(cap, float(d[1]) * c.centring_damping))

            new_Hc = current_Hc + d_Hc
            new_Vc = current_Vc + d_Vc
            self.history.append(IterationResult(
                iteration=i, centroid_x_pix=cx, centroid_y_pix=cy,
                error_x_pix=err_x, error_y_pix=err_y,
                correction_Hc_mm=d_Hc, correction_Vc_mm=d_Vc,
            ))
            log.info("  iter %d: dHc=%+.4f mm, dVc=%+.4f mm (damped %.2fx)",
                     i, d_Hc, d_Vc, c.centring_damping)
            self._gated_move_centre_pair(
                new_Hc, new_Vc,
                f"centre iter {i}: drive slit centre toward frame centre")

        log.warning("centring did not converge after %d iterations",
                    c.centring_max_iterations)
        log.info("best |err|=%.1f at Hcenter=%.4f, Vcenter=%.4f",
                 best_err, best_Hc, best_Vc)
        # Mark phase 1 as committed if we made any improvement; this
        # tells the finally block whether to restore Hcenter/Vcenter.
        starting_err = math.hypot(
            self.history[0].error_x_pix if self.history else 0,
            self.history[0].error_y_pix if self.history else 0)
        if best_err < starting_err:
            self._centring_committed = True
        return False

    # ---- phase 2: closing -------------------------------------------------

    def close_slits(self) -> None:
        """Alternating H / V size reduction to target_h_size_mm /
        target_v_size_mm. Stops if the centroid algorithm returns
        None (beam no longer visible) or both targets reached.
        """
        c = self.config
        log.info("close slits to target H=%.3f mm, V=%.3f mm in %.3f mm "
                 "steps", c.target_h_size_mm, c.target_v_size_mm,
                 c.closing_step_mm)

        h = float(caget(f"{self.slit_prefix}Hsize"))
        v = float(caget(f"{self.slit_prefix}Vsize"))
        i = 0
        while h > c.target_h_size_mm + 1e-6 or v > c.target_v_size_mm + 1e-6:
            i += 1
            if h > c.target_h_size_mm + 1e-6:
                new_h = max(c.target_h_size_mm, h - c.closing_step_mm)
                self._gated_move_slit("Hsize", new_h,
                                       f"close step {i}: H size %.3f -> %.3f mm"
                                       % (h, new_h))
                h = new_h
                cent = self._measure_centroid()
                if cent is None:
                    log.info("beam no longer visible at H=%.3f mm "
                             "(centroid fit returned None) -- "
                             "slits effectively closed past beam.", h)
                    break
            if v > c.target_v_size_mm + 1e-6:
                new_v = max(c.target_v_size_mm, v - c.closing_step_mm)
                self._gated_move_slit("Vsize", new_v,
                                       f"close step {i}: V size %.3f -> %.3f mm"
                                       % (v, new_v))
                v = new_v
                cent = self._measure_centroid()
                if cent is None:
                    log.info("beam no longer visible at V=%.3f mm "
                             "(centroid fit returned None) -- "
                             "slits effectively closed past beam.", v)
                    break
        log.info("closing complete: H=%.3f, V=%.3f mm",
                 float(caget(f"{self.slit_prefix}Hsize")),
                 float(caget(f"{self.slit_prefix}Vsize")))

    # ---- shared helpers ---------------------------------------------------

    def _centroid_failure_message(self, reason: str) -> str:
        return (
            f"centroid fit failed: {reason}. "
            "Likely upstream-precondition failures (check, in order):\n"
            "  1. FES shutter open?  caget S02BM-PSS:FES:BeamBlockingM\n"
            "  2. B-shutter open?    caget S02BM-PSS:SBS:BeamBlockingM\n"
            "  3. Beam on / DMM at energy?\n"
            "  4. Slits open enough to admit beam?\n"
            "  5. Sample out of beam path?\n"
            "  6. Detector in position to see beam?"
        )

    # ---- orchestrator -----------------------------------------------------

    def run(self) -> bool:
        c = self.config
        self.detect_camera_and_lens()
        self._snapshot = _Snapshot.capture(
            self._cam_prefix, self.slit_prefix, self.slit_blade_prefixes)
        log.info("snapshotted pre-procedure state (slits + camera); "
                 "will restore on exit")
        log.info("  slit: Hcenter=%.4f Vcenter=%.4f Hsize=%.4f Vsize=%.4f mm",
                 self._snapshot.slit_h_center_mm,
                 self._snapshot.slit_v_center_mm,
                 self._snapshot.slit_h_size_mm,
                 self._snapshot.slit_v_size_mm)

        cora = (CoraProcedureLog(
            slug="centre_and_close_slits",
            target_asset_ids=[
                f"{c.slit_station}_station_slits",  # pending cora
            ],
            parameters=vars(c),
        ) if c.enable_cora_log else None)
        if cora:
            cora.open()

        converged = False
        closed = False
        try:
            self.calibrate_centring_sensitivity()
            if cora:
                cora.append_step("calibrate_centring", vars(self.sensitivity))
            converged = self.centre_iterate()
            if cora:
                cora.append_step("centre_iterate",
                                 {"iterations": len(self.history),
                                  "converged": converged})
            if converged:
                # Phase 1 succeeded; remember so finally doesn't restore centre.
                self._centring_committed = True
                self.close_slits()
                closed = True
                if cora:
                    cora.append_step("close_slits", {"closed": True})
            else:
                log.warning("skipping close phase because centring did not "
                            "converge to threshold (best state was committed)")
            return converged and closed
        except OperatorAbort as exc:
            log.warning("operator aborted: %s", exc)
            if cora:
                cora.append_step("abort", {"reason": str(exc)})
            return False
        finally:
            if self._snapshot is not None:
                # On clean completion (centred + closed): leave slits as-is
                # (the procedure's deliberate output). Otherwise restore.
                # If centring committed but we aborted in phase 2, we'd
                # ideally keep the new centre and restore only the size;
                # for v0.0.1 the simpler "restore everything on abort"
                # is the rule. Operator can re-run.
                restore_slits = not (converged and closed)
                log.info("restoring pre-procedure state (slits restored: %s)",
                         "yes" if restore_slits else
                         "no -- procedure completed cleanly, keeping new state")
                confirm_motion(
                    self._snapshot.restore_plan(restore_slits=restore_slits),
                    step_label="restore: returning %scamera to pre-procedure "
                               "state" % ("slits + " if restore_slits else ""),
                    dry_run=False, auto_yes=c.auto_yes,
                    announce_only=(not c.confirm_restore),
                )
                self._snapshot.restore(restore_slits=restore_slits)
            if cora:
                outcome = "complete" if (converged and closed) else "truncate"
                cora.close(outcome=outcome)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Centre an L3-style slit aperture on the detector, "
                    "then close it incrementally to a target size. "
                    "Two phases: (1) calibrate + iterate the slit "
                    "Hcenter/Vcenter virtual motors to drive the spot to "
                    "the centre of the frame; (2) shrink Hsize/Vsize in "
                    "small alternating steps to the target size.")
    p.add_argument("--slit-station", choices=list(SLIT_STATIONS), default="B",
                   help="Which slit station: A (front-end L3 at z=25225 mm) "
                        "or B (2-BM-B entrance at z=50500 mm). Default: B.")
    # Phase 1
    p.add_argument("--centring-step-mm", type=float, default=0.5,
                   help="Perturbation applied to Hcenter / Vcenter for "
                        "sensitivity calibration. Default: 0.5 mm.")
    p.add_argument("--centring-threshold-pix", type=float, default=5.0,
                   help="Centroid must be within N pixels of frame centre "
                        "in both axes to declare centring converged. "
                        "Default: 5 pix.")
    p.add_argument("--centring-max-iterations", type=int, default=5,
                   help="Default: 5.")
    p.add_argument("--centring-damping", type=float, default=0.5,
                   help="Damping factor on the per-iter centring correction. "
                        "Default: 0.5.")
    p.add_argument("--centring-max-correction-mm", type=float, default=1.0,
                   help="Clip on per-iter |dHcenter|, |dVcenter|. "
                        "Default: 1.0 mm.")
    # Phase 2
    p.add_argument("--closing-step-mm", type=float, default=0.1,
                   help="Size reduction per closing step (alternating "
                        "H / V). Default: 0.1 mm.")
    p.add_argument("--target-h-size-mm", type=float, default=0.0,
                   help="Final H aperture. Default: 0 (fully closed).")
    p.add_argument("--target-v-size-mm", type=float, default=0.0,
                   help="Final V aperture. Default: 0 (fully closed).")
    # Image acquisition
    p.add_argument("--exposure-time", type=float, default=0.2,
                   help="Camera exposure (s). Default: 0.2.")
    p.add_argument("--centroid-algorithm", choices=["com", "binmask"],
                   default="com",
                   help="Default: com (intensity-weighted centre of mass).")
    p.add_argument("--threshold-fraction", type=float, default=0.5,
                   help="(com only) Threshold as fraction of frame max. "
                        "Default: 0.5.")
    p.add_argument("--bg-corner-size", type=int, default=100,
                   help="(binmask only) Default: 100.")
    p.add_argument("--bg-sigma-threshold", type=float, default=5.0,
                   help="(binmask only) Default: 5.0.")
    p.add_argument("--frames-per-measurement", type=int, default=1,
                   help="Acquire and average N frames per centroid. "
                        "Default: 1.")
    p.add_argument("--camera-pixel-um", type=float, default=3.45,
                   help="Camera sensor pixel pitch, pre-binning. "
                        "Default: 3.45.")
    # Operator UX
    p.add_argument("--yes", action="store_true",
                   help="Auto-confirm every motion prompt.")
    p.add_argument("--confirm-restore", action="store_true",
                   help="Also gate the restore path.")
    p.add_argument("--no-cora-log", action="store_true",
                   help="Skip cora Procedure-record logging.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print plan + skip every motion.")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)
    setup_console_logger(level=args.log_level)
    config = Config(
        slit_station=args.slit_station,
        centring_step_mm=args.centring_step_mm,
        centring_threshold_pix=args.centring_threshold_pix,
        centring_max_iterations=args.centring_max_iterations,
        centring_damping=args.centring_damping,
        centring_max_correction_mm=args.centring_max_correction_mm,
        closing_step_mm=args.closing_step_mm,
        target_h_size_mm=args.target_h_size_mm,
        target_v_size_mm=args.target_v_size_mm,
        exposure_time=args.exposure_time,
        centroid_algorithm=args.centroid_algorithm,
        threshold_fraction=args.threshold_fraction,
        bg_corner_size=args.bg_corner_size,
        bg_sigma_threshold=args.bg_sigma_threshold,
        frames_per_measurement=args.frames_per_measurement,
        camera_pixel_um=args.camera_pixel_um,
        dry_run=args.dry_run,
        auto_yes=args.yes,
        confirm_restore=args.confirm_restore,
        enable_cora_log=(not args.no_cora_log),
    )
    proc = CentreAndCloseSlits(config)
    try:
        ok = proc.run()
    except Exception as exc:
        log.error("procedure failed: %s", exc, exc_info=True)
        return 2
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
