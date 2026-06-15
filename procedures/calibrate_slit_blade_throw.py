"""Calibrate each L3-style slit blade motor by measuring how far its
corresponding spot edge moves per mm of motor throw (cora slug:
``calibrate_slit_blade_throw``).

For each of the four blade motors in the chosen slit station, the
procedure:

1. Snapshots the blade's baseline position.
2. Moves the blade by ``+blade_throw_mm`` (gated), acquires an image
   and measures the spot bounding box.
3. Moves the blade by ``-blade_throw_mm`` from baseline (full motor
   throw is ``2 * blade_throw_mm``), gated. Acquires + measures bbox.
4. Moves the blade back to baseline (gated).
5. Compares the two bboxes -- the edge that shifted the most is the
   one the blade controls. Computes
   ``slope_pix_per_mm = pixel_shift / (-2 * blade_throw_mm)``.

After all 4 blades:

- The two H blades should agree on ``|slope|`` (same physical axis).
- The two V blades should agree on ``|slope|``.
- H and V should also agree if all four blade motors share the same
  underlying mm-per-encoder-step calibration.

A slope that deviates from its same-axis partner, or a V/H mean
ratio noticeably off 1.0, points at the mis-calibrated blade.

The procedure does NOT touch the slit virtual motors (``Slit*Hsize``
etc.) -- it drives the underlying motor records directly, so the
sensitivity it measures is the raw motor-mm calibration, independent
of the slit calc.

After the procedure ends (success or abort), each blade is restored
to its baseline position. This is a measurement procedure; no
deliberate output is left in place.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from dataclasses import dataclass, field

import numpy as np
from epics import caget, caput

from ._shared.centroid import center_of_mass
from ._shared.cora_log import CoraProcedureLog
from ._shared.epics import (
    OperatorAbort,
    acquire_image,
    confirm_motion,
    move_motor,
    safe_restore,
)
from ._shared.log import setup_console_logger


# ---------------------------------------------------------------------------
# PV constants
# ---------------------------------------------------------------------------

# Same SLIT_STATIONS dict shape as centre_and_close_slits, but
# here we only need the blade motor PVs (the virtual-motor PVs
# aren't touched by this measurement).
SLIT_STATIONS = {
    "A": {
        "prefix": "2bma:Slit1",
        # Per item_020 (A-station L3 Slits):
        #   m13 = H+ (X+, outboard)
        #   m14 = H- (X-, inboard)
        #   m15 = V+ (Y+, up)
        #   m16 = V- (Y-, down)
        "blade_prefixes": ("2bma:m13", "2bma:m14", "2bma:m15", "2bma:m16"),
        # z position from source (item_020 layout map).
        "z_mm": 25225,
        # No upstream slit between source and A.
        "upstream_station": None,
    },
    "B": {
        "prefix": "2bma:Slit2",
        # Per item_020 (B-station L3-style Slits):
        #   m9  = V+ (Y+, up)
        #   m10 = V- (Y-, down)
        #   m11 = H pair
        #   m12 = H pair
        "blade_prefixes": ("2bma:m9", "2bma:m10", "2bma:m11", "2bma:m12"),
        "z_mm": 50500,
        # A's aperture, projected forward to B's plane, must contain
        # B's whole blade-throw range -- else A limits the spot first
        # and B's slope measurement is biased low.
        "upstream_station": "A",
    },
}

PV_LENS_SELECT = "2bm:MCTOptics:LensSelect"
PV_CAMERA_SELECT = "2bm:MCTOptics:CameraSelect"
CAMERA_PREFIXES_BY_INDEX = {0: "2bmSP1:", 1: "2bmSP2:"}
LENS_MAGNIFICATIONS_BY_INDEX = {0: 1.1, 1: 5.0, 2: 10.0}


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration + state
# ---------------------------------------------------------------------------

@dataclass
class Config:
    slit_station: str = "B"
    blade_throw_mm: float = 0.5        # ±this much from each blade's baseline

    # ROI half-side around the spot centroid, used by the bbox
    # measurement to exclude far-field multilayer stripes from the
    # edge detection. Must be large enough to contain the whole
    # spot at +blade_throw_mm (where one edge has moved further
    # OUT than baseline).
    edge_roi_half_size_pix: int = 400

    # Aperture-edge detection: half-max crossings of 1D row/column
    # profiles, with subpixel linear interpolation. Robust against
    # halo + multilayer-stripe noise that defeated the earlier
    # 2D bbox-of-above-threshold approach.
    aperture_edge_level: float = 0.5         # fraction of (p90 - p10)
    aperture_min_dynamic_range: float = 100  # counts; below this, no spot

    # Camera
    exposure_time: float = 0.2
    threshold_fraction: float = 0.5    # used only for the initial COM centring
    frames_per_measurement: int = 1
    camera_pixel_um: float = 3.45

    # Operator UX
    dry_run: bool = False
    auto_yes: bool = False
    confirm_restore: bool = False
    enable_cora_log: bool = True


@dataclass
class BladeCalibrationResult:
    blade_pv: str
    baseline_mm: float
    primary_edge: str               # "top", "bottom", "left", "right" or "?"
    pixel_displacement: float       # signed (primary edge's bbox shift, in pixels)
    motor_throw_mm: float           # signed full throw: (baseline-Δ) − (baseline+Δ)
    slope_pix_per_mm: float         # primary edge shift / motor throw
    all_edge_changes: dict          # dict {"top": Δpix, ...}


@dataclass
class _Snapshot:
    """Snapshot of camera state + the 4 blade baselines.

    On any exit path the blades are restored to baseline. This is a
    measurement procedure (no deliberate output), so restore is
    always full.
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
    blade_positions: dict = field(default_factory=dict)  # pv -> baseline value

    @classmethod
    def capture(cls, cam_prefix: str, blade_pvs: list[str]) -> "_Snapshot":
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
            blade_positions={
                pv: float(caget(f"{pv}.RBV")) for pv in blade_pvs
            },
        )

    def restore_plan(self) -> list[dict]:
        cp = self.cam_prefix
        plan: list[dict] = []
        # Blades first
        for pv, val in self.blade_positions.items():
            plan.append({"pv": pv, "current": "?",
                         "target": val, "units": "mm"})
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

    def restore(self) -> None:
        """Restore blades first, then camera."""
        cp = self.cam_prefix
        actions = []
        for pv, val in self.blade_positions.items():
            actions.append((
                f"{pv} -> {val:.4f} mm",
                lambda pv=pv, val=val: move_motor(pv, val, timeout=30),
            ))
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

class CalibrateSlitBladeThrow:
    """Stateful executor for ``calibrate_slit_blade_throw``."""

    def __init__(self, config: Config) -> None:
        self.config = config
        if config.slit_station not in SLIT_STATIONS:
            raise ValueError(
                f"unknown slit_station {config.slit_station!r}; "
                f"expected one of {list(SLIT_STATIONS)}"
            )
        s = SLIT_STATIONS[config.slit_station]
        self.slit_prefix = s["prefix"]
        self.blade_pvs: list[str] = list(s["blade_prefixes"])
        self.results: list[BladeCalibrationResult] = []
        self._snapshot: _Snapshot | None = None
        self._cam_prefix: str = ""
        self._magnification: float = 1.0
        self._pixel_um: float = config.camera_pixel_um
        self._frame_w: int = 0
        self._frame_h: int = 0
        # Spot centre (y, x) for the bbox ROI; updated from initial COM.
        self._roi_cy: float = 0.0
        self._roi_cx: float = 0.0

    # ---- detection -------------------------------------------------------

    def detect_camera_and_lens(self) -> None:
        cam_idx_raw = caget(PV_CAMERA_SELECT)
        lens_idx_raw = caget(PV_LENS_SELECT)
        if cam_idx_raw is None:
            raise RuntimeError(f"could not read {PV_CAMERA_SELECT}")
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
        self._pixel_um = self.config.camera_pixel_um * bin_x
        self._frame_w = int(caget(f"{self._cam_prefix}cam1:SizeX_RBV"))
        self._frame_h = int(caget(f"{self._cam_prefix}cam1:SizeY_RBV"))
        log.info("detected: camera=%s -> %s (%dx%d, pitch %.2f um), "
                 "lens=%s -> %.2fx; slit station %s, blades %s",
                 cam_label, self._cam_prefix, self._frame_w, self._frame_h,
                 self._pixel_um, lens_label, self._magnification,
                 self.config.slit_station, self.blade_pvs)

    # ---- image helpers ---------------------------------------------------

    def _acquire_frame(self) -> np.ndarray:
        c = self.config
        n = max(1, int(c.frames_per_measurement))
        if n == 1:
            return acquire_image(self._cam_prefix, exposure_time=c.exposure_time)
        f0 = acquire_image(self._cam_prefix, exposure_time=c.exposure_time)
        stack = np.empty((n,) + f0.shape, dtype=np.float32)
        stack[0] = f0
        for i in range(1, n):
            stack[i] = acquire_image(self._cam_prefix,
                                     exposure_time=c.exposure_time)
        return np.mean(stack, axis=0)

    def _measure_centroid(self) -> tuple[float, float] | None:
        """Initial COM to set the bbox-ROI centre. Same algorithm as
        the centre_and_close_slits procedure's COM path."""
        frame = self._acquire_frame()
        com = center_of_mass(frame, self.config.threshold_fraction)
        if com is None:
            return None
        log.info("baseline spot centroid: pix=(%.1f, %.1f) of %dx%d frame",
                 com[0], com[1], self._frame_w, self._frame_h)
        return com

    def _measure_bbox(self) -> dict | None:
        """Find the 4 edges of the slit-defined bright aperture using
        half-max crossings of 1D row/column-averaged profiles.

        For each axis: collapse the 2D ROI into a profile by averaging
        across the perpendicular axis; compute robust min/max via the
        10th/90th percentiles; threshold at ``p10 + level * (p90 - p10)``;
        find the leftmost and rightmost samples above threshold; refine
        each to subpixel position via linear interpolation between the
        bracketing samples.

        Returns a dict ``{'top', 'bottom', 'left', 'right'}`` in
        full-frame pixel coordinates, or ``None`` if either axis lacks
        enough dynamic range (slit closed past the beam, or detector
        signal too low to define an edge).
        """
        c = self.config
        frame = self._acquire_frame()
        h, w = frame.shape
        cy = int(self._roi_cy)
        cx = int(self._roi_cx)
        rh = c.edge_roi_half_size_pix
        y0 = max(0, cy - rh)
        y1 = min(h, cy + rh)
        x0 = max(0, cx - rh)
        x1 = min(w, cx + rh)
        roi = frame[y0:y1, x0:x1].astype(float)

        h_profile = roi.mean(axis=0)   # intensity vs X (length = ROI width)
        v_profile = roi.mean(axis=1)   # intensity vs Y (length = ROI height)

        h_edges = self._profile_half_max_crossings(
            h_profile, c.aperture_edge_level, c.aperture_min_dynamic_range)
        v_edges = self._profile_half_max_crossings(
            v_profile, c.aperture_edge_level, c.aperture_min_dynamic_range)
        if h_edges is None or v_edges is None:
            log.warning("edge detection failed: "
                        "h_profile range=%.0f (p10=%.0f, p90=%.0f); "
                        "v_profile range=%.0f (p10=%.0f, p90=%.0f); "
                        "min dynamic range = %.0f",
                        np.percentile(h_profile, 90) - np.percentile(h_profile, 10),
                        np.percentile(h_profile, 10),
                        np.percentile(h_profile, 90),
                        np.percentile(v_profile, 90) - np.percentile(v_profile, 10),
                        np.percentile(v_profile, 10),
                        np.percentile(v_profile, 90),
                        c.aperture_min_dynamic_range)
            return None
        left_sub, right_sub = h_edges
        top_sub, bottom_sub = v_edges
        bbox = {
            "left":   float(left_sub + x0),
            "right":  float(right_sub + x0),
            "top":    float(top_sub + y0),
            "bottom": float(bottom_sub + y0),
        }
        log.info("aperture edges @ %.0f%%-max: "
                 "top=%.1f bottom=%.1f left=%.1f right=%.1f "
                 "(height=%.1f, width=%.1f px); "
                 "h_profile p10/p90=%.0f/%.0f, v_profile p10/p90=%.0f/%.0f",
                 c.aperture_edge_level * 100,
                 bbox["top"], bbox["bottom"], bbox["left"], bbox["right"],
                 bbox["bottom"] - bbox["top"], bbox["right"] - bbox["left"],
                 np.percentile(h_profile, 10), np.percentile(h_profile, 90),
                 np.percentile(v_profile, 10), np.percentile(v_profile, 90))
        return bbox

    @staticmethod
    def _profile_half_max_crossings(
            profile: np.ndarray, level: float, min_dynamic_range: float):
        """Find left and right crossings of (p10 + level * (p90 - p10))
        in a 1D profile, with subpixel linear interpolation between
        bracketing samples.

        Returns (left_pos, right_pos) in profile-array indices (floats)
        or None if (p90 - p10) is below ``min_dynamic_range`` (the
        profile is too flat to define an edge -- e.g. slit closed past
        the beam, so the profile is uniform background).
        """
        p_lo = float(np.percentile(profile, 10))
        p_hi = float(np.percentile(profile, 90))
        if (p_hi - p_lo) < min_dynamic_range:
            return None
        thr = p_lo + level * (p_hi - p_lo)
        above = profile > thr
        if not above.any():
            return None
        left_idx = int(np.argmax(above))
        if 0 < left_idx < len(profile):
            v0, v1 = profile[left_idx - 1], profile[left_idx]
            if v1 > v0:
                left_pos = (left_idx - 1) + (thr - v0) / (v1 - v0)
            else:
                left_pos = float(left_idx)
        else:
            left_pos = float(left_idx)
        right_idx = len(profile) - 1 - int(np.argmax(above[::-1]))
        if 0 <= right_idx < len(profile) - 1:
            v0, v1 = profile[right_idx], profile[right_idx + 1]
            if v0 > v1:
                right_pos = right_idx + (v0 - thr) / (v0 - v1)
            else:
                right_pos = float(right_idx)
        else:
            right_pos = float(right_idx)
        return left_pos, right_pos

    # ---- gated motion helpers --------------------------------------------

    def _gated_blade_move(self, blade_pv: str, target_mm: float,
                          step_label: str) -> None:
        current = float(caget(f"{blade_pv}.RBV"))
        proceed = confirm_motion(
            [{"pv": f"{blade_pv}.VAL", "current": current,
              "target": target_mm, "units": "mm"}],
            step_label=step_label,
            dry_run=self.config.dry_run,
            auto_yes=self.config.auto_yes,
        )
        if proceed:
            move_motor(blade_pv, target_mm, timeout=30)

    # ---- per-blade calibration -------------------------------------------

    def calibrate_blade(self, blade_pv: str) -> BladeCalibrationResult:
        c = self.config
        baseline = float(caget(f"{blade_pv}.RBV"))
        log.info("calibrate blade %s (baseline = %.4f mm, throw = +/- "
                 "%.3f mm)", blade_pv, baseline, c.blade_throw_mm)

        # +throw
        self._gated_blade_move(
            blade_pv, baseline + c.blade_throw_mm,
            f"calibrate {blade_pv}: +{c.blade_throw_mm:.3f} mm")
        bbox_plus = self._measure_bbox()

        # -throw (from baseline; full motor throw = 2 * blade_throw_mm)
        self._gated_blade_move(
            blade_pv, baseline - c.blade_throw_mm,
            f"calibrate {blade_pv}: -{c.blade_throw_mm:.3f} mm "
            f"(full throw {2*c.blade_throw_mm:.3f} mm)")
        bbox_minus = self._measure_bbox()

        # Restore baseline
        self._gated_blade_move(
            blade_pv, baseline,
            f"calibrate {blade_pv}: restore baseline {baseline:.4f} mm")

        if bbox_plus is None or bbox_minus is None:
            log.warning("blade %s: one or both bbox measurements failed; "
                        "no slope computed", blade_pv)
            return BladeCalibrationResult(
                blade_pv=blade_pv, baseline_mm=baseline,
                primary_edge="?", pixel_displacement=0.0,
                motor_throw_mm=-2 * c.blade_throw_mm,
                slope_pix_per_mm=0.0,
                all_edge_changes={},
            )

        # Edge changes: bbox at (baseline - throw) MINUS bbox at
        # (baseline + throw). Motor went from baseline+throw to
        # baseline-throw, a motion of -2*throw mm.
        edge_changes = {
            edge: bbox_minus[edge] - bbox_plus[edge]
            for edge in ("top", "bottom", "left", "right")
        }
        # Primary edge = the one with the largest absolute change.
        primary_edge = max(edge_changes, key=lambda k: abs(edge_changes[k]))
        primary_displacement = edge_changes[primary_edge]
        motor_throw_full = -2 * c.blade_throw_mm
        slope = primary_displacement / motor_throw_full
        log.info("blade %s: primary edge=%s, displacement=%+.1f pix for "
                 "%+.3f mm motor throw -> slope = %+.2f pix/mm",
                 blade_pv, primary_edge, primary_displacement,
                 motor_throw_full, slope)
        log.info("  all edges (pix): top=%+.1f bottom=%+.1f left=%+.1f "
                 "right=%+.1f", edge_changes["top"], edge_changes["bottom"],
                 edge_changes["left"], edge_changes["right"])
        return BladeCalibrationResult(
            blade_pv=blade_pv, baseline_mm=baseline,
            primary_edge=primary_edge,
            pixel_displacement=primary_displacement,
            motor_throw_mm=motor_throw_full,
            slope_pix_per_mm=slope,
            all_edge_changes=edge_changes,
        )

    # ---- reporting -------------------------------------------------------

    def report_results(self) -> None:
        log.info("")
        log.info("=== Blade calibration summary (slit station %s) ===",
                 self.config.slit_station)
        log.info("%-14s %-9s %-12s %-12s %-10s",
                 "blade", "edge", "px_shift", "motor_mm", "px/mm")
        log.info("-" * 65)
        h_slopes = []
        v_slopes = []
        for r in self.results:
            log.info("%-14s %-9s %+10.1f  %+10.3f  %+10.2f",
                     r.blade_pv, r.primary_edge, r.pixel_displacement,
                     r.motor_throw_mm, r.slope_pix_per_mm)
            if r.primary_edge in ("left", "right"):
                h_slopes.append(abs(r.slope_pix_per_mm))
            elif r.primary_edge in ("top", "bottom"):
                v_slopes.append(abs(r.slope_pix_per_mm))
        log.info("")
        if len(h_slopes) >= 2:
            mean_h = sum(h_slopes) / len(h_slopes)
            log.info("H blades: |slopes|=%s  mean=%.2f  spread=%.1f%%",
                     [f"{s:.2f}" for s in h_slopes], mean_h,
                     100 * (max(h_slopes) - min(h_slopes)) / mean_h)
        if len(v_slopes) >= 2:
            mean_v = sum(v_slopes) / len(v_slopes)
            log.info("V blades: |slopes|=%s  mean=%.2f  spread=%.1f%%",
                     [f"{s:.2f}" for s in v_slopes], mean_v,
                     100 * (max(v_slopes) - min(v_slopes)) / mean_v)
        if h_slopes and v_slopes:
            mean_h = sum(h_slopes) / len(h_slopes)
            mean_v = sum(v_slopes) / len(v_slopes)
            ratio = mean_v / mean_h
            log.info("V/H mean ratio = %.3f (1.0 means H and V have "
                     "the same blade calibration)", ratio)
            if not (0.9 <= ratio <= 1.1):
                log.warning("V/H ratio = %.3f is OUTSIDE [0.9, 1.1]: one "
                            "axis is mis-calibrated by ~%.0f%% relative "
                            "to the other.", ratio, abs(ratio - 1) * 100)
        log.info("")
        log.info("Interpretation:")
        log.info("- Same-axis blades should agree on |slope|: any spread")
        log.info("  >5%% points at one specific blade with wrong mm/encoder.")
        log.info("- V/H ratio should be ~1.0: if not, the whole-axis pair")
        log.info("  (both blades) has a different scaling than the other.")
        log.info("- Combine the two: a single outlier blade vs. an entire-")
        log.info("  axis miscal narrows down where to investigate the IOC")
        log.info("  motor record (.MRES, .ERES, gear ratio, encoder dir).")

    # ---- orchestrator -----------------------------------------------------

    def _check_upstream_aperture(self) -> None:
        """Warn if an upstream slit's aperture, projected forward to this
        station's plane, is too small to contain the spot through the
        full blade-throw range. The geometric projection assumes a
        point source at z=0; penumbra from finite source size only
        widens the beam further, so this check is conservative.

        Purely additive: emits ``log.warning`` only -- no caputs, no
        gates, no procedure-flow change. Operator can ignore and the
        run proceeds exactly as without the check; only the measured
        slope on a clipped blade would be biased low.
        """
        c = self.config
        station = SLIT_STATIONS[c.slit_station]
        upstream_key = station.get("upstream_station")
        if upstream_key is None:
            return
        upstream = SLIT_STATIONS[upstream_key]
        z_target = float(station["z_mm"])
        z_upstream = float(upstream["z_mm"])
        mag = z_target / z_upstream  # >1: downstream projection enlarges
        target_prefix = station["prefix"]
        upstream_prefix = upstream["prefix"]
        try:
            target_h = float(caget(f"{target_prefix}Hsize"))
            target_v = float(caget(f"{target_prefix}Vsize"))
            upstream_h = float(caget(f"{upstream_prefix}Hsize"))
            upstream_v = float(caget(f"{upstream_prefix}Vsize"))
        except Exception as exc:
            log.debug("upstream-aperture check skipped (caget failed: %s)", exc)
            return
        # Max half-extent any single blade reaches during the procedure:
        # current half-aperture + blade_throw_mm (one blade moves while
        # the other stays put, so the extreme is on the moving side).
        target_extreme_h = target_h / 2 + c.blade_throw_mm
        target_extreme_v = target_v / 2 + c.blade_throw_mm
        # Upstream half-aperture must cover that extreme when projected
        # back: required_upstream_half = target_extreme / mag.
        required_upstream_h = 2 * target_extreme_h / mag
        required_upstream_v = 2 * target_extreme_v / mag
        # 1.5x margin in the recommended fix value.
        for axis, current, required in [
                ("H", upstream_h, required_upstream_h),
                ("V", upstream_v, required_upstream_v)]:
            if current < required:
                log.warning(
                    "upstream %s station %ssize = %.3f mm is SMALLER than "
                    "the safe minimum %.3f mm (target station %s's "
                    "%s-blade reaches half-extent %.3f mm at +throw; "
                    "projection mag from z=%.0f to z=%.0f is %.2fx). "
                    "Upstream slit will clip the spot before %s blades do, "
                    "biasing the slope measurement LOW. To fix:  "
                    "caput %s%ssize %.2f",
                    upstream_key, upstream_prefix, current, required,
                    c.slit_station, axis,
                    target_extreme_h if axis == "H" else target_extreme_v,
                    z_upstream, z_target, mag, c.slit_station,
                    upstream_prefix, axis, max(required * 1.5, 5.0))

    def run(self) -> bool:
        c = self.config
        self.detect_camera_and_lens()
        self._snapshot = _Snapshot.capture(self._cam_prefix, self.blade_pvs)
        log.info("snapshotted blade baselines (mm):")
        for pv, val in self._snapshot.blade_positions.items():
            log.info("  %s = %+.4f", pv, val)
        self._check_upstream_aperture()

        cora = (CoraProcedureLog(
            slug="calibrate_slit_blade_throw",
            target_asset_ids=[
                f"{c.slit_station}_station_slits",
            ],
            parameters=vars(c),
        ) if c.enable_cora_log else None)
        if cora:
            cora.open()

        all_ok = False
        try:
            # Baseline COM to set the ROI centre for bbox measurements
            com = self._measure_centroid()
            if com is None:
                raise RuntimeError(
                    "could not measure baseline spot centroid; "
                    "the centroid algorithm returned no signal above "
                    f"threshold_fraction={c.threshold_fraction}. "
                    "Open the slits enough to admit a visible beam "
                    "before running."
                )
            # COM returns (x, y); we use (y, x) for the ROI
            self._roi_cx = float(com[0])
            self._roi_cy = float(com[1])

            for blade in self.blade_pvs:
                result = self.calibrate_blade(blade)
                self.results.append(result)
                if cora:
                    cora.append_step("calibrate_blade", {
                        "blade_pv": result.blade_pv,
                        "primary_edge": result.primary_edge,
                        "pixel_displacement": result.pixel_displacement,
                        "motor_throw_mm": result.motor_throw_mm,
                        "slope_pix_per_mm": result.slope_pix_per_mm,
                    })

            self.report_results()
            all_ok = True
            return True
        except OperatorAbort as exc:
            log.warning("operator aborted: %s", exc)
            if cora:
                cora.append_step("abort", {"reason": str(exc)})
            return False
        finally:
            # Always restore: this is a measurement procedure; nothing
            # the operator wants to keep.
            if self._snapshot is not None:
                log.info("restoring blade baselines + camera state")
                confirm_motion(
                    self._snapshot.restore_plan(),
                    step_label="restore: returning blades + camera to "
                               "pre-procedure state",
                    dry_run=False,
                    auto_yes=c.auto_yes,
                    announce_only=(not c.confirm_restore),
                )
                self._snapshot.restore()
            if cora:
                outcome = "complete" if all_ok else "truncate"
                cora.close(outcome=outcome)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Calibrate each L3-style slit blade motor by driving "
                    "it +/- a known throw and measuring how far the "
                    "corresponding spot edge moves on the detector. Reports "
                    "pixels-per-mm for each of the 4 blades and flags any "
                    "outliers (mis-calibrated motors).")
    p.add_argument("--slit-station", choices=list(SLIT_STATIONS), default="B",
                   help="Which station's blades to test. Default: B.")
    p.add_argument("--blade-throw-mm", type=float, default=0.5,
                   help="Move each blade by +/- this much from baseline. "
                        "Full motor throw is 2x this. Default: 0.5 mm.")
    p.add_argument("--edge-roi-half-size-pix", type=int, default=400,
                   help="Half-side of the ROI around the spot centre used "
                        "for bbox edge detection. Must be large enough to "
                        "contain the spot at +blade-throw-mm (one edge "
                        "moves further out than baseline). Default: 400 pix.")
    p.add_argument("--aperture-edge-level", type=float, default=0.5,
                   help="Fraction of (p90 - p10) of the 1D profile used as "
                        "the threshold for edge crossings. 0.5 = half-max, "
                        "which matches the geometric slit-blade position "
                        "for a sharp edge with penumbra. Default: 0.5.")
    p.add_argument("--aperture-min-dynamic-range", type=float, default=100.0,
                   help="If (p90 - p10) of either the H or V profile is "
                        "below this many counts the procedure treats the "
                        "spot as not visible. Default: 100.")
    p.add_argument("--exposure-time", type=float, default=0.2,
                   help="Camera exposure (s). Default: 0.2.")
    p.add_argument("--threshold-fraction", type=float, default=0.5,
                   help="COM threshold (fraction of frame max) used only "
                        "for the initial centring of the bbox ROI. "
                        "Default: 0.5.")
    p.add_argument("--frames-per-measurement", type=int, default=1,
                   help="Average N frames per bbox measurement. Default: 1.")
    p.add_argument("--camera-pixel-um", type=float, default=3.45,
                   help="Camera sensor pitch, pre-binning. Default: 3.45.")
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
        blade_throw_mm=args.blade_throw_mm,
        edge_roi_half_size_pix=args.edge_roi_half_size_pix,
        aperture_edge_level=args.aperture_edge_level,
        aperture_min_dynamic_range=args.aperture_min_dynamic_range,
        exposure_time=args.exposure_time,
        threshold_fraction=args.threshold_fraction,
        frames_per_measurement=args.frames_per_measurement,
        camera_pixel_um=args.camera_pixel_um,
        dry_run=args.dry_run,
        auto_yes=args.yes,
        confirm_restore=args.confirm_restore,
        enable_cora_log=(not args.no_cora_log),
    )
    proc = CalibrateSlitBladeThrow(config)
    try:
        ok = proc.run()
    except Exception as exc:
        log.error("procedure failed: %s", exc, exc_info=True)
        return 2
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
