"""Detector Z-rail alignment to the beam (cora slug: ``detector_z_rail_alignment``).

Walks the Optique Peter detector along its 1 m PRO225SL Z stage
with a small square X-ray aperture defined by the B-station slits,
fits the centroid drift across Z, and uses the detector optical
table (``2bmb:table3.AX`` / ``.AY``) to rotate the rail back parallel
to the beam.

Reference: ``2bm-docs/source/procedures/item_002.rst`` for the
operator-facing spec; this module is the executable body.

Operating envelope (v0.0.1, "build trust" phase):

* Z stage moves are clamped to the band ``[200, 500]`` mm by a
  software guard. The motor's own ``.HLM`` / ``.LLM`` are not
  modified.
* Operator must have set the camera (Camera 1 / Camera 2), the lens
  slot (Lens1 / Lens2 / Lens3), and the B-station slits to a small
  square aperture before running. The procedure does not force any
  of these — it reads them and adapts.
* Operator must have opened the front-end shutter before running.
  The procedure does not toggle the shutter.
* Before every motor motion the procedure prints a plan block (PV,
  current, target, delta, units) and waits for ``y`` / ``N``.
  ``--yes`` bypasses the prompt; ``--dry-run`` prints and skips.
* On any exit path (success, abort, exception, Ctrl-C) the procedure
  restores the operator's pre-procedure camera state and returns the
  Z stage to its captured baseline. The new ``table3.AY`` / ``.AX``
  values are the procedure's deliberate output and are NOT restored.

Invoke as a module:

    python -m procedures.detector_z_rail_alignment \\
        --z-near 200 --z-far 500 --exposure-time 0.05

Run ``-h`` for the full parameter list.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from dataclasses import dataclass, field

from epics import caget, caput

from ._shared.centroid import center_of_mass, pixels_to_object_um
from ._shared.cora_log import CoraProcedureLog
from ._shared.epics import (
    OperatorAbort,
    acquire_image,
    confirm_motion,
    move_motor,
    move_table_axis,
    safe_restore,
)


# ---------------------------------------------------------------------------
# PV constants
# ---------------------------------------------------------------------------

# Optique Peter Z stage (Aerotech PRO225SL-1000 on a dedicated IOC)
PV_OP_Z_MOTOR = "2bmbAERO:m1"

# Detector optical table soft PVs (synApps table.db, GEOM=SRI; see
# 2bm-docs/manual/item_020.rst "Detector optical table" section).
# Literal '.' in the PV name — these are NOT motor record fields.
PV_TABLE_AY = "2bmb:table3.AY"
PV_TABLE_AX = "2bmb:table3.AX"

# Six underlying jacks aggregated by 2bmb:table3 (per item_020.rst).
TABLE_JACK_PREFIXES = (
    "2bmb:m9",   # M2Y
    "2bmb:m10",  # M2X
    "2bmb:m11",  # M2Z
    "2bmb:m12",  # M1Y
    "2bmb:m13",  # M0X
    "2bmb:m14",  # M0Y
)

# MCTOptics IOC (selectors are operator-set; procedure only reads)
PV_LENS_SELECTED = "2bm:MCTOptics:LensSelected"
PV_CAMERA_SELECTED = "2bm:MCTOptics:CameraSelected"

# Camera prefix derived from MCTOptics CameraSelected enum string
CAMERA_PREFIXES = {
    "Camera 1": "2bmSP1:",   # FLIR Oryx 5MP
    "Camera 2": "2bmSP2:",   # FLIR Oryx 31MP
}

# Lens magnification keyed by MCTOptics LensSelected enum string.
# Update these when the installed objectives change.
LENS_MAGNIFICATIONS = {
    "Lens1": 1.1,
    "Lens2": 5.0,
    "Lens3": 10.0,
}

# Z-stage software guard (procedure-level; does not modify motor limits)
Z_GUARD_MIN_MM = 200.0
Z_GUARD_MAX_MM = 500.0


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration + state
# ---------------------------------------------------------------------------

@dataclass
class Config:
    z_near: float = 200.0
    z_far: float = 500.0
    z_calibration_step_urad: float = 50.0
    exposure_time: float = 0.05
    convergence_threshold_urad: float = 5.0
    max_iterations: int = 5
    threshold_fraction: float = 0.5
    camera_pixel_um: float = 3.45
    min_jacobian_um_per_urad: float = 0.001
    dry_run: bool = False
    auto_yes: bool = False
    confirm_restore: bool = False
    enable_cora_log: bool = True


@dataclass
class Jacobian:
    """Table -> centroid sensitivity at z_far (units: um centroid / urad table)."""
    J_AY_X: float = 0.0
    J_AY_Y: float = 0.0
    J_AX_X: float = 0.0
    J_AX_Y: float = 0.0


@dataclass
class Baseline:
    table_AY: float = 0.0
    table_AX: float = 0.0


@dataclass
class IterationResult:
    iteration: int
    X_near_um: float
    Y_near_um: float
    X_far_um: float
    Y_far_um: float
    slope_X_um_per_mm: float
    slope_Y_um_per_mm: float
    tilt_X_urad: float
    tilt_Y_urad: float
    correction_AY_urad: float = 0.0
    correction_AX_urad: float = 0.0
    converged: bool = False


@dataclass
class _Snapshot:
    """Pre-procedure state for restore-on-exit.

    Deliberately omits ``table3.AY`` / ``.AX`` — those are the
    procedure's deliberate output and must NOT be restored.

    Deliberately omits MCTOptics selections and the FES shutter — the
    operator is responsible for those (v0.0.1).
    """
    cam_prefix: str = ""
    cam_was_acquiring: bool = False
    cam_acquire_time: float = 0.0
    cam_num_images: int = 1
    cam_image_mode: str = "Single"
    cam_trigger_mode: str = "Off"
    cam_trigger_source: str = "Software"
    cam_trigger_overlap: str = "Off"
    cam_exposure_mode: str = "Timed"
    cam_array_callbacks: str = "Disable"
    z_position: float = 0.0

    @classmethod
    def capture(cls, cam_prefix: str) -> "_Snapshot":
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
            z_position=float(caget(f"{PV_OP_Z_MOTOR}.RBV")),
        )

    def restore_plan(self) -> list[dict]:
        """Plan-block representation of what restore() will do."""
        cp = self.cam_prefix
        plan: list[dict] = [
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
            {"pv": PV_OP_Z_MOTOR, "current": "?",
             "target": self.z_position, "units": "mm"},
        ]
        if self.cam_was_acquiring:
            plan.append({"pv": f"{cp}cam1:Acquire", "current": 0,
                         "target": 1, "units": "(resume)"})
        return plan

    def restore(self) -> None:
        cp = self.cam_prefix
        actions = [
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
            ("Z stage to baseline",
             lambda: move_motor(PV_OP_Z_MOTOR, self.z_position, timeout=180)),
        ]
        if self.cam_was_acquiring:
            actions.append(("resume continuous acquire",
                            lambda: caput(f"{cp}cam1:Acquire", 1)))
        safe_restore(actions)


# ---------------------------------------------------------------------------
# The procedure
# ---------------------------------------------------------------------------

class DetectorZRailAlignment:
    """Stateful executor for ``detector_z_rail_alignment``."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.baseline = Baseline()
        self.jacobian = Jacobian()
        self.history: list[IterationResult] = []
        self._snapshot: _Snapshot | None = None
        self._cam_prefix: str = ""
        self._magnification: float = 1.0

        # Z-range guard (procedure-level safety band).
        if not (Z_GUARD_MIN_MM <= config.z_near < config.z_far <= Z_GUARD_MAX_MM):
            raise ValueError(
                f"Z range [{config.z_near}, {config.z_far}] mm violates "
                f"safety band [{Z_GUARD_MIN_MM}, {Z_GUARD_MAX_MM}] mm "
                f"(z_near must also be < z_far)"
            )

    # ---- detection of operator-set camera / lens --------------------------

    def detect_camera_and_lens(self) -> None:
        """Read MCTOptics CameraSelected / LensSelected and derive
        ``cam_prefix`` + ``magnification``. Operator-set; not modified."""
        camera = caget(PV_CAMERA_SELECTED, as_string=True)
        lens = caget(PV_LENS_SELECTED, as_string=True)
        if camera is None:
            raise RuntimeError(
                f"could not read {PV_CAMERA_SELECTED} — is MCTOptics IOC "
                "reachable from this host?"
            )
        if camera not in CAMERA_PREFIXES:
            raise RuntimeError(
                f"unknown camera selection {camera!r}; "
                f"expected one of {list(CAMERA_PREFIXES)}"
            )
        self._cam_prefix = CAMERA_PREFIXES[camera]

        if lens is None:
            raise RuntimeError(f"could not read {PV_LENS_SELECTED}")
        if lens not in LENS_MAGNIFICATIONS:
            raise RuntimeError(
                f"unknown lens selection {lens!r}; "
                f"expected one of {list(LENS_MAGNIFICATIONS)}"
            )
        self._magnification = LENS_MAGNIFICATIONS[lens]

        log.info("detected: camera=%s (%s), lens=%s (%.2fx magnification)",
                 camera, self._cam_prefix, lens, self._magnification)

    # ---- gated motion helpers --------------------------------------------

    def _gate(self, plan: list[dict], step_label: str) -> bool:
        return confirm_motion(
            plan,
            step_label=step_label,
            dry_run=self.config.dry_run,
            auto_yes=self.config.auto_yes,
        )

    def _gated_move_z(self, target: float, step_label: str) -> None:
        if not (Z_GUARD_MIN_MM <= target <= Z_GUARD_MAX_MM):
            raise ValueError(
                f"Z target {target} mm outside safety band "
                f"[{Z_GUARD_MIN_MM}, {Z_GUARD_MAX_MM}]"
            )
        current = float(caget(f"{PV_OP_Z_MOTOR}.RBV"))
        proceed = self._gate(
            [{"pv": PV_OP_Z_MOTOR, "current": current,
              "target": target, "units": "mm"}],
            step_label=step_label,
        )
        if proceed:
            move_motor(PV_OP_Z_MOTOR, target, timeout=180)

    def _gated_move_table(self, axis_pv: str, target: float,
                          step_label: str) -> None:
        current = float(caget(axis_pv))
        proceed = self._gate(
            [{"pv": axis_pv, "current": current,
              "target": target, "units": "deg"}],
            step_label=step_label,
        )
        if proceed:
            move_table_axis(axis_pv, target, TABLE_JACK_PREFIXES, timeout=60)

    def _gated_move_table_pair(self, ay_target: float, ax_target: float,
                               step_label: str) -> None:
        ay_now = float(caget(PV_TABLE_AY))
        ax_now = float(caget(PV_TABLE_AX))
        proceed = self._gate(
            [
                {"pv": PV_TABLE_AY, "current": ay_now,
                 "target": ay_target, "units": "deg"},
                {"pv": PV_TABLE_AX, "current": ax_now,
                 "target": ax_target, "units": "deg"},
            ],
            step_label=step_label,
        )
        if proceed:
            move_table_axis(PV_TABLE_AY, ay_target, TABLE_JACK_PREFIXES,
                            timeout=60)
            move_table_axis(PV_TABLE_AX, ax_target, TABLE_JACK_PREFIXES,
                            timeout=60)

    def _measure_centroid(self) -> tuple[float, float]:
        if self.config.dry_run:
            log.info("[dry-run] would acquire image and fit centroid")
            return (0.0, 0.0)
        frame = acquire_image(self._cam_prefix,
                              exposure_time=self.config.exposure_time)
        com = center_of_mass(frame, self.config.threshold_fraction)
        if com is None:
            raise RuntimeError(
                "centroid fit failed: no signal above threshold "
                "(slits too closed, beam off, shutter shut?)"
            )
        return pixels_to_object_um(
            com,
            camera_pixel_um=self.config.camera_pixel_um,
            magnification=self._magnification,
        )

    # ---- procedure phases ------------------------------------------------

    def record_baseline(self) -> None:
        self.baseline = Baseline(
            table_AY=float(caget(PV_TABLE_AY)),
            table_AX=float(caget(PV_TABLE_AX)),
        )
        log.info("baseline table AY=%.6g, AX=%.6g (deg)",
                 self.baseline.table_AY, self.baseline.table_AX)

    def calibrate_jacobian(self) -> None:
        c = self.config
        delta = c.z_calibration_step_urad * 1e-3 / 17.4533  # urad -> deg
        # 1 urad = 1e-6 rad = 5.7296e-5 deg; using shorter constant for clarity:
        delta = c.z_calibration_step_urad * 5.72958e-5

        log.info("calibrate Jacobian at z_far=%.3f mm with delta=%.1f urad "
                 "(%.3e deg)", c.z_far, c.z_calibration_step_urad, delta)

        self._gated_move_z(c.z_far, "step: move Z to far for Jacobian calibration")
        x_f0, y_f0 = self._measure_centroid()

        # AY perturbation
        self._gated_move_table(
            PV_TABLE_AY, self.baseline.table_AY + delta,
            f"calibration: perturb AY by +{c.z_calibration_step_urad:.1f} urad",
        )
        x_f1, y_f1 = self._measure_centroid()
        self.jacobian.J_AY_X = (x_f1 - x_f0) / c.z_calibration_step_urad
        self.jacobian.J_AY_Y = (y_f1 - y_f0) / c.z_calibration_step_urad
        log.info("  J_AY_X=%+.4f, J_AY_Y=%+.4f um/urad",
                 self.jacobian.J_AY_X, self.jacobian.J_AY_Y)
        self._gated_move_table(
            PV_TABLE_AY, self.baseline.table_AY,
            "calibration: restore AY",
        )

        # AX perturbation
        self._gated_move_table(
            PV_TABLE_AX, self.baseline.table_AX + delta,
            f"calibration: perturb AX by +{c.z_calibration_step_urad:.1f} urad",
        )
        x_f2, y_f2 = self._measure_centroid()
        self.jacobian.J_AX_X = (x_f2 - x_f0) / c.z_calibration_step_urad
        self.jacobian.J_AX_Y = (y_f2 - y_f0) / c.z_calibration_step_urad
        log.info("  J_AX_X=%+.4f, J_AX_Y=%+.4f um/urad",
                 self.jacobian.J_AX_X, self.jacobian.J_AX_Y)
        self._gated_move_table(
            PV_TABLE_AX, self.baseline.table_AX,
            "calibration: restore AX",
        )

        if (abs(self.jacobian.J_AY_X) < c.min_jacobian_um_per_urad
                or abs(self.jacobian.J_AX_Y) < c.min_jacobian_um_per_urad):
            raise RuntimeError(
                f"Jacobian below sanity floor "
                f"(|J_AY_X|={abs(self.jacobian.J_AY_X):.5f}, "
                f"|J_AX_Y|={abs(self.jacobian.J_AX_Y):.5f}); "
                "test step too small, slits closed, or table not moving?"
            )

    def _measure_slope(self) -> tuple[float, float, float, float, float, float]:
        self._gated_move_z(self.config.z_near,
                           "iteration: move Z to near for slope measurement")
        x_n, y_n = self._measure_centroid()
        self._gated_move_z(self.config.z_far,
                           "iteration: move Z to far for slope measurement")
        x_f, y_f = self._measure_centroid()
        dz = self.config.z_far - self.config.z_near
        return x_n, y_n, x_f, y_f, (x_f - x_n) / dz, (y_f - y_n) / dz

    def iterate(self) -> bool:
        c = self.config
        log.info("iterate (max_iterations=%d, threshold=%.1f urad)",
                 c.max_iterations, c.convergence_threshold_urad)

        sign_AY = math.copysign(1.0, self.jacobian.J_AY_X)
        sign_AX = math.copysign(1.0, self.jacobian.J_AX_Y)
        deg_per_urad = 5.72958e-5

        for i in range(1, c.max_iterations + 1):
            x_n, y_n, x_f, y_f, slope_X, slope_Y = self._measure_slope()
            tilt_X_urad = slope_X * 1000.0
            tilt_Y_urad = slope_Y * 1000.0

            converged = (abs(tilt_X_urad) <= c.convergence_threshold_urad
                         and abs(tilt_Y_urad) <= c.convergence_threshold_urad)

            result = IterationResult(
                iteration=i,
                X_near_um=x_n, Y_near_um=y_n,
                X_far_um=x_f, Y_far_um=y_f,
                slope_X_um_per_mm=slope_X,
                slope_Y_um_per_mm=slope_Y,
                tilt_X_urad=tilt_X_urad,
                tilt_Y_urad=tilt_Y_urad,
                converged=converged,
            )

            if converged:
                self.history.append(result)
                log.info("  iter %d: CONVERGED  tilt_X=%.2f urad, tilt_Y=%.2f urad",
                         i, tilt_X_urad, tilt_Y_urad)
                return True

            d_AY = -sign_AY * tilt_X_urad
            d_AX = -sign_AX * tilt_Y_urad
            result.correction_AY_urad = d_AY
            result.correction_AX_urad = d_AX
            self.history.append(result)
            log.info("  iter %d: tilt_X=%+.2f urad -> dAY=%+.2f urad; "
                     "tilt_Y=%+.2f urad -> dAX=%+.2f urad",
                     i, tilt_X_urad, d_AY, tilt_Y_urad, d_AX)

            new_AY = float(caget(PV_TABLE_AY)) + d_AY * deg_per_urad
            new_AX = float(caget(PV_TABLE_AX)) + d_AX * deg_per_urad
            self._gated_move_table_pair(
                new_AY, new_AX,
                f"iteration {i}: apply corrective table tilt",
            )

        log.warning("did not converge after %d iterations", c.max_iterations)
        return False

    # ---- orchestrator ----------------------------------------------------

    def run(self) -> bool:
        c = self.config

        self.detect_camera_and_lens()
        self._snapshot = _Snapshot.capture(self._cam_prefix)
        log.info("snapshotted pre-procedure state (camera + Z); "
                 "will restore on exit")
        log.info("  cam was_acquiring=%s, ImageMode=%s, TriggerMode=%s, "
                 "TriggerSource=%s",
                 self._snapshot.cam_was_acquiring,
                 self._snapshot.cam_image_mode,
                 self._snapshot.cam_trigger_mode,
                 self._snapshot.cam_trigger_source)
        log.info("  Z position = %.3f mm", self._snapshot.z_position)

        cora = (CoraProcedureLog(
            slug="detector_z_rail_alignment",
            target_asset_ids=[
                "Optique_Peter_focus_Z",
                "Detector_optical_table",   # pending cora registration
                "Scintillator_LuAG",
            ],
            parameters=vars(c),
        ) if c.enable_cora_log else None)
        if cora:
            cora.open()

        converged = False
        try:
            self.record_baseline()
            if cora:
                cora.append_step("baseline", vars(self.baseline))
            self.calibrate_jacobian()
            if cora:
                cora.append_step("calibrate", vars(self.jacobian))
            converged = self.iterate()
            if cora:
                cora.append_step("iterate",
                                 {"iterations": len(self.history),
                                  "converged": converged})
            return converged
        except OperatorAbort as exc:
            log.warning("operator aborted procedure: %s", exc)
            if cora:
                cora.append_step("abort", {"reason": str(exc)})
            return False
        finally:
            if self._snapshot is not None:
                log.info("restoring pre-procedure state")
                confirm_motion(
                    self._snapshot.restore_plan(),
                    step_label="restore: returning camera + Z to "
                               "pre-procedure state",
                    dry_run=c.dry_run,
                    auto_yes=c.auto_yes,
                    announce_only=(not c.confirm_restore),
                )
                if not c.dry_run:
                    self._snapshot.restore()
            if cora:
                outcome = "complete" if (self.history
                                         and self.history[-1].converged) \
                                     else "truncate"
                cora.close(outcome=outcome)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Align the Optique Peter detector Z rail to the beam "
                    "using the detector optical table beneath the rail. "
                    "Camera and lens are auto-detected from MCTOptics; "
                    "operator must open the shutter and set slits.")
    p.add_argument("--z-near", type=float, default=200.0,
                   help=f"Upstream Z anchor (mm). "
                        f"Must be in [{Z_GUARD_MIN_MM}, {Z_GUARD_MAX_MM}]. "
                        f"Default: 200.")
    p.add_argument("--z-far", type=float, default=500.0,
                   help=f"Downstream Z anchor (mm). "
                        f"Must be in [{Z_GUARD_MIN_MM}, {Z_GUARD_MAX_MM}]. "
                        f"Default: 500.")
    p.add_argument("--calibration-step-urad", type=float, default=50.0,
                   help="Test step applied to table AY/AX for Jacobian "
                        "discovery. Default: 50 urad.")
    p.add_argument("--exposure-time", type=float, default=0.05,
                   help="Camera exposure (s). Default: 0.05.")
    p.add_argument("--convergence-urad", type=float, default=5.0,
                   help="Stop iterating when |tilt_X|, |tilt_Y| are below "
                        "this. Default: 5 urad.")
    p.add_argument("--max-iterations", type=int, default=5)
    p.add_argument("--threshold-fraction", type=float, default=0.5,
                   help="Centroid threshold as fraction of frame max. "
                        "Default: 0.5.")
    p.add_argument("--camera-pixel-um", type=float, default=3.45,
                   help="Camera sensor pixel size (um). Default: 3.45 "
                        "(Oryx 5MP / 31MP).")
    p.add_argument("--yes", action="store_true",
                   help="Auto-confirm every motion prompt (skip y/N gate). "
                        "Use for headless or scripted runs.")
    p.add_argument("--confirm-restore", action="store_true",
                   help="Also gate the restore path on y/N. "
                        "Default: restore is announced but not gated.")
    p.add_argument("--no-cora-log", action="store_true",
                   help="Skip cora Procedure-record logging.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print every planned motion and skip; never moves "
                        "any motor.")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-7s  %(name)s: %(message)s",
    )
    config = Config(
        z_near=args.z_near,
        z_far=args.z_far,
        z_calibration_step_urad=args.calibration_step_urad,
        exposure_time=args.exposure_time,
        convergence_threshold_urad=args.convergence_urad,
        max_iterations=args.max_iterations,
        threshold_fraction=args.threshold_fraction,
        camera_pixel_um=args.camera_pixel_um,
        dry_run=args.dry_run,
        auto_yes=args.yes,
        confirm_restore=args.confirm_restore,
        enable_cora_log=(not args.no_cora_log),
    )
    proc = DetectorZRailAlignment(config)
    try:
        converged = proc.run()
    except Exception as exc:
        log.error("procedure failed: %s", exc, exc_info=True)
        return 2
    return 0 if converged else 1


if __name__ == "__main__":
    sys.exit(main())
