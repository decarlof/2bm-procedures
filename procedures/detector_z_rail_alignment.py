"""Detector Z-rail alignment to the beam (cora slug: ``detector_z_rail_alignment``).

Walks the Optique Peter detector along its 1 m PRO225SL Z stage
with a small square X-ray aperture defined by the B-station slits,
fits the centroid drift across Z, and uses the detector optical
table (``2bmb:table3.AX`` / ``.AY``) to rotate the rail back parallel
to the beam.

Reference: ``2bm-docs/source/procedures/item_002.rst`` for the
operator-facing spec; this module is the executable body.

Algorithm in two phases:

1. **Calibration (iteration 0).** At ``z_far``, apply a known
   ``Δ`` to ``table3.AY`` and measure the centroid shift to record
   the *sign and magnitude* of the table → centroid Jacobian
   (``J_AY_X``, ``J_AY_Y``). Repeat with ``ΔAX`` for the second
   column (``J_AX_X``, ``J_AX_Y``). The Jacobian is auto-discovered
   on every run so it survives table-record reloads, encoder
   re-zeros, and cable swaps.

2. **Iterative correction.** Measure centroid at ``z_near`` and
   ``z_far``; compute the per-mm slope; convert to a corrective AY
   / AX step using the geometric formula
   (``ΔAY_urad = −1000 · slope_X_µm_per_mm``) with the sign drawn
   from the calibrated Jacobian; apply; repeat until both slopes
   are below ``convergence_threshold`` or ``max_iterations`` is
   reached.

Invoke as a module:

    python -m procedures.detector_z_rail_alignment \\
        --z-near 50 --z-far 350 --exposure-time 0.05

Run ``-h`` for the full parameter list.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from dataclasses import dataclass, field

from ._shared.centroid import center_of_mass, pixels_to_object_um
from ._shared.cora_log import CoraProcedureLog
from ._shared.epics import (
    acquire_image,
    caput_wait,
    cawait_value,
    move_motor,
)
from ._shared.slits import set_horizontal_aperture, set_vertical_aperture
from epics import caget, caput


# ---------------------------------------------------------------------------
# PV constants
# ---------------------------------------------------------------------------

# Optique Peter Z stage (Aerotech PRO225SL-1000 on a dedicated IOC)
PV_OP_Z_MOTOR = "2bmbAERO:m1"

# Detector optical table virtual record (calc-driven composites)
PV_TABLE_AY = "2bmb:table3.AY"
PV_TABLE_AY_RBV = "2bmb:table3.AY.RBV"
PV_TABLE_AY_DMOV = "2bmb:table3.AY.DMOV"
PV_TABLE_AX = "2bmb:table3.AX"
PV_TABLE_AX_RBV = "2bmb:table3.AX.RBV"
PV_TABLE_AX_DMOV = "2bmb:table3.AX.DMOV"

# MCTOptics IOC
PV_LENS_SELECT = "2bm:MCTOptics:LensSelect"
PV_LENS_SELECTED = "2bm:MCTOptics:LensSelected"
PV_CAMERA_SELECT = "2bm:MCTOptics:CameraSelect"
PV_CAMERA_SELECTED = "2bm:MCTOptics:CameraSelected"

# Camera areaDetector prefix (camera 0 = Oryx 5MP)
CAM0_PREFIX = "2bmSP1:"
CAM1_PREFIX = "2bmSP2:"

# Front-end shutter
PV_FES_OPEN = "S02BM-PSS:FES:OpenEPICSC"
PV_FES_CLOSE = "S02BM-PSS:FES:CloseEPICSC"
PV_FES_POSITION = "S02BM-PSS:FES:Position"


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration + state
# ---------------------------------------------------------------------------

@dataclass
class Config:
    z_near: float = 50.0
    z_far: float = 350.0
    z_calibration_step_urad: float = 50.0
    lens_slot: int = 0
    camera_slot: int = 0
    exposure_time: float = 0.05
    slit_h_mm: float = 1.0
    slit_v_mm: float = 1.0
    convergence_threshold_urad: float = 5.0
    max_iterations: int = 5
    threshold_fraction: float = 0.5
    camera_pixel_um: float = 3.45
    magnification: float = 1.1
    min_jacobian_um_per_urad: float = 0.001    # sanity floor for calibration
    dry_run: bool = False
    enable_cora_log: bool = True


@dataclass
class Jacobian:
    """Table → centroid sensitivity at ``z_far`` (units: µm centroid / µrad table).

    Off-diagonal terms are kept but only the diagonal (``J_AY_X``,
    ``J_AX_Y``) is used by the iteration formula in v0.0.1.
    """
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
        self._cam_prefix = CAM0_PREFIX if config.camera_slot == 0 else CAM1_PREFIX

    # ---- step 1-5: setup --------------------------------------------------

    def setup(self) -> None:
        c = self.config
        log.info("step 1: select lens slot %d (1.1× = 0)", c.lens_slot)
        caput_wait(PV_LENS_SELECT, c.lens_slot)
        cawait_value(PV_LENS_SELECTED, c.lens_slot, timeout=120)

        log.info("step 2: select camera slot %d (Oryx 5MP = 0)", c.camera_slot)
        caput_wait(PV_CAMERA_SELECT, c.camera_slot)
        cawait_value(PV_CAMERA_SELECTED, c.camera_slot, timeout=120)

        log.info("step 3: exposure time = %s s", c.exposure_time)
        caput_wait(f"{self._cam_prefix}cam1:AcquireTime", c.exposure_time)

        log.info("step 4: B-station slits H=%s mm, V=%s mm",
                 c.slit_h_mm, c.slit_v_mm)
        set_horizontal_aperture(c.slit_h_mm)
        set_vertical_aperture(c.slit_v_mm)

        log.info("step 5: open front-end shutter")
        caput(PV_FES_OPEN, 1)
        # FES Position == 1 means OPEN at S02BM convention; adapt if needed
        cawait_value(PV_FES_POSITION, 1, timeout=30)

    # ---- step 6: record baseline -----------------------------------------

    def record_baseline(self) -> None:
        self.baseline = Baseline(
            table_AY=float(caget(PV_TABLE_AY_RBV)),
            table_AX=float(caget(PV_TABLE_AX_RBV)),
        )
        log.info("step 6: baseline table positions AY=%.3f µrad, AX=%.3f µrad",
                 self.baseline.table_AY, self.baseline.table_AX)

    # ---- step 7: calibrate Jacobian --------------------------------------

    def _move_z(self, z_mm: float) -> None:
        log.debug("move Z to %.3f mm", z_mm)
        if self.config.dry_run:
            log.info("[dry-run] would move %s to %.3f", PV_OP_Z_MOTOR, z_mm)
            return
        move_motor(PV_OP_Z_MOTOR, z_mm, timeout=120)

    def _move_table_axis(self, axis_pv: str, value: float,
                         dmov_pv: str) -> None:
        if self.config.dry_run:
            log.info("[dry-run] would move %s to %.3f µrad", axis_pv, value)
            return
        caput_wait(axis_pv, value, dmov_pvname=dmov_pv, timeout=60)

    def _measure_centroid(self) -> tuple[float, float]:
        """Acquire one image and return centroid in object-side µm."""
        if self.config.dry_run:
            log.info("[dry-run] would acquire image and fit centroid")
            return (0.0, 0.0)
        frame = acquire_image(self._cam_prefix,
                              exposure_time=self.config.exposure_time)
        com = center_of_mass(frame, self.config.threshold_fraction)
        if com is None:
            raise RuntimeError("centroid fit failed: no signal above threshold "
                               "(slits too closed, beam off, shutter shut?)")
        return pixels_to_object_um(com,
                                   camera_pixel_um=self.config.camera_pixel_um,
                                   magnification=self.config.magnification)

    def calibrate_jacobian(self) -> None:
        """Step 7: discover table → centroid Jacobian at ``z_far``.

        The Jacobian is measured at fixed Z to decouple the table
        perturbation from the rail tilt the procedure is trying to
        remove.
        """
        c = self.config
        delta = c.z_calibration_step_urad

        log.info("step 7: calibrate Jacobian at z_far = %.3f mm "
                 "with Δ = %.1f µrad", c.z_far, delta)
        self._move_z(c.z_far)
        x_f0, y_f0 = self._measure_centroid()
        log.debug("  baseline centroid at z_far: X=%.3f µm, Y=%.3f µm",
                  x_f0, y_f0)

        # column 1: AY perturbation
        self._move_table_axis(PV_TABLE_AY, self.baseline.table_AY + delta,
                              PV_TABLE_AY_DMOV)
        x_f1, y_f1 = self._measure_centroid()
        self.jacobian.J_AY_X = (x_f1 - x_f0) / delta
        self.jacobian.J_AY_Y = (y_f1 - y_f0) / delta
        log.info("  J_AY_X = %+.4f µm/µrad, J_AY_Y = %+.4f µm/µrad",
                 self.jacobian.J_AY_X, self.jacobian.J_AY_Y)
        self._move_table_axis(PV_TABLE_AY, self.baseline.table_AY,
                              PV_TABLE_AY_DMOV)

        # column 2: AX perturbation
        self._move_table_axis(PV_TABLE_AX, self.baseline.table_AX + delta,
                              PV_TABLE_AX_DMOV)
        x_f2, y_f2 = self._measure_centroid()
        self.jacobian.J_AX_X = (x_f2 - x_f0) / delta
        self.jacobian.J_AX_Y = (y_f2 - y_f0) / delta
        log.info("  J_AX_X = %+.4f µm/µrad, J_AX_Y = %+.4f µm/µrad",
                 self.jacobian.J_AX_X, self.jacobian.J_AX_Y)
        self._move_table_axis(PV_TABLE_AX, self.baseline.table_AX,
                              PV_TABLE_AX_DMOV)

        # sanity floor
        if (abs(self.jacobian.J_AY_X) < c.min_jacobian_um_per_urad
                or abs(self.jacobian.J_AX_Y) < c.min_jacobian_um_per_urad):
            raise RuntimeError(
                f"Jacobian below sanity floor "
                f"(|J_AY_X|={abs(self.jacobian.J_AY_X):.5f}, "
                f"|J_AX_Y|={abs(self.jacobian.J_AX_Y):.5f}); "
                "test step too small, slits closed, or table not moving?"
            )

    # ---- step 8: iterative correction loop -------------------------------

    def _measure_slope(self) -> tuple[float, float, float, float, float, float]:
        """At z_near and z_far, return centroids + per-mm slopes."""
        self._move_z(self.config.z_near)
        x_n, y_n = self._measure_centroid()
        self._move_z(self.config.z_far)
        x_f, y_f = self._measure_centroid()
        dz = self.config.z_far - self.config.z_near
        slope_X = (x_f - x_n) / dz
        slope_Y = (y_f - y_n) / dz
        return x_n, y_n, x_f, y_f, slope_X, slope_Y

    def iterate(self) -> bool:
        """Iterate measure-and-correct until convergence or max_iterations.

        Returns True if converged, False if max_iterations reached.
        """
        c = self.config
        log.info("step 8: iterate (max_iterations=%d, threshold=%.1f µrad)",
                 c.max_iterations, c.convergence_threshold_urad)

        # sign of Jacobian diagonal → sign of correction
        sign_AY = math.copysign(1.0, self.jacobian.J_AY_X)
        sign_AX = math.copysign(1.0, self.jacobian.J_AX_Y)

        for i in range(1, c.max_iterations + 1):
            x_n, y_n, x_f, y_f, slope_X, slope_Y = self._measure_slope()
            tilt_X_urad = slope_X * 1000.0      # µm/mm → µrad
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
                log.info("  iter %d: CONVERGED  tilt_X=%.2f µrad, tilt_Y=%.2f µrad",
                         i, tilt_X_urad, tilt_Y_urad)
                return True

            # geometric correction; sign from calibrated Jacobian
            d_AY = -sign_AY * tilt_X_urad
            d_AX = -sign_AX * tilt_Y_urad
            result.correction_AY_urad = d_AY
            result.correction_AX_urad = d_AX
            self.history.append(result)
            log.info("  iter %d: tilt_X=%+.2f µrad → ΔAY=%+.2f µrad; "
                     "tilt_Y=%+.2f µrad → ΔAX=%+.2f µrad",
                     i, tilt_X_urad, d_AY, tilt_Y_urad, d_AX)

            new_AY = float(caget(PV_TABLE_AY_RBV)) + d_AY
            new_AX = float(caget(PV_TABLE_AX_RBV)) + d_AX
            self._move_table_axis(PV_TABLE_AY, new_AY, PV_TABLE_AY_DMOV)
            self._move_table_axis(PV_TABLE_AX, new_AX, PV_TABLE_AX_DMOV)

        log.warning("step 8: did not converge after %d iterations", c.max_iterations)
        return False

    # ---- step 10: teardown -----------------------------------------------

    def teardown(self) -> None:
        log.info("step 10: close front-end shutter")
        caput(PV_FES_CLOSE, 1)

    # ---- orchestrator ----------------------------------------------------

    def run(self) -> bool:
        c = self.config
        cora = CoraProcedureLog(
            slug="detector_z_rail_alignment",
            target_asset_ids=[
                "Optique_Peter_focus_Z",
                "Detector_optical_table",       # pending cora registration
                "Oryx_5MP_camera",
                "Scintillator_LuAG",
            ],
            parameters=vars(c),
        ) if c.enable_cora_log else None

        if cora:
            cora.open()
        try:
            self.setup()
            if cora: cora.append_step("setup", {"timestamp": time.time()})
            self.record_baseline()
            if cora: cora.append_step("baseline", vars(self.baseline))
            self.calibrate_jacobian()
            if cora: cora.append_step("calibrate", vars(self.jacobian))
            converged = self.iterate()
            if cora: cora.append_step("iterate", {"iterations": len(self.history),
                                                   "converged": converged})
            return converged
        finally:
            self.teardown()
            if cora:
                cora.close(outcome="complete" if self.history and self.history[-1].converged
                           else "truncate")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Align the Optique Peter detector Z rail to the beam "
                    "using the detector optical table beneath the rail.")
    p.add_argument("--z-near", type=float, default=50.0,
                   help="Upstream Z anchor (mm). Default: 50.")
    p.add_argument("--z-far", type=float, default=350.0,
                   help="Downstream Z anchor (mm). Default: 350.")
    p.add_argument("--calibration-step-urad", type=float, default=50.0,
                   help="Test step applied to table AY/AX for Jacobian "
                        "discovery. Default: 50 µrad.")
    p.add_argument("--lens-slot", type=int, default=0,
                   help="MCTOptics lens slot (0=1.1×, 1=2×, 2=10×). Default: 0.")
    p.add_argument("--camera-slot", type=int, default=0,
                   help="MCTOptics camera slot (0=Oryx 5MP, 1=Oryx 31MP). Default: 0.")
    p.add_argument("--exposure-time", type=float, default=0.05,
                   help="Camera exposure (s). Default: 0.05.")
    p.add_argument("--slit-h", type=float, default=1.0,
                   help="B-station horizontal slit aperture (mm). Default: 1.0.")
    p.add_argument("--slit-v", type=float, default=1.0,
                   help="B-station vertical slit aperture (mm). Default: 1.0.")
    p.add_argument("--convergence-urad", type=float, default=5.0,
                   help="Stop iterating when |slope_X|, |slope_Y| are below "
                        "this. Default: 5 µrad.")
    p.add_argument("--max-iterations", type=int, default=5)
    p.add_argument("--threshold-fraction", type=float, default=0.5,
                   help="Centroid threshold as fraction of frame max. Default: 0.5.")
    p.add_argument("--camera-pixel-um", type=float, default=3.45,
                   help="Camera sensor pixel size (µm). Default: 3.45 (Oryx 5MP).")
    p.add_argument("--magnification", type=float, default=1.1,
                   help="Objective magnification. Default: 1.1 (lens 0).")
    p.add_argument("--no-cora-log", action="store_true",
                   help="Skip cora Procedure-record logging.")
    p.add_argument("--dry-run", action="store_true",
                   help="Do not move motors or trigger acquisitions; "
                        "log the would-be actions and exit.")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s  %(levelname)-7s  %(name)s: %(message)s")
    config = Config(
        z_near=args.z_near,
        z_far=args.z_far,
        z_calibration_step_urad=args.calibration_step_urad,
        lens_slot=args.lens_slot,
        camera_slot=args.camera_slot,
        exposure_time=args.exposure_time,
        slit_h_mm=args.slit_h,
        slit_v_mm=args.slit_v,
        convergence_threshold_urad=args.convergence_urad,
        max_iterations=args.max_iterations,
        threshold_fraction=args.threshold_fraction,
        camera_pixel_um=args.camera_pixel_um,
        magnification=args.magnification,
        dry_run=args.dry_run,
        enable_cora_log=(not args.no_cora_log),
    )
    proc = DetectorZRailAlignment(config)
    try:
        converged = proc.run()
    except Exception as exc:
        log.error("procedure aborted: %s", exc, exc_info=True)
        return 2
    return 0 if converged else 1


if __name__ == "__main__":
    sys.exit(main())
