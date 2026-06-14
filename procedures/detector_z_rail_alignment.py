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

import numpy as np
from epics import caget, caput

from ._shared.centroid import centroid_above_background, pixels_to_object_um
from ._shared.cora_log import CoraProcedureLog
from ._shared.epics import (
    OperatorAbort,
    acquire_image,
    confirm_motion,
    move_motor,
    move_table_axis,
    safe_restore,
)
from ._shared.log import setup_console_logger


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

# MCTOptics IOC (selectors are operator-set; procedure only reads).
# Use the setpoint (Select) PVs not the readback (Selected) PVs:
# at this IOC version Selected may not exist, and Selected uses
# different display strings ("Camera Selected 1") than Select
# ("Camera 1"). We key by integer enum index (returned by caget
# without as_string) to be tolerant of either label scheme.
PV_LENS_SELECT = "2bm:MCTOptics:LensSelect"
PV_CAMERA_SELECT = "2bm:MCTOptics:CameraSelect"

# Camera prefix by mbbo enum index (per pvinfo on 2bm:MCTOptics:CameraSelect:
# STATE 0 = "Camera 1", STATE 1 = "Camera 2").
CAMERA_PREFIXES_BY_INDEX = {
    0: "2bmSP1:",   # FLIR Oryx 5MP
    1: "2bmSP2:",   # FLIR Oryx 31MP
}

# Lens magnification by mbbo enum index (STATE 0 = "Lens1", etc.).
# Update these when the installed objectives change.
LENS_MAGNIFICATIONS_BY_INDEX = {
    0: 1.1,
    1: 5.0,
    2: 10.0,
}

# Z-stage software guard (procedure-level; does not modify motor limits)
Z_GUARD_MIN_MM = 200.0
Z_GUARD_MAX_MM = 500.0


# ---------------------------------------------------------------------------
# Preconditions (Layer 2: machine-readable form of the doc table).
#
# Data ONLY -- not exercised at runtime in v0.0.1. The procedure relies on
# the operator to establish each of these before launching. The single
# implicit runtime check is in `_measure_centroid`: if no signal is found
# above threshold, the error message points back to this list.
#
# Cora can ingest the list once the schema lands. The fields:
#   state:        unique identifier; matches the State column in the
#                 item_002.rst Preconditions table.
#   description:  free-text human description.
#   predicate:    informal expression of the test; not a callable today.
#                 Includes the PV name(s) where one is known, "TBD" when
#                 the satisfying procedure is itself a stub, or a free-text
#                 description when the predicate spans multiple PVs.
#   satisfied_by: procedure name (the cora Method.name) that establishes
#                 the state.
#   doc:          path to the satisfying procedure's spec doc (relative to
#                 the 2bm-docs docs/source root).
# ---------------------------------------------------------------------------

PRECONDITIONS = [
    {
        "state": "beamline_enabled",
        "description": "Hutches searched + locked; ACIS upstream permit "
                       "(composites BLEPS + machine state); FES open.",
        "predicate": "S02BM-PSS:StaA:SecureM == 1 (ON)  AND  "
                     "S02BM-PSS:StaB:SecureM == 1 (ON)  AND  "
                     "S02BM-PSS:FES:BeamBlockingM == 0 (OFF)  AND  "
                     "SR-ACIS:2BM:FesPermitM == 1 (ON, ACIS upstream "
                     "permit; aggregates BLEPS + APS machine state)",
        "satisfied_by": "enable_beamline",
        "doc": "procedures/item_003.rst",
    },
    {
        "state": "a_slits_open",
        "description": "A-station slits open enough that propagation "
                       "produces ~1x1 mm at sample / detector.",
        "predicate": "A-station Slit H size RBV >= 0.5 mm AND "
                     "Slit V size RBV >= 0.5 mm (PVs TBD; see item_020)",
        "satisfied_by": "set_a_slits",
        "doc": "procedures/item_004.rst",
    },
    {
        "state": "energy_configured",
        "description": "Mirror M1 + DMM at energy-dependent positions "
                       "per the energy package's lookup tables.",
        "predicate": "TBD (parameterised by energy; needs lookup from "
                     "https://github.com/xray-imaging/energy/blob/main/"
                     "src/energy/data/energy2bm.json)",
        "satisfied_by": "set_energy_to_preselect",
        "doc": "procedures/item_005.rst",
    },
    {
        "state": "flag_in_beam",
        "description": "Diagnostic phosphor flag (2bma:m44, vertical) at "
                       "the position appropriate for the beamline mode: "
                       "0 mm (user) for pink, energy-dependent value from "
                       "energy_move_flag for mono.",
        "predicate": "2bma:m44.RBV ~ target_flag_y_mm  (target is mode-"
                     "and-energy dependent; key 'energy_move_flag' in "
                     "https://github.com/xray-imaging/energy/blob/main/"
                     "src/energy/data/energy2bm.json for mono; 0 for pink).",
        "satisfied_by": "set_flag_in",
        "doc": "procedures/item_006.rst",
    },
    {
        "state": "b_shutter_open",
        "description": "P6-50 safety shutter open (beam reaches 2-BM-B).",
        "predicate": "S02BM-PSS:SBS:BeamBlockingM == 0  "
                     "(STATE 0 OFF = NOT blocking = OPEN; inverted enum)",
        "satisfied_by": "open_b_shutter",
        "doc": "procedures/item_007.rst",
    },
    {
        "state": "b_slits_configured",
        "description": "B-station slits at 1.0x1.0 mm centred on the "
                       "energy-set vertical Y.",
        "predicate": "(2bma:m12.RBV - 2bma:m11.RBV) ~ 1.0 mm  AND  "
                     "(2bma:m9.RBV - 2bma:m10.RBV) ~ 1.0 mm  AND  "
                     "(2bma:m9.RBV + 2bma:m10.RBV)/2 ~ y_for_energy(E)",
        "satisfied_by": "set_b_slits",
        "doc": "procedures/item_008.rst",
    },
    {
        "state": "sample_out_of_beam",
        "description": "Relevant sample-stack axis (mount-dependent) "
                       "at its out-of-beam position.",
        "predicate": "TBD (axis depends on current sample mount)",
        "satisfied_by": "move_sample_out_of_beam",
        "doc": "procedures/item_009.rst",
    },
    {
        "state": "microscope_configured",
        "description": "MCTOptics lens at 1.1x (slot 0); detector table "
                       "Y at beam centre; Z stage in mid-band.",
        "predicate": "2bm:MCTOptics:LensSelect == 0  AND  "
                     "2bmb:table3.Y ~ y_for_energy(E)  AND  "
                     "200 <= 2bmbAERO:m1.RBV <= 500",
        "satisfied_by": "configure_microscope_for_alignment",
        "doc": "procedures/item_010.rst",
    },
    # Below: preconditions the procedure relies on but does not have a
    # dedicated satisfying procedure for. Documented here so cora's
    # dependency graph is complete; operator (or another procedure) is
    # responsible.
    {
        "state": "fes_shutter_open",
        "description": "Front-end shutter open (sub-precondition of "
                       "beamline_enabled, called out separately).",
        "predicate": "S02BM-PSS:FES:BeamBlockingM == 0  (OPEN)",
        "satisfied_by": "enable_beamline",  # bundled
        "doc": "procedures/item_003.rst",
    },
    {
        "state": "mctoptics_ioc_reachable",
        "description": "2bm:MCTOptics IOC running and reachable from the "
                       "host running the procedure (the IOC lives on "
                       "tomdet.xray.aps.anl.gov).",
        "predicate": "caget(2bm:MCTOptics:CameraSelect) succeeds",
        "satisfied_by": None,
        "doc": None,
    },
    {
        "state": "pss_interlocks_satisfied",
        "description": "Nobody in 2-BM-A or 2-BM-B; hutch searches "
                       "complete and locked.",
        "predicate": "S02BM-PSS:StaA:SecureM == 1 (ON)  AND  "
                     "S02BM-PSS:StaB:SecureM == 1 (ON). Sub-condition of "
                     "beamline_enabled.",
        "satisfied_by": None,
        "doc": None,
    },
]


# Aerotech PRO225SL-1000 datasheet straightness floor (~9.5 um over
# the full 1 m travel). Treated here as a per-mm slope contribution
# that no amount of linear table-tilt correction can remove; it sets
# the lower bound for any convergence threshold regardless of optics.
RAIL_STRAIGHTNESS_FLOOR_URAD = 10.0


def expected_noise_floor_urad(pixel_um: float, bin_x: int, magnification: float,
                              dz_mm: float,
                              centroid_noise_pix: float = 1.0,
                              straightness_floor_urad: float =
                                  RAIL_STRAIGHTNESS_FLOOR_URAD) -> float:
    """Estimate the practical floor on the convergence threshold.

    Two contributors:

    - **Measurement noise** -- ``centroid_noise_pix x object_pitch_um / dz``,
      expressed in urad. Scales inversely with magnification: higher mag ->
      finer measurable slope.
    - **Rail straightness** -- the PRO225SL's intrinsic non-straightness.
      Independent of optics; a hard floor that linear table tilt cannot
      remove.

    Returns the larger of the two -- the procedure cannot meaningfully
    converge below this value at the given operating point.
    """
    object_pitch_um = pixel_um * bin_x / magnification
    measurement_urad = (centroid_noise_pix * object_pitch_um
                        / dz_mm * 1000.0)
    return max(measurement_urad, straightness_floor_urad)


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration + state
# ---------------------------------------------------------------------------

@dataclass
class Config:
    z_near: float = 200.0
    z_far: float = 500.0
    z_calibration_step_urad: float = 50.0
    exposure_time: float = 0.2
    # None -> auto-compute at runtime from lens/binning/dz via
    # expected_noise_floor_urad() x convergence_safety_margin.
    convergence_threshold_urad: float | None = None
    convergence_safety_margin: float = 1.5
    centroid_noise_pix: float = 1.0
    # Centroid algorithm tunables (centroid_above_background):
    # bg_corner_size = pixels per side of each of 4 corner boxes
    # used to estimate background mean+std. bg_sigma_threshold =
    # the threshold is bg_mean + N*bg_std; smaller N includes more
    # of the dim halo (and more noise), larger N keeps only clearly-
    # above-background pixels.
    bg_corner_size: int = 100
    bg_sigma_threshold: float = 5.0
    max_iterations: int = 5
    camera_pixel_um: float = 3.45
    # Damping factor on the computed correction (0 < damping <= 1). 0.5
    # halves the move each iteration; safer in the presence of an
    # imperfect sensitivity matrix or table cross-coupling beyond what
    # a 2x2 linear model captures.
    damping: float = 0.5
    # Abort if |slope| grows by more than this factor from one iter
    # to the next -- protection against runaway divergence.
    divergence_grow_threshold: float = 1.5
    # Abort if |slope| exceeds the best |slope| seen so far in this
    # run by more than this factor. Catches the slow-bleed divergence
    # that the per-step ratio misses (e.g. 1.2x per iter over 9 iters
    # = 5x cumulative, well past anything useful).
    divergence_cumulative_threshold: float = 2.0
    # Warn at this condition number of the sensitivity matrix M --
    # higher than this means one axis-direction has marginal SNR and
    # the procedure is at risk of sign-flip divergence. Doesn't auto-
    # abort; operator can choose to proceed.
    sensitivity_cond_warn_threshold: float = 5.0
    # If True, prompt y/N before each Z measurement move. If False
    # (default), Z measurement moves are announced but not gated --
    # they stay within the [Z_GUARD_MIN, Z_GUARD_MAX] band and don't
    # change the alignment, only sample it. Table moves are ALWAYS
    # gated regardless of this flag.
    gate_z: bool = False
    # Minimum |det(M)| where M is the 2x2 slope-sensitivity matrix
    # (units: (um/mm)^2 / urad^2). Below this M is treated as
    # near-singular and calibration aborts with a clear message.
    # Set very low (1e-8) because the safety net is now the
    # max_correction_per_iter_urad clip + divergence guard + per-step
    # operator gate -- a moderately ill-conditioned M can still drive
    # useful corrections in its well-conditioned direction.
    min_sensitivity_det: float = 1.0e-8
    # Hard clip on |d_AY| and |d_AX| per iteration (urad). Even if M_inv
    # computes a huge correction (because the table has weak authority
    # in one direction and we're trying to fully zero the slope), apply
    # at most this much per iteration. Keeps us in the linear range
    # near the calibration point; convergence over more iterations
    # rather than one big move.
    max_correction_per_iter_urad: float = 200.0
    dry_run: bool = False
    auto_yes: bool = False
    confirm_restore: bool = False
    enable_cora_log: bool = True


@dataclass
class Sensitivity:
    """2x2 slope-per-axis sensitivity matrix M built by calibrate_sensitivity().

    Defines how table tilts affect the **slope** of centroid drift vs Z
    (the quantity the procedure is trying to drive to zero):

        Δslope_X (um/mm) = M_AY_X * ΔAY (urad)  +  M_AX_X * ΔAX (urad)
        Δslope_Y (um/mm) = M_AY_Y * ΔAY (urad)  +  M_AX_Y * ΔAX (urad)

    Diagonal terms (M_AY_X, M_AX_Y) capture the principal effect of
    each table axis on the slope it primarily controls; off-diagonal
    terms (M_AY_Y, M_AX_X) capture cross-coupling between the two
    table axes and the centroid axes (which is significant on this
    table -- the previous diagonal-only correction diverged).

    Iteration solves M @ (ΔAY, ΔAX) = -(slope_X, slope_Y) for the
    correction (which is then damped by config.damping).
    """
    M_AY_X: float = 0.0
    M_AY_Y: float = 0.0
    M_AX_X: float = 0.0
    M_AX_Y: float = 0.0

    def as_matrix(self) -> np.ndarray:
        return np.array([[self.M_AY_X, self.M_AX_X],
                         [self.M_AY_Y, self.M_AX_Y]])

    def determinant(self) -> float:
        return self.M_AY_X * self.M_AX_Y - self.M_AX_X * self.M_AY_Y


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

    ``table_AY`` / ``table_AX`` are snapshotted at entry but only put
    back when ``restore(restore_table=True)`` is called -- i.e. on
    OperatorAbort or any exception. On clean convergence the procedure
    leaves the optimised AY/AX in place (they're the deliberate output).
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
    table_AY: float = 0.0
    table_AX: float = 0.0

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
            table_AY=float(caget(PV_TABLE_AY)),
            table_AX=float(caget(PV_TABLE_AX)),
        )

    def restore_plan(self, restore_table: bool = True) -> list[dict]:
        """Plan-block representation of what ``restore()`` will do.

        If ``restore_table=False`` the table AY/AX rows are omitted
        (clean-convergence path -- the new values are the procedure's
        deliberate output and stay in place).
        """
        cp = self.cam_prefix
        plan: list[dict] = []
        # Table FIRST -- matches the restore() execution order so the
        # plan block reflects what actually runs first.
        if restore_table:
            plan.extend([
                {"pv": PV_TABLE_AY, "current": "?",
                 "target": self.table_AY, "units": "deg"},
                {"pv": PV_TABLE_AX, "current": "?",
                 "target": self.table_AX, "units": "deg"},
            ])
        plan.extend([
            {"pv": PV_OP_Z_MOTOR, "current": "?",
             "target": self.z_position, "units": "mm"},
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

    def restore(self, restore_table: bool = True) -> None:
        """Restore PVs in priority order:

        1. Table AY/AX (highest priority -- affects alignment
           fundamentally, hardest to fix by hand, six jacks to
           coordinate). Skipped if ``restore_table=False`` (i.e. clean
           convergence wants to keep the new alignment).
        2. Z stage (moderate priority -- one motor, easy to recover
           manually if needed).
        3. Camera state (low priority -- nine caputs, all to fixed
           values; trivial to recover manually).

        Ordering matters when restore is interrupted (operator
        Ctrl-C during the restore itself): the most safety-critical
        PVs go back first, even if the rest never runs. See
        ``safe_restore`` for the per-action Ctrl-C handling.
        """
        cp = self.cam_prefix
        actions = []
        if restore_table:
            # Table FIRST -- the most safety-critical restore (six
            # jacks, kinematic coordination, alignment-defining).
            actions.extend([
                ("table.AY to baseline",
                 lambda: move_table_axis(PV_TABLE_AY, self.table_AY,
                                          TABLE_JACK_PREFIXES, timeout=60)),
                ("table.AX to baseline",
                 lambda: move_table_axis(PV_TABLE_AX, self.table_AX,
                                          TABLE_JACK_PREFIXES, timeout=60)),
            ])
        actions.extend([
            # Z stage second.
            ("Z stage to baseline",
             lambda: move_motor(PV_OP_Z_MOTOR, self.z_position, timeout=180)),
            # Camera state last -- a stranded camera config is the
            # easiest of the three categories to fix manually.
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

class DetectorZRailAlignment:
    """Stateful executor for ``detector_z_rail_alignment``."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.baseline = Baseline()
        self.sensitivity = Sensitivity()
        self.history: list[IterationResult] = []
        self._snapshot: _Snapshot | None = None
        self._cam_prefix: str = ""
        self._magnification: float = 1.0
        self._pixel_um: float = config.camera_pixel_um   # auto-set by detect_*
        # Effective convergence threshold -- resolved in run() after
        # detect_camera_and_lens() runs (needs binning + lens to compute
        # the noise floor).
        self._convergence_threshold_urad: float = 0.0
        # Set by iterate() when max_iterations is exhausted but the best
        # |tilt| seen is strictly better than the starting state -- the
        # finally-block then leaves the table at the best pose instead
        # of restoring it to baseline.
        self._best_state_committed: bool = False

        # Z-range guard (procedure-level safety band).
        if not (Z_GUARD_MIN_MM <= config.z_near < config.z_far <= Z_GUARD_MAX_MM):
            raise ValueError(
                f"Z range [{config.z_near}, {config.z_far}] mm violates "
                f"safety band [{Z_GUARD_MIN_MM}, {Z_GUARD_MAX_MM}] mm "
                f"(z_near must also be < z_far)"
            )

    # ---- detection of operator-set camera / lens --------------------------

    def detect_camera_and_lens(self) -> None:
        """Read MCTOptics ``CameraSelect`` / ``LensSelect`` (the setpoint
        mbbo records) and derive ``cam_prefix`` + ``magnification`` from
        their enum index. Operator-set; not modified."""
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

        # Human labels for logging only (don't gate on string match -- the
        # IOC may use either "Camera 1" or "Camera Selected 1" depending
        # on version, and we already have the authoritative index above).
        cam_label = caget(PV_CAMERA_SELECT, as_string=True) or f"index {cam_idx}"
        lens_label = caget(PV_LENS_SELECT, as_string=True) or f"index {lens_idx}"

        if cam_idx not in CAMERA_PREFIXES_BY_INDEX:
            raise RuntimeError(
                f"unknown camera enum index {cam_idx} ({cam_label!r}); "
                f"expected one of {sorted(CAMERA_PREFIXES_BY_INDEX)}"
            )
        if lens_idx not in LENS_MAGNIFICATIONS_BY_INDEX:
            raise RuntimeError(
                f"unknown lens enum index {lens_idx} ({lens_label!r}); "
                f"expected one of {sorted(LENS_MAGNIFICATIONS_BY_INDEX)}"
            )
        self._cam_prefix = CAMERA_PREFIXES_BY_INDEX[cam_idx]
        self._magnification = LENS_MAGNIFICATIONS_BY_INDEX[lens_idx]

        # Read camera binning so the effective pixel pitch reflects what
        # image1:ArrayData actually delivers. A 2x2-binned 6480x4860 sensor
        # returns 3240x2430 "pixels" each spanning two sensor pixels
        # (effective pitch = 2 * sensor pitch).
        bin_x = int(caget(f"{self._cam_prefix}cam1:BinX_RBV") or 1)
        bin_y = int(caget(f"{self._cam_prefix}cam1:BinY_RBV") or 1)
        if bin_x != bin_y:
            log.warning("camera BinX=%d != BinY=%d -- using BinX for the "
                        "pixel pitch; centroid x/y will be anisotropic",
                        bin_x, bin_y)
        self._pixel_um = self.config.camera_pixel_um * bin_x

        log.info("detected: camera=%s [idx %d] -> %s (bin %dx%d, "
                 "effective pixel pitch %.2f um); "
                 "lens=%s [idx %d] -> %.2fx magnification",
                 cam_label, cam_idx, self._cam_prefix,
                 bin_x, bin_y, self._pixel_um,
                 lens_label, lens_idx, self._magnification)

    # ---- resolve convergence threshold from physical floors ---------------

    def _resolve_convergence_threshold(self) -> None:
        """Compute the effective convergence threshold from the lens +
        binning + dz, OR honour the operator's --convergence-urad override.

        Two floors contribute:
        - Measurement noise: centroid_noise_pix * obj_pitch / dz (lens-dep).
        - Rail straightness: ~10 urad at any sub-300 mm range (independent).

        Default threshold = floor x safety_margin (1.5x). Operator overrides
        win, but a warning is logged if the override is below the floor
        (procedure cannot converge).
        """
        c = self.config
        dz = c.z_far - c.z_near
        floor = expected_noise_floor_urad(
            pixel_um=c.camera_pixel_um,
            bin_x=int(round(self._pixel_um / c.camera_pixel_um)),
            magnification=self._magnification,
            dz_mm=dz,
            centroid_noise_pix=c.centroid_noise_pix,
        )
        suggested = floor * c.convergence_safety_margin
        if c.convergence_threshold_urad is None:
            self._convergence_threshold_urad = suggested
            log.info("convergence threshold auto-set to %.1f urad "
                     "(noise floor %.1f urad x safety margin %.2f). "
                     "Override with --convergence-urad.",
                     suggested, floor, c.convergence_safety_margin)
        else:
            self._convergence_threshold_urad = c.convergence_threshold_urad
            if c.convergence_threshold_urad < floor:
                log.warning(
                    "operator-supplied convergence threshold %.1f urad "
                    "is BELOW the physical noise floor %.1f urad "
                    "(measurement + rail straightness). The procedure "
                    "cannot meaningfully converge to this; expect to "
                    "exhaust max_iterations.",
                    c.convergence_threshold_urad, floor)
            else:
                log.info("convergence threshold = %.1f urad "
                         "(operator-supplied; floor is %.1f urad)",
                         c.convergence_threshold_urad, floor)

    # ---- gated motion helpers --------------------------------------------

    def _gate(self, plan: list[dict], step_label: str) -> bool:
        return confirm_motion(
            plan,
            step_label=step_label,
            dry_run=self.config.dry_run,
            auto_yes=self.config.auto_yes,
        )

    def _gated_move_z(self, target: float, step_label: str) -> None:
        """Move Z. By default Z moves are announced but NOT gated --
        they stay within the safety band and don't change the
        alignment, only sample it. Set ``--gate-z`` for full gating."""
        if not (Z_GUARD_MIN_MM <= target <= Z_GUARD_MAX_MM):
            raise ValueError(
                f"Z target {target} mm outside safety band "
                f"[{Z_GUARD_MIN_MM}, {Z_GUARD_MAX_MM}]"
            )
        current = float(caget(f"{PV_OP_Z_MOTOR}.RBV"))
        proceed = confirm_motion(
            [{"pv": PV_OP_Z_MOTOR, "current": current,
              "target": target, "units": "mm"}],
            step_label=step_label,
            dry_run=self.config.dry_run,
            auto_yes=self.config.auto_yes,
            announce_only=not self.config.gate_z,
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
        """Acquire one image, fit centroid above threshold, return the
        centroid position in object-side micrometres.

        Runs in both dry-run and live modes -- only motor motion is
        skipped under --dry-run; the camera read still happens so the
        operator can validate the read+fit pipeline against the live
        beam before committing to any motor motion. The pixel-space
        centroid and its offset from the frame centre are logged so
        the operator can confirm visually that the algorithm has
        latched onto the bright spot they see on MEDM.
        """
        frame = acquire_image(self._cam_prefix,
                              exposure_time=self.config.exposure_time)
        result = centroid_above_background(
            frame,
            bg_corner_size=self.config.bg_corner_size,
            bg_sigma_threshold=self.config.bg_sigma_threshold,
        )
        if result is None:
            raise RuntimeError(
                "centroid fit failed: no pixels above background+"
                f"{self.config.bg_sigma_threshold:.1f}*std. "
                "Likely upstream-precondition failures (check, in order):\n"
                "  1. FES shutter open?  caget S02BM-PSS:FES:BeamBlockingM "
                "(expect OFF)\n"
                "  2. B-shutter open?    caget S02BM-PSS:SBS:BeamBlockingM "
                "(expect OFF)\n"
                "  3. Beam on / DMM at energy? (visible spot on live "
                "view?)\n"
                "  4. B-station slits open ~1x1 mm?  caget 2bma:m9 m10 m11 "
                "m12\n"
                "  5. Sample out of beam path?\n"
                "  6. Optique Peter at the right Z and table Y?\n"
                "See the Preconditions section of "
                "docs/source/procedures/item_002.rst for the full "
                "checklist (PRECONDITIONS in this module mirrors it)."
            )
        px, py, diag = result
        h, w = frame.shape
        dx_pix = px - w / 2.0
        dy_pix = py - h / 2.0
        x_um, y_um = pixels_to_object_um(
            (px, py),
            camera_pixel_um=self._pixel_um,   # sensor pitch * binning
            magnification=self._magnification,
        )
        log.info("centroid: pix=(%.1f, %.1f) in %dx%d frame; "
                 "offset-from-centre=(%+.1f, %+.1f) pix; "
                 "object-um=(%+.2f, %+.2f); beam=%d pix (%.2f%%), "
                 "threshold=%.0f (bg_median=%.0f, bg_sigma=%.1f)",
                 px, py, w, h, dx_pix, dy_pix, x_um, y_um,
                 diag["n_beam_pix"], 100 * diag["frame_pix_fraction"],
                 diag["threshold"], diag["bg_median"], diag["bg_sigma"])
        return (x_um, y_um)

    # ---- procedure phases ------------------------------------------------

    def record_baseline(self) -> None:
        self.baseline = Baseline(
            table_AY=float(caget(PV_TABLE_AY)),
            table_AX=float(caget(PV_TABLE_AX)),
        )
        log.info("baseline table AY=%.6g, AX=%.6g (deg)",
                 self.baseline.table_AY, self.baseline.table_AX)

    def _measure_slope(self) -> tuple[float, float, float, float, float, float]:
        """Move Z to z_near, acquire; move Z to z_far, acquire.

        Returns ``(x_n, y_n, x_f, y_f, slope_X, slope_Y)`` where the
        slopes are in object-side micrometres per mm of Z travel.
        """
        self._gated_move_z(self.config.z_near, "measure slope: Z to near")
        x_n, y_n = self._measure_centroid()
        self._gated_move_z(self.config.z_far, "measure slope: Z to far")
        x_f, y_f = self._measure_centroid()
        dz = self.config.z_far - self.config.z_near
        return x_n, y_n, x_f, y_f, (x_f - x_n) / dz, (y_f - y_n) / dz

    def calibrate_sensitivity(self) -> None:
        """Build the 2x2 slope-sensitivity matrix M.

        For each of (AY, AX): measure baseline slope (Z near + Z far),
        perturb the axis by ``z_calibration_step_urad``, re-measure
        slope, restore. The diagonals tell us how each axis affects
        the slope it primarily controls; the off-diagonals capture
        cross-coupling.

        This is the right physical quantity for the iteration step.
        The previous design measured centroid-shift-at-z-far per
        axis-urad ("Jacobian"), which is geometry-dependent and does
        NOT directly drive slope correction (uniform centroid shifts
        cancel between z_near and z_far and leave slope unchanged).
        """
        c = self.config
        delta_urad = c.z_calibration_step_urad
        delta_deg = delta_urad * 5.72958e-5

        log.info("calibrate sensitivity matrix at z=[%.0f, %.0f] mm "
                 "with delta=%.1f urad (%.3e deg)",
                 c.z_near, c.z_far, delta_urad, delta_deg)

        # Baseline slope
        log.info("calibrate: measure baseline slope")
        _, _, _, _, slope0_X, slope0_Y = self._measure_slope()
        log.info("  baseline: slope_X=%+.4f um/mm (tilt %+.1f urad), "
                 "slope_Y=%+.4f um/mm (tilt %+.1f urad)",
                 slope0_X, slope0_X * 1000.0,
                 slope0_Y, slope0_Y * 1000.0)

        # AY perturb + re-measure
        self._gated_move_table(
            PV_TABLE_AY, self.baseline.table_AY + delta_deg,
            f"calibration: perturb AY by +{delta_urad:.1f} urad")
        log.info("calibrate: measure slope with AY perturbed")
        _, _, _, _, slope_AY_X, slope_AY_Y = self._measure_slope()
        log.info("  AY-perturbed: slope_X=%+.4f, slope_Y=%+.4f um/mm",
                 slope_AY_X, slope_AY_Y)
        self._gated_move_table(
            PV_TABLE_AY, self.baseline.table_AY,
            "calibration: restore AY")

        # AX perturb + re-measure
        self._gated_move_table(
            PV_TABLE_AX, self.baseline.table_AX + delta_deg,
            f"calibration: perturb AX by +{delta_urad:.1f} urad")
        log.info("calibrate: measure slope with AX perturbed")
        _, _, _, _, slope_AX_X, slope_AX_Y = self._measure_slope()
        log.info("  AX-perturbed: slope_X=%+.4f, slope_Y=%+.4f um/mm",
                 slope_AX_X, slope_AX_Y)
        self._gated_move_table(
            PV_TABLE_AX, self.baseline.table_AX,
            "calibration: restore AX")

        # Build M: rows = (slope_X, slope_Y), cols = (AY, AX)
        self.sensitivity = Sensitivity(
            M_AY_X=(slope_AY_X - slope0_X) / delta_urad,
            M_AY_Y=(slope_AY_Y - slope0_Y) / delta_urad,
            M_AX_X=(slope_AX_X - slope0_X) / delta_urad,
            M_AX_Y=(slope_AX_Y - slope0_Y) / delta_urad,
        )
        M = self.sensitivity
        det = M.determinant()
        log.info("sensitivity matrix M (um/mm of slope per urad of axis):")
        log.info("  d_slope_X = %+.6f * dAY + %+.6f * dAX",
                 M.M_AY_X, M.M_AX_X)
        log.info("  d_slope_Y = %+.6f * dAY + %+.6f * dAX",
                 M.M_AY_Y, M.M_AX_Y)
        log.info("  det(M) = %+.4e", det)

        # SVD diagnostic: condition number tells the operator whether
        # the procedure has independent control over both slopes. High
        # cond (>10) means one axis-combination has weak slope authority
        # -- corrections in that direction will be either ineffective
        # (good: max-correction clip stops us pushing too hard) or
        # over-amplified by M_inv (also clipped). Cond > 100 means
        # convergence in one direction is essentially impossible with
        # this table geometry.
        try:
            sv = np.linalg.svd(M.as_matrix(), compute_uv=False)
            cond = sv[0] / sv[1] if sv[1] > 0 else float("inf")
            log.info("  singular values: %.3e, %.3e   condition number: %.1f",
                     sv[0], sv[1], cond)
            if cond > 100:
                log.warning("  VERY high condition number -- table has weak "
                            "control over one slope direction; expect "
                            "convergence in only the dominant direction.")
            elif cond > c.sensitivity_cond_warn_threshold:
                log.warning("  elevated condition number %.1f (warn at "
                            "%.1f). One column of M has marginal SNR; "
                            "the sign of the weaker entry may be noise. "
                            "If iteration diverges, abort and re-run "
                            "calibrate with a larger "
                            "--calibration-step-urad (try 100 or 200).",
                            cond, c.sensitivity_cond_warn_threshold)
        except Exception as exc:
            log.debug("SVD diagnostic failed: %s", exc)

        # Sanity check: M not near-singular. Skip in dry-run (centroids
        # are real but Z/table didn't move, so all measurements are at
        # the same physical state -> matrix is exactly zero).
        if not c.dry_run and abs(det) < c.min_sensitivity_det:
            raise RuntimeError(
                f"sensitivity matrix near-singular "
                f"(|det|={abs(det):.4e} < min={c.min_sensitivity_det:.4e}). "
                f"Likely causes: calibration step ({delta_urad} urad) too "
                "small for the centroid noise floor, slits over-closed, "
                "or table AY and AX have near-parallel slope effects. "
                "Try --calibration-step-urad 100."
            )

    def iterate(self) -> bool:
        """Iterative correction using M_inv @ -slope, with damping
        and a divergence guard.

        Tracks the best |tilt| seen across iterations along with the
        table pose at that measurement. If ``max_iterations`` is hit
        without satisfying the convergence threshold, and the best
        |tilt| is strictly better than the starting state, the table
        is moved back to that best pose and the run is declared a
        "soft commit" (``self._best_state_committed = True``). The
        finally-block in ``run()`` then leaves the table at the best
        pose instead of restoring it to baseline.
        """
        c = self.config
        threshold = self._convergence_threshold_urad
        log.info("iterate (max_iterations=%d, threshold=%.1f urad, "
                 "damping=%.2f, divergence_grow_threshold=%.2fx)",
                 c.max_iterations, threshold,
                 c.damping, c.divergence_grow_threshold)

        deg_per_urad = 5.72958e-5
        M = self.sensitivity.as_matrix()
        try:
            M_inv = np.linalg.inv(M)
        except np.linalg.LinAlgError as exc:
            raise RuntimeError(f"sensitivity matrix not invertible: {exc} "
                               "(re-calibrate with larger step)") from exc

        prev_slope_mag = None
        # Best-state tracking. The "starting" pose is the table state at
        # iter 1's measurement (== baseline). best_pose is updated each
        # time a strictly better |tilt| is seen.
        starting_tilt_mag: float | None = None
        best_iter = 0
        best_tilt_mag = float("inf")
        best_AY = self.baseline.table_AY
        best_AX = self.baseline.table_AX

        for i in range(1, c.max_iterations + 1):
            # Record pose at this measurement (before the iteration's
            # correction is applied). For iter 1 this is the baseline;
            # for iter N>1 it's the pose set by iter N-1's correction.
            pose_AY = float(caget(PV_TABLE_AY))
            pose_AX = float(caget(PV_TABLE_AX))

            x_n, y_n, x_f, y_f, slope_X, slope_Y = self._measure_slope()
            tilt_X = slope_X * 1000.0
            tilt_Y = slope_Y * 1000.0
            slope_mag = math.hypot(tilt_X, tilt_Y)
            if starting_tilt_mag is None:
                starting_tilt_mag = slope_mag

            # Update best state if this measurement beats the prior best.
            if slope_mag < best_tilt_mag:
                best_iter = i
                best_tilt_mag = slope_mag
                best_AY = pose_AY
                best_AX = pose_AX

            # Divergence guards (two of them):
            #
            # 1. Per-iteration ratio: catches a single big step. Works
            #    well for sudden divergence but misses slow-bleed cases
            #    where each step grows by <1.5x but they accumulate.
            #
            # 2. Cumulative-vs-best ratio: catches the slow-bleed case.
            #    If |slope| has grown to >2x its best-seen value, the
            #    procedure has clearly gone the wrong way and isn't
            #    recovering. Abort -- restore puts table back to
            #    baseline. (best_tilt_mag was already updated above
            #    before this check, so if THIS iter is the best,
            #    slope_mag == best_tilt_mag and the check is moot.)
            if not c.dry_run:
                if (prev_slope_mag is not None and
                        slope_mag > prev_slope_mag * c.divergence_grow_threshold):
                    log.error("DIVERGING at iter %d: |slope| grew from %.1f "
                              "to %.1f urad", i, prev_slope_mag, slope_mag)
                    raise RuntimeError(
                        f"divergence at iter {i}: |slope|={slope_mag:.1f} "
                        f"urad exceeds previous {prev_slope_mag:.1f} urad "
                        f"by >{c.divergence_grow_threshold:.1f}x. "
                        "Sensitivity matrix may be inaccurate; "
                        "re-calibrate with larger --calibration-step-urad."
                    )
                if (best_tilt_mag < float("inf") and
                        slope_mag > best_tilt_mag *
                        c.divergence_cumulative_threshold):
                    log.error("DIVERGING (cumulative) at iter %d: "
                              "|slope|=%.1f urad exceeds best-seen "
                              "%.1f urad (iter %d) by >%.1fx",
                              i, slope_mag, best_tilt_mag, best_iter,
                              c.divergence_cumulative_threshold)
                    raise RuntimeError(
                        f"cumulative divergence at iter {i}: |slope|"
                        f"={slope_mag:.1f} urad has grown to >"
                        f"{c.divergence_cumulative_threshold:.1f}x the "
                        f"best |slope|={best_tilt_mag:.1f} urad seen "
                        f"(iter {best_iter}). Procedure isn't recovering; "
                        "calibration likely has a sign error on one "
                        "axis. Re-calibrate with larger "
                        "--calibration-step-urad (try 100 or 200)."
                    )
            prev_slope_mag = slope_mag

            converged = (abs(tilt_X) <= threshold
                         and abs(tilt_Y) <= threshold)

            result = IterationResult(
                iteration=i,
                X_near_um=x_n, Y_near_um=y_n,
                X_far_um=x_f, Y_far_um=y_f,
                slope_X_um_per_mm=slope_X,
                slope_Y_um_per_mm=slope_Y,
                tilt_X_urad=tilt_X,
                tilt_Y_urad=tilt_Y,
                converged=converged,
            )

            if converged:
                self.history.append(result)
                log.info("  iter %d: CONVERGED  tilt_X=%.2f urad, "
                         "tilt_Y=%.2f urad", i, tilt_X, tilt_Y)
                return True

            # Solve M @ d = -slope for d = (dAY, dAX) in urad; damp,
            # then clip each axis to max_correction_per_iter_urad.
            # The clip keeps us in the linear range near the calibration
            # point even if M_inv computes huge values (which happens
            # when one slope direction has weak control authority).
            d = M_inv @ np.array([-slope_X, -slope_Y])
            d_AY_raw = float(d[0]) * c.damping
            d_AX_raw = float(d[1]) * c.damping
            cap = c.max_correction_per_iter_urad
            d_AY = max(-cap, min(cap, d_AY_raw))
            d_AX = max(-cap, min(cap, d_AX_raw))
            clipped = (d_AY != d_AY_raw) or (d_AX != d_AX_raw)
            result.correction_AY_urad = d_AY
            result.correction_AX_urad = d_AX
            self.history.append(result)
            if clipped:
                log.info("  iter %d: tilt_X=%+.2f, tilt_Y=%+.2f urad -> "
                         "dAY=%+.2f urad, dAX=%+.2f urad "
                         "(damped %.2fx, CLIPPED to +/- %.0f urad; raw was "
                         "dAY=%+.2f dAX=%+.2f)",
                         i, tilt_X, tilt_Y, d_AY, d_AX, c.damping, cap,
                         d_AY_raw, d_AX_raw)
            else:
                log.info("  iter %d: tilt_X=%+.2f, tilt_Y=%+.2f urad -> "
                         "dAY=%+.2f urad, dAX=%+.2f urad (damped %.2fx)",
                         i, tilt_X, tilt_Y, d_AY, d_AX, c.damping)

            new_AY = float(caget(PV_TABLE_AY)) + d_AY * deg_per_urad
            new_AX = float(caget(PV_TABLE_AX)) + d_AX * deg_per_urad
            self._gated_move_table_pair(
                new_AY, new_AX,
                f"iteration {i}: apply corrective table tilt")

        # max_iterations exhausted. Decide whether the best state is
        # worth committing to.
        log.warning("did not converge after %d iterations "
                    "(threshold %.1f urad)", c.max_iterations, threshold)
        log.info("best state: iter %d, |tilt|=%.1f urad, "
                 "table.AY=%.6g deg, table.AX=%.6g deg",
                 best_iter, best_tilt_mag, best_AY, best_AX)
        log.info("starting |tilt| was %.1f urad", starting_tilt_mag)
        if best_tilt_mag < starting_tilt_mag:
            improvement_pct = (1.0 - best_tilt_mag / starting_tilt_mag) * 100
            log.info("commit-to-best: moving table to best-state pose "
                     "(%.1f%% improvement vs starting)", improvement_pct)
            if not c.dry_run:
                self._gated_move_table_pair(
                    best_AY, best_AX,
                    f"post-iteration: commit to best pose "
                    f"(iter {best_iter}, |tilt|={best_tilt_mag:.1f} urad)")
            self._best_state_committed = True
        else:
            log.warning("no improvement over starting state "
                        "(best %.1f urad >= start %.1f urad); "
                        "restore path will return table to baseline.",
                        best_tilt_mag, starting_tilt_mag)
            self._best_state_committed = False
        return False

    # ---- orchestrator ----------------------------------------------------

    def run(self) -> bool:
        c = self.config

        self.detect_camera_and_lens()
        self._resolve_convergence_threshold()
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
            self.calibrate_sensitivity()
            if cora:
                cora.append_step("calibrate", vars(self.sensitivity))
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
                # Table AY/AX get restored on every exit path EXCEPT:
                #  - clean convergence (new values are the deliberate
                #    output), OR
                #  - max-iterations-without-convergence but the best
                #    |tilt| beat the starting state -- iterate() has
                #    already moved the table back to the best pose and
                #    set self._best_state_committed = True.
                # On OperatorAbort, exception, or max-iter with no net
                # improvement, table goes back to baseline.
                restore_table = not (converged or self._best_state_committed)
                log.info("restoring pre-procedure state "
                         "(table AY/AX restored: %s)",
                         "yes" if restore_table else
                         "no -- procedure converged, keeping new alignment")
                # Always announce + run restore -- even in dry-run, the
                # acquire_image() calls mutate camera state (TriggerMode,
                # ImageMode, NumImages) and we want those put back.
                # The Z restore is a no-op when no Z move happened.
                confirm_motion(
                    self._snapshot.restore_plan(restore_table=restore_table),
                    step_label="restore: returning camera + Z%s to "
                               "pre-procedure state" %
                               (" + table" if restore_table else ""),
                    dry_run=False,            # restore is not gated by dry-run
                    auto_yes=c.auto_yes,
                    announce_only=(not c.confirm_restore),
                )
                self._snapshot.restore(restore_table=restore_table)
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
                   help="Test step applied to table AY/AX for sensitivity "
                        "matrix discovery. Default: 50 urad. If calibration "
                        "fails with a near-singular determinant, try 100.")
    p.add_argument("--exposure-time", type=float, default=0.2,
                   help="Camera exposure (s). Default: 0.2 (gives a clean "
                        "bright spot on 1.1x lens at typical 2-BM-B flux; "
                        "increase if the centroid signal is weak).")
    p.add_argument("--convergence-urad", type=float, default=None,
                   help="Stop iterating when |tilt_X|, |tilt_Y| are below "
                        "this (urad). DEFAULT: auto-computed from the "
                        "detected lens + binning + dz + rail straightness "
                        "floor x --convergence-safety-margin. Operator "
                        "override wins; a warning is logged if the "
                        "override is below the physical noise floor.")
    p.add_argument("--convergence-safety-margin", type=float, default=1.5,
                   help="Multiplier on the noise floor when auto-computing "
                        "the convergence threshold. Default: 1.5.")
    p.add_argument("--centroid-noise-pix", type=float, default=1.0,
                   help="Assumed centroid-fit noise in pixels, used by "
                        "the auto-threshold calculation. Default: 1.0 "
                        "(typical for COM on a clean spot).")
    p.add_argument("--max-iterations", type=int, default=5)
    p.add_argument("--damping", type=float, default=0.5,
                   help="Damping factor 0 < d <= 1 on the computed iteration "
                        "correction. 1.0 = full correction, 0.5 = half. "
                        "Default: 0.5 (safer in the presence of imperfect "
                        "sensitivity matrix or beyond-linear coupling).")
    p.add_argument("--divergence-threshold", type=float, default=1.5,
                   help="Abort if |slope| grows by more than this factor "
                        "from one iteration to the next. Default: 1.5.")
    p.add_argument("--divergence-cumulative-threshold", type=float,
                   default=2.0,
                   help="Abort if |slope| exceeds the best |slope| seen "
                        "so far by more than this factor. Catches slow-"
                        "bleed divergence that the per-step ratio misses. "
                        "Default: 2.0.")
    p.add_argument("--sensitivity-cond-warn-threshold", type=float,
                   default=5.0,
                   help="Log a warning if the calibrated sensitivity-matrix "
                        "condition number exceeds this. Default: 5.0. "
                        "Doesn't auto-abort -- operator can choose to "
                        "proceed and rely on the divergence guards.")
    p.add_argument("--max-correction-urad", type=float, default=200.0,
                   help="Hard clip on per-iteration correction magnitude "
                        "for each table axis. Default: 200 urad. Keeps "
                        "corrections within the linear range of the "
                        "calibration point when M-inversion computes a "
                        "large value (which happens when the table has "
                        "weak control authority in one slope direction).")
    p.add_argument("--bg-corner-size", type=int, default=100,
                   help="Pixels per side of each of the 4 corner boxes "
                        "used to estimate the background noise floor for "
                        "the centroid threshold. Default: 100. Reduce if "
                        "the spot is large enough to encroach on the "
                        "frame corners.")
    p.add_argument("--bg-sigma-threshold", type=float, default=5.0,
                   help="The centroid binary mask is built by thresholding "
                        "at bg_mean + N*bg_std. Default: 5.0. Smaller N "
                        "includes more of the dim halo (and more noise); "
                        "larger N keeps only clearly-above-background "
                        "pixels.")
    p.add_argument("--camera-pixel-um", type=float, default=3.45,
                   help="Camera SENSOR pixel pitch (um), pre-binning. "
                        "Procedure multiplies by cam1:BinX_RBV at runtime "
                        "to get the effective pixel pitch of the delivered "
                        "image. Default: 3.45 (both Oryx 5MP and 31MP).")
    p.add_argument("--gate-z", action="store_true",
                   help="Also gate Z measurement moves on y/N. Default off: "
                        "Z moves stay within the safety band and only sample "
                        "alignment, so they're announced but not gated. "
                        "Table moves are ALWAYS gated regardless.")
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
    setup_console_logger(level=args.log_level)
    config = Config(
        z_near=args.z_near,
        z_far=args.z_far,
        z_calibration_step_urad=args.calibration_step_urad,
        exposure_time=args.exposure_time,
        convergence_threshold_urad=args.convergence_urad,
        convergence_safety_margin=args.convergence_safety_margin,
        centroid_noise_pix=args.centroid_noise_pix,
        max_iterations=args.max_iterations,
        damping=args.damping,
        divergence_grow_threshold=args.divergence_threshold,
        divergence_cumulative_threshold=args.divergence_cumulative_threshold,
        sensitivity_cond_warn_threshold=args.sensitivity_cond_warn_threshold,
        max_correction_per_iter_urad=args.max_correction_urad,
        bg_corner_size=args.bg_corner_size,
        bg_sigma_threshold=args.bg_sigma_threshold,
        camera_pixel_um=args.camera_pixel_um,
        gate_z=args.gate_z,
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
