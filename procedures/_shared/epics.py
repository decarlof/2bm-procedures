"""Thin PyEpics helpers used across procedures.

Kept deliberately small. Each helper wraps one common pattern with a
timeout and a sensible error message so the procedure modules don't
each re-invent the boilerplate.
"""

from __future__ import annotations

import logging
import time
from typing import Callable, Sequence

import numpy as np
from epics import caget, caput


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Operator abort + motion confirmation gate
# ---------------------------------------------------------------------------

class OperatorAbort(RuntimeError):
    """Raised when the operator declines a motion at ``confirm_motion``.

    The procedure's ``run()`` method should let this propagate to its
    ``try / finally`` so the snapshot-restore path still executes.
    """


def confirm_motion(plan: Sequence[dict], *,
                   step_label: str = "",
                   dry_run: bool = False,
                   auto_yes: bool = False,
                   announce_only: bool = False) -> bool:
    """Print a planned-motion block and (in interactive mode) gate on
    operator approval.

    ``plan`` is a list of dicts:

        {"pv": "2bmb:table3.AY", "current": 0.000134,
         "target": 0.001234, "units": "deg"}

    Returns ``True`` if the caller should proceed with motion, ``False``
    if dry-run should skip. Raises ``OperatorAbort`` on ``N`` reply.

    ``announce_only`` prints the plan but never prompts — used by the
    restore path so a panic exit never blocks on stdin.
    """
    if step_label:
        print(f"\n{step_label}")
    print()
    header = ("   {:<32}{:>14}  {:>14}  {:>14}  {}"
              .format("PV", "current", "target", "delta", "units"))
    print(header)
    print("   " + "-" * (len(header) - 3))
    for step in plan:
        pv = step.get("pv", "?")
        units = step.get("units", "")
        current = step.get("current")
        target = step.get("target")

        def _fmt(v):
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                return f"{v:+14.6g}"
            return f"{str(v):>14}"

        try:
            delta = float(target) - float(current)
            delta_s = f"{delta:+14.6g}"
        except (TypeError, ValueError):
            delta_s = " " * 14
        print(f"   {pv:<32}{_fmt(current)}  {_fmt(target)}  {delta_s}  {units}")
    print()

    if dry_run:
        print("   [dry-run] skipping motion")
        return False
    if announce_only:
        return True
    if auto_yes:
        print("   [--yes] auto-confirmed")
        return True

    try:
        reply = input("   Proceed? [y/N]: ").strip().lower()
    except EOFError:
        reply = ""
    if reply != "y":
        raise OperatorAbort("operator declined motion at confirmation gate")
    return True


# ---------------------------------------------------------------------------
# Basic ca operations with timeout + value-readback verification
# ---------------------------------------------------------------------------

def caput_wait(pvname: str, value, dmov_pvname: str | None = None,
               timeout: float = 30.0):
    """``caput`` + wait for completion.

    Always uses put-callback (``ca_put(wait=True)``). For a motor
    record's ``.VAL`` field the put-callback fires when the record
    completes its processing chain -- which for the motor record is
    DMOV=1 (motion done). This is the correct way to wait for motor
    motion; the previous "caput + poll DMOV" approach had a race
    where caget could see the *old* DMOV=1 before the motor record
    had set DMOV=0 to indicate motion in progress, causing the
    function to return before motion had even started.

    If ``dmov_pvname`` is given, a belt-and-suspenders DMOV=1 check
    is performed after the put-callback returns -- if DMOV is still
    0 within 5 s of put-callback completion, ``TimeoutError`` is
    raised.
    """
    log.debug("caput %s = %s", pvname, value)
    caput(pvname, value, wait=True, timeout=timeout)
    if dmov_pvname is None:
        return
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if int(caget(dmov_pvname) or 0) == 1:
            return
        time.sleep(0.05)
    raise TimeoutError(
        f"after put-callback {pvname}={value} completed, "
        f"{dmov_pvname} is still 0 -- motor not idle?"
    )


def cawait_value(pvname: str, target, timeout: float = 30.0,
                 tolerance: float = 0.0):
    """Block until ``pvname`` reads ``target`` (numeric within
    ``tolerance``, or exact match for bool / string), else
    ``TimeoutError``."""
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
        f"{pvname} did not reach {target!r} (+/-{tolerance}) within {timeout} s"
    )


# ---------------------------------------------------------------------------
# Motor convenience: move + wait
# ---------------------------------------------------------------------------

def move_motor(motor_prefix: str, position: float, timeout: float = 60.0):
    """Move a standard EPICS motor record to ``position`` and wait for
    ``<motor>.DMOV == 1``. ``motor_prefix`` is the motor base PV (e.g.
    ``2bmbAERO:m1``); ``.VAL`` and ``.DMOV`` are appended internally."""
    caput_wait(f"{motor_prefix}.VAL", position,
               dmov_pvname=f"{motor_prefix}.DMOV", timeout=timeout)


# ---------------------------------------------------------------------------
# Table soft-PV move (synApps ``table.db`` composite axes)
# ---------------------------------------------------------------------------

def move_table_axis(soft_pv: str, value: float,
                    jack_motor_prefixes: Sequence[str],
                    timeout: float = 60.0) -> None:
    """Write a ``table.db`` soft PV (e.g. ``2bmb:table3.AY``) and wait
    for all underlying jack motors to settle.

    The soft PV is not a motor record itself — its ``.DMOV`` doesn't
    exist — so motion-done is detected by ANDing the ``.DMOV`` of every
    underlying jack the table aggregates.

    Parameters
    ----------
    soft_pv
        e.g. ``"2bmb:table3.AY"`` (literal dot, not a field separator).
    value
        Target value in the units defined by ``table.db`` (typically
        degrees for ``.AX/.AY/.AZ``, mm for ``.X/.Y/.Z``).
    jack_motor_prefixes
        Motor base PVs the table aggregates, e.g.
        ``("2bmb:m9", "2bmb:m10", "2bmb:m11",
           "2bmb:m12", "2bmb:m13", "2bmb:m14")`` for ``2bmb:table3``.
        ``.DMOV`` is appended to each internally.
    """
    log.debug("table soft-move %s = %s", soft_pv, value)
    dmov_pvs = [f"{prefix}.DMOV" for prefix in jack_motor_prefixes]
    caput(soft_pv, value)
    # Phase 1: wait up to 2 s for ANY jack DMOV to drop to 0
    # (motion started). If none drops the kinematic engine probably
    # computed a sub-resolution change and no jack moved; treat as
    # a no-op success.
    grace_deadline = time.monotonic() + 2.0
    motion_started = False
    while time.monotonic() < grace_deadline:
        if any(int(caget(p) or 0) == 0 for p in dmov_pvs):
            motion_started = True
            break
        time.sleep(0.05)
    if not motion_started:
        log.debug("table %s = %s: no jack DMOV dropped within 2 s "
                  "(sub-resolution / no-op)", soft_pv, value)
        return
    # Phase 2: wait for all jack DMOVs to come back to 1 (all done).
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if all(int(caget(p) or 0) == 1 for p in dmov_pvs):
            return
        time.sleep(0.1)
    not_done = [p for p in dmov_pvs if int(caget(p) or 0) != 1]
    raise TimeoutError(
        f"table {soft_pv} -> {value}: jacks did not reach DMOV=1 within "
        f"{timeout} s (still moving: {not_done})"
    )


# ---------------------------------------------------------------------------
# areaDetector image fetch
# ---------------------------------------------------------------------------

def acquire_image(cam_prefix: str, image_prefix: str | None = None,
                  exposure_time: float | None = None,
                  timeout: float = 30.0) -> np.ndarray:
    """Trigger one acquisition on an areaDetector camera and return the
    frame as a 2-D ``uint16`` numpy array.

    Modeled on the align-main ``take_image()`` reference pattern:

    * Explicitly stops any in-progress acquire so the cam settings
      stick (Continuous mode is the operator's default for tomoscan).
    * Forces ``TriggerMode=Off`` and ``NumImages=1`` before the trigger
      so a stale external-trigger configuration can't make the call
      hang waiting for a PSO pulse that never comes.
    * Reads frame dimensions from the camera plugin
      (``cam1:SizeX_RBV / SizeY_RBV``) — matches the reference and
      survives image-plugin misconfig.
    * Applies a bit-depth wraparound (``np.mod(img, 2**N)``) per
      ``cam1:PixelFormat_RBV`` to clean any signed-overflow values
      from the buffer transfer (Mono8 -> 8, Mono12* -> 12, Mono16 -> 16).

    The procedure is responsible for snapshotting + restoring the
    operator's pre-procedure camera state — this helper only mutates
    what's strictly needed for the single-shot acquire.

    Parameters
    ----------
    cam_prefix
        Camera areaDetector prefix including the trailing colon, e.g.
        ``"2bmSP2:"`` for the Oryx 31MP.
    image_prefix
        Image plugin prefix; defaults to ``<cam_prefix>image1:``.
    exposure_time
        If given, set ``cam1:AcquireTime`` before triggering. Otherwise
        leave the existing value alone.
    """
    image_prefix = image_prefix or f"{cam_prefix}image1:"

    if int(caget(f"{cam_prefix}cam1:Acquire") or 0) == 1:
        caput(f"{cam_prefix}cam1:Acquire", 0, wait=True, timeout=5.0)
        cawait_value(f"{cam_prefix}cam1:Acquire", 0, timeout=5.0)

    if exposure_time is not None:
        caput(f"{cam_prefix}cam1:AcquireTime", exposure_time, wait=True)
    caput(f"{cam_prefix}cam1:TriggerMode", "Off", wait=True)
    caput(f"{cam_prefix}cam1:ImageMode", "Single", wait=True)
    caput(f"{cam_prefix}cam1:NumImages", 1, wait=True)

    caput(f"{cam_prefix}cam1:Acquire", 1, wait=True, timeout=timeout)
    cawait_value(f"{cam_prefix}cam1:Acquire", 0, timeout=timeout)

    width = int(caget(f"{cam_prefix}cam1:SizeX_RBV"))
    height = int(caget(f"{cam_prefix}cam1:SizeY_RBV"))
    arr = caget(f"{image_prefix}ArrayData", count=width * height)
    if arr is None:
        raise RuntimeError(f"image fetch returned None from {image_prefix}ArrayData")
    img = np.asarray(arr).reshape((height, width))

    pixel_format = caget(f"{cam_prefix}cam1:PixelFormat_RBV",
                         as_string=True) or "Mono8"
    if pixel_format.startswith("Mono16"):
        bits = 16
    elif pixel_format.startswith("Mono12"):
        bits = 12
    else:
        bits = 8
    img = np.mod(img.astype("int32"), 2**bits).astype("uint16")
    return img


# ---------------------------------------------------------------------------
# Run a list of restore actions, swallowing per-action exceptions so a
# single failure doesn't block the rest of the restore.
# ---------------------------------------------------------------------------

def safe_restore(actions: Sequence[tuple[str, Callable[[], None]]]) -> None:
    """Execute each ``(label, callable)`` in order. If any raises, log a
    warning and continue. Used by procedure ``_Snapshot.restore()``
    methods so one bad PV doesn't block restoring the others.

    ``KeyboardInterrupt`` is caught per-action too -- the design
    contract is that restore runs to completion regardless of how the
    procedure aborted, including a panic-Ctrl-C from the operator
    during the restore itself. The operator can still interrupt a
    truly-hung action by hitting Ctrl-C three times in a row (catches
    accumulate; eventually KeyboardInterrupt propagates out).
    """
    interrupt_count = 0
    for i, (label, fn) in enumerate(actions):
        try:
            fn()
            interrupt_count = 0  # reset on any successful action
        except KeyboardInterrupt:
            interrupt_count += 1
            remaining = len(actions) - i - 1
            log.warning("restore %s interrupted by Ctrl-C; continuing "
                        "with remaining %d action(s) (Ctrl-C %d/3 -- "
                        "press 3 times in a row to abort restore)",
                        label, remaining, interrupt_count)
            if interrupt_count >= 3:
                log.error("restore aborted after 3 consecutive Ctrl-C; "
                          "remaining actions skipped; PVs may be in "
                          "inconsistent state -- restore manually")
                raise
        except Exception as exc:
            log.warning("restore %s failed: %s", label, exc)
