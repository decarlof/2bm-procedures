# Changelog

All notable changes to `2bm-procedures` are recorded here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## v0.0.2 — 2026-06-14 — `calibrate_slit_blade_throw` validated; A V miscal diagnosed and fixed

### Summary

Second procedure to reach validated end-to-end status:
`procedures.calibrate_slit_blade_throw`. Drives each L3-style slit
blade motor by ±`blade_throw_mm` from baseline, measures how far
the spot's corresponding edge moves on the detector, and reports
per-blade pixels-per-mm slopes with same-axis spread and V/H ratio
flags.

First real-data application immediately surfaced a quantitative
diagnosis of the A-station V-blade calibration error that had been
an operator workaround for months ("set Vsize = 0.6 mm to get a
square image when Hsize = 1.0 mm"):

- **A station before fix:** H mean **334 px/mm**, V mean **497**,
  **V/H = 1.487** (WARN) — V blades over-scaled by 49%.
- **A station after fix** (live IOC `caput .MRES` on m15 and m16
  from `4e-4` to `5.95e-4` via SET/Use position-redefinition;
  persisted by autosave): H mean 334, V mean 336,
  **V/H = 1.008** (PASS).
- **B station** (no fix needed): H mean 168, V mean 171,
  **V/H = 1.019** (PASS). Same-axis spreads: H 0.5%, V 1.2%.
- All slope ratios consistent with geometric expectations: A
  slits at z=25225 mm, B slits at z=50500 mm, detector at
  z≈50500. A's px/mm is ~2× B's, matching the 2× geometric
  magnification of A's projection at the detector plane.

In addition to the calibration data, the field test surfaced two
real beamline observations worth recording (neither actionable in
this release):

- A's V blade pair has a small mechanical **tilt** — the
  blade-throw procedure measures ~334 px/mm per V blade (matching
  H), but `centre_and_close_slits`'s centring-sensitivity matrix
  measures only 134 px/mm for V vs 330 for H. The discrepancy is
  consistent with a tilted blade-pair shifting the slit
  projection less than 1:1 as the centre moves. Not a calibration
  problem; just a geometric fact about A's V kinematic.
- **DMM image flip:** at A, blade-to-edge mappings are mirrored
  from item_020's labels (H+ outboard moves the LEFT edge of the
  spot; V+ up moves the BOTTOM edge). At B, no flip. The DMM
  Bragg reflection between A and B inverts one axis of A's image
  relative to B's image at the detector. Useful to know for any
  future blade-direction debugging.

### What this procedure does

`calibrate_slit_blade_throw` characterises the per-blade pixel-
per-mm slope of each of the four blade motors in a chosen L3-style
slit station (A or B). For each blade: snapshot baseline, drive
to baseline + `blade_throw_mm` (gated), measure spot edge bbox
via 1D row/column profile half-max crossings; drive to baseline −
`blade_throw_mm` (gated), measure again; restore baseline
(gated). Compute the slope as (Δprimary-edge_pix) / (−2 ×
`blade_throw_mm`). Same-axis-blade agreement and V/H mean ratio
flag mis-calibrated motor records.

Output is **pure diagnostic** — every blade is restored to its
baseline at the end of the run (no deliberate output left in
place). The operator interprets the report and changes IOC
`.MRES` (or .ERES, gear ratio) manually if a miscal is found.

The operator-facing spec is at
[`2bm-docs/procedures/item_012.rst`](https://docs2bm.readthedocs.io/en/latest/source/procedures/item_012.html);
that page is the source of truth for parameters, predicates,
preconditions, and failure modes. This CHANGELOG documents the
implementation history.

### Architecture

- **One blade at a time**. The four blade motor records (e.g.
  `2bma:m13.VAL` for A's H+ blade) are driven directly, NOT via
  the virtual `Slit*Hsize` / `Vcenter` ao records. The slope
  reported is the raw motor-mm calibration, independent of the
  slit calc.
- **Per-motion confirmation gate** on every blade move (3 gates
  per blade × 4 blades = 12 gates per run).
- **Snapshot + restore** for all four blade baselines plus full
  camera state. `try/finally` always restores — this is a
  measurement, not a state-change procedure.
- **Profile-based edge detection.** 1D row/column-averaged
  profiles of the ROI, threshold at `p10 + 0.5 × (p90 − p10)`
  with subpixel linear interpolation between bracketing samples.
  Halo is < 50% of plateau so half-max naturally clips it;
  profile averaging suppresses multilayer-stripe noise; subpixel
  interpolation gives ~0.1 px slope precision on a single frame.
- **Auto-detection** of camera and lens via MCTOptics
  `CameraSelect` / `LensSelect`, same pattern as
  `detector_z_rail_alignment`.
- **Upstream-aperture guardrail** (added in this version): on
  startup, projects any upstream slit's aperture forward to the
  target station's plane and `log.warning`s if it's too small to
  contain the worst-case blade extent during the procedure. Pure
  log; no caputs or flow change. For B, the upstream is A. For A,
  no upstream slit and the check is a no-op.
- **ANSI-coloured console logging**, same `_shared/log.py` as
  `detector_z_rail_alignment`.

### cora-process mapping

| 2bm-procedures (v0.0.2)              | cora aggregate |
|--------------------------------------|----------------|
| `calibrate_slit_blade_throw.py`      | `Procedure` body |
| `SLIT_STATIONS[A|B]` dict            | `Procedure.target_assets` (per-station `Slit` Asset) |
| `BladeCalibrationResult`             | one `procedure_step` per blade with `slope_pix_per_mm` payload |
| Per-blade result `slope_pix_per_mm`  | new field `calibration_slope_pix_per_mm` on each blade-motor Asset (4 per station) |
| `_Snapshot.restore()`                | `Procedure.rollback` Method (always-restore variant) |
| `_shared/cora_log.py`                | REST client to the audit-spine endpoint |
| Run outcome                          | `Procedure.outcome` (`complete` / `truncate` / `abort`) |

### Bugs fixed during v0.0.2 development

- **A-station blade PV list was wrong.** `SLIT_STATIONS["A"]`
  carried `("2bma:m1", "m2", "m3", "m4")` as a stale TODO copied
  forward from an earlier draft — those PVs are M1 mirror motors,
  not slit blades. Caught at the first gate prompt on the first
  real-data run (no caput happened). Per `item_020`, correct
  A-station blades are `m13` (H+ outboard), `m14` (H− inboard),
  `m15` (V+ up), `m16` (V− down). Fixed in `c08b6ab`; same fix
  applied to `centre_and_close_slits.py`, which had the same
  wrong list and had been silently polling the mirror-motor
  `.DMOV` for motion-done detection during A-station centring
  runs.
- **Bbox detector collapsed to ROI on multilayer-stripe images.**
  Initial implementation used "mask `>` (bg_median + 5 × MAD_σ)
  then take 2D bounding box". The slit-defined bright square at
  the detector has a faint scatter halo extending well outside the
  geometric aperture; on the noisy multilayer-stripe background,
  the halo exceeded the MAD-derived threshold everywhere in the
  400-px-half-side ROI. Every blade reported the bbox edges as
  exactly the ROI edges; every reported displacement was zero.
  Replaced with 1D row/column-averaged profiles + half-max
  crossings with subpixel linear interpolation between bracketing
  samples. Halo is < 50% of plateau so half-max clips it cleanly.
  Synthetic-data unit test confirms 99.5 / 399.5 crossings on a
  100-400 plateau. Fixed in `91ab863`.

### Operator-side findings (not code changes)

- **A V blades miscal root cause:** the IOC substitution file
  `/net/s2dserv/xorApps/epics/synApps_5_8/ioc/2bma/iocBoot/ioc2bma/motor.substitutions`
  uses a single template for **every** motor m1..m44+ — all
  share `MRES=2.5e-4`, `EGU="degrees"`, `VELO=1`. The per-axis
  mechanical differences (lead-screw pitch, gear ratio) between
  the A H blade assembly and the A V blade assembly were never
  compensated. Live autosave had already raised MRES to `4e-4`
  for these blades, but the per-axis differentiation was still
  missing. Other motors on the crate are very likely similarly
  mis-calibrated — this procedure provides the tooling to
  characterise any of them.
- **A V tilt** (see Summary). Confirmed by H/V centring-
  sensitivity asymmetry surviving the MRES fix.
- **DMM image flip** between A and B (see Summary).

### CLI surface (v0.0.2)

```
python -m procedures.calibrate_slit_blade_throw [OPTIONS]

Geometry / slit:
  --slit-station                    "A" or "B"
  --blade-throw-mm                  ± move per blade from baseline (default 0.5)
  --edge-roi-half-size-pix          ROI half-side around spot centre (default 400)

Edge detection:
  --aperture-edge-level             Half-max fraction of (p90-p10) (default 0.5)
  --aperture-min-dynamic-range      Min profile range for visible spot (default 100)
  --threshold-fraction              COM threshold (initial centroid) (default 0.5)

Camera:
  --exposure-time                   Per-frame exposure s (default 0.2)
  --frames-per-measurement          Average N frames per bbox (default 1)
  --camera-pixel-um                 Sensor pitch um (default 3.45)

Confirmation / safety:
  --yes                             Auto-confirm every prompt
  --confirm-restore                 Also gate the restore path
  --dry-run                         Plan + skip every motion

Logging / observability:
  --log-level                       DEBUG | INFO | WARNING | ERROR (default INFO)
  --no-cora-log                     Skip cora Procedure record
```

### Open follow-ups (will inform later versions)

- **cora wiring**. Extend the per-station `Slit` Asset with four
  `calibration_slope_pix_per_mm` fields (one per blade motor)
  populated from this procedure's report; register the procedure
  in `cora/docs/deployments/2-bm/procedures.md`.
- **Characterise the rest of the OMS VME58 crate.** Many other
  motors share the same MRES template default; almost certainly
  most have un-compensated per-axis mechanical differences. A
  follow-up procedure could iterate this calibration over an
  operator-supplied motor list and produce a summary table.
- **Optional `--prepare` flag.** Open upstream slits + set target
  to 1×1 mm before the calibration, then restore on exit. Convenience
  one-button operation. Current workflow asks the operator to do
  this manually (one or two caputs).
- **Per-station `z_mm` should live in `item_020.rst` rather than
  hardcoded in `SLIT_STATIONS`.** Once `2bm-docs` exposes a
  machine-readable form of the layout map, source the projection
  geometry from there.

### Acknowledgements

The bbox-failure debugging was steered by the operator on the
floor — without their willingness to paste full run logs after
each algorithm iteration the half-max-profile fix would have
taken much longer to converge on. The downstream "set Vsize = 0.6
when Hsize = 1.0" trick that motivated the V calibration analysis
was the first hint that the IOC was wrong, not the operator.

---

## v0.0.1 — 2026-06-14 — first end-to-end convergence on 2-BM-B

### Summary

First version of `procedures.detector_z_rail_alignment` that runs
**end-to-end on the live 2-BM-B beamline** and converges. Field test
(2026-06-14, 14:17 → 14:48 CDT):

- Camera: FLIR Oryx 31MP (`2bmSP2:`) via MCTOptics
- Lens: 1.1× (slot 0)
- Z safety band: 200–500 mm
- Starting misalignment: **|tilt| = 431 µrad** (`tilt_X=-169, tilt_Y=+397`)
- After 5 iterations: **|tilt| = 26 µrad** (`tilt_X=-10, tilt_Y=+24`)
- Sensitivity-matrix condition number: **1.1** (essentially diagonal)
- Wall time: ~30 minutes (dominated by operator y/N confirmations
  and Aerotech Z moves)

Convergence trajectory was textbook geometric decay (each iteration
~0.5× the previous |tilt|, matching `damping=0.5`). The final
residual sits at the PRO225SL-1000 rail's intrinsic straightness
floor (~10–20 µrad over a 300 mm sub-range), so further iteration
would not improve the physical alignment.

### What this procedure does

`detector_z_rail_alignment` aligns the Optique Peter detector Z
rail (`2bmbAERO:m1`) to the X-ray beam by walking the detector
along Z with a small square aperture defined by the B-station
slits, fitting the centroid drift, and tilting the detector
optical table (`2bmb:table3.AY` / `.AX`) to rotate the rail back
parallel to the beam. Removes the **linear** rail-to-beam
misalignment; the non-linear residual is the rail's intrinsic
straightness, which is not correctable from the table.

The operator-facing spec is at
[`2bm-docs/procedures/item_002.rst`](https://docs2bm.readthedocs.io/en/latest/source/procedures/item_002.html)
and that page is the source of truth for parameters, predicates,
preconditions, and failure modes. This CHANGELOG documents the
implementation history.

### Architecture (the model the code implements)

- **Calibration → Iteration**. Calibration measures a 2×2
  **slope-sensitivity matrix M** (Δslope_X/slope_Y per µrad of
  AY/AX). Iteration solves `M @ (ΔAY, ΔAX) = −(slope_X, slope_Y)`
  via `numpy.linalg.inv`, damps by 0.5×, clips per axis to
  `max_correction_per_iter_urad`, applies through the soft PVs
  with motion-done by polling the six underlying jacks' `.DMOV`.
- **Auto-detection**. Camera prefix and lens magnification are
  read from MCTOptics (`2bm:MCTOptics:CameraSelect` /
  `LensSelect`) keyed by enum index; binning is read from
  `cam1:BinX_RBV`. Convergence threshold is auto-set to
  `noise_floor × safety_margin` where noise floor =
  `max(measurement_noise(lens, bin, dz), rail_straightness_floor)`.
- **Snapshot + restore**. At entry the procedure snapshots the
  active camera's full state (9 PVs), the Z stage RBV, and the
  table soft PVs (`AY`, `AX`). `try/finally` runs restore on
  every exit path. Restore order is **table → Z → camera**
  (highest-stakes first; if restore itself is interrupted, the
  most critical PVs are already back). Table is restored only on
  non-success exits (`OperatorAbort`, exception, max-iter without
  improvement); clean convergence keeps the new AY/AX as the
  procedure's deliberate output, and max-iter-with-improvement
  commits to the best-seen pose.
- **Per-motion confirmation gate**. Every **table** move
  (calibration perturb, calibration restore, iteration
  correction) prints a `pv | current | target | delta | units`
  plan block and waits for `y/N`. Z measurement moves are
  announced-only by default (safety-band-protected, only sample
  alignment); operator can flag `--gate-z` to gate them too.
  `N` raises `OperatorAbort` → `try/finally` runs restore.
- **Divergence guards** (two of them):
  - Per-iteration ratio (default 1.5×) — catches a single
    over-large step.
  - Cumulative-vs-best ratio (default 2.0×) — catches slow-bleed
    divergence the per-step ratio misses (a 1.2×/iter slide over
    9 iters is a 4.3× cumulative blow-up that the per-step
    check sleeps through).
- **Centroid algorithm**: intensity-weighted COM
  (`center_of_mass`, `threshold_fraction = 0.5`). An alternative
  background-thresholded geometric centroid
  (`centroid_above_background`) is available via
  `--centroid-algorithm binmask` for image cases where saturated
  features outside the spot bias COM. COM was empirically more
  accurate on this beamline's DMM frames (8 px from operator
  hand-eyeballed centre, vs 30-40 px for `binmask`); kept as the
  default.
- **Z safety band** [200, 500] mm enforced at `__init__`; motor
  `.HLM` / `.LLM` not modified.
- **ANSI-coloured console logging** modeled on the `energy` package.

### cora-process mapping

This procedure is intended to become a first-class
[`cora`](https://github.com/xray-imaging/cora) `Procedure`
aggregate. The shape of v0.0.1 anticipates that mapping:

| 2bm-procedures (v0.0.1)              | cora aggregate |
|--------------------------------------|----------------|
| `detector_z_rail_alignment.py`       | `Procedure` body |
| Module-level `PRECONDITIONS` list    | `Procedure.preconditions` |
| Each iteration step                  | `Method` invocations (recorded as `procedure_step`) |
| `_Snapshot.restore()`                | `Procedure.rollback` Method |
| `_shared/cora_log.py`                | REST client to the audit-spine endpoint |
| Convergence outcome (true/false/best-committed) | `Procedure.outcome` (`complete` / `truncate` / `abort`) |

The `PRECONDITIONS` list is machine-readable today (11 entries,
8 satisfied by stub procedures `item_003`–`item_010`, 3
satisfied by operator action). When cora's schema gains a
`preconditions` field, the migration is one `json.dumps` call.

The `cora_log.py` shim already opens / appends-step / closes a
local `Procedure` record on every run; pointing it at a real
cora REST endpoint is a few lines once that endpoint stabilises.

### CLI surface (v0.0.1)

The procedure's full parameter set (from `--help`):

```
python -m procedures.detector_z_rail_alignment [OPTIONS]

Geometry:
  --z-near, --z-far                 Z anchors, in [200, 500] mm
  --calibration-step-urad           Test perturbation for M (default 50)
  --max-correction-urad             Per-iter correction clip (default 200)

Convergence:
  --max-iterations                  Iteration cap (default 5)
  --convergence-urad                Stop when both |tilt_*| below (default: auto)
  --convergence-safety-margin       Multiplier on noise floor (default 1.5)
  --centroid-noise-pix              For auto-threshold calc (default 1.0)
  --damping                         Damping factor (default 0.5)
  --divergence-threshold            Per-iter |slope| growth limit (default 1.5)
  --divergence-cumulative-threshold |slope| vs best limit (default 2.0)
  --sensitivity-cond-warn-threshold cond(M) warning threshold (default 5.0)

Camera + centroid:
  --exposure-time                   Per-frame exposure s (default 0.2)
  --camera-pixel-um                 Sensor pitch um (default 3.45)
  --centroid-algorithm              "com" | "binmask" (default "com")
  --threshold-fraction              (com only) fraction of max (default 0.5)
  --bg-corner-size                  (binmask only) corner sample box (default 100)
  --bg-sigma-threshold              (binmask only) N sigma above bg (default 5.0)
  --frames-per-measurement          Average N frames per centroid (default 1)

Confirmation / safety:
  --yes                             Auto-confirm every prompt
  --gate-z                          Also gate Z measurement moves
  --confirm-restore                 Also gate the restore path
  --dry-run                         Plan + skip every motion

Logging / observability:
  --log-level                       DEBUG | INFO | WARNING | ERROR (default INFO)
  --no-cora-log                     Skip cora Procedure record
```

### Bugs fixed during v0.0.1 development

Mining the commit log for the failure modes the field test
surfaced:

- `acquire_image()`: applied `np.mod(img, 4096)` for Mono12Packed
  cargo-culted from `align-main`'s `take_image()`. The 2-BM Oryx
  31MP via MCTOptics delivers values left-shifted into uint16
  (4095 → 65520), not raw 12-bit; the modulo collapsed the
  saturated pixels down to ~3120 and completely reshuffled
  which pixels were brightest, breaking COM. **This was the big
  one** — every prior calibration attempt was running on a
  corrupted image. Fixed in `0cfca9e`.
- `caput_wait()`: motor moves were `caput(wait=False)` + DMOV
  poll, which has a race where `caget` sees the OLD `DMOV=1`
  before the record sets `DMOV=0` to indicate motion in
  progress, so the function returned before motion had even
  started. Switched to put-callback (`caput(wait=True)`) with a
  belt-and-suspenders DMOV verification. Fixed in `7a484a3`.
- Restore stranded the table on second-Ctrl-C. Operator hit
  Ctrl-C during iteration → finally ran restore → restore
  started camera state (fast) then Z (slow); operator hit Ctrl-C
  again thinking restore was hung; KeyboardInterrupt aborted
  `safe_restore` before reaching the table entries. Two fixes:
  reorder restore so **table goes first**; make `safe_restore`
  catch KeyboardInterrupt per-action (continues; requires 3
  consecutive Ctrl-Cs to abort restore). Fixed in `06b0696`.
- Sensitivity matrix used to be a centroid-shift Jacobian at
  fixed Z (`J_AY_X = ΔX_centroid_at_z_far / ΔAY`), which is
  geometry-dependent and doesn't directly drive slope
  correction — uniform centroid shifts cancel between
  `z_near` / `z_far` and leave slope unchanged. Replaced with a
  proper slope-sensitivity matrix M (`Δslope_X / ΔAY`). The
  iteration used to use only `sign(J)` instead of the full M
  inverse, applying corrections proportional to slope rather
  than properly inverted. Fixed in `9305e09`.
- Slow-bleed divergence (each iter 1.2× the previous, well
  under the 1.5× per-step guard, but 4.3× cumulative over 9
  iterations). Added cumulative-vs-best divergence guard. Fixed
  in `06b0696`.

### Open follow-ups (will inform later versions)

- **Per-iteration M re-calibration**. M is currently calibrated
  once at run start. The deliberate-tilt sanity test (M at
  baseline-pose vs M at deliberately-perturbed-pose) showed M is
  pose-dependent — at small misalignments and small table-pose
  shifts, M holds; at larger excursions, M can shift sign on
  individual elements. Re-calibrating M every K iterations would
  handle the non-linear regime at the cost of K× cal time.
- **ROI mask**. When the image contains bright features outside
  the actual spot (off-axis multilayer stripes, hot pixels), COM
  can be biased. v0.0.1 ships with the binmask alternative and
  the procedure tolerates a fixed positional bias (it cancels in
  the slope), but a seeded ROI (operator points at expected spot
  location with `--roi cx cy size`) would clean this up. Scope
  deferred until we see an image case the current centroid choice
  can't handle.
- **Frame averaging on by default for difficult lighting**.
  `--frames-per-measurement N` is implemented; defaults to 1.
  Worth bumping to 4 for routine use once we measure how much it
  helps the M condition number in practice.
- **Persistent cora REST integration**. `cora_log.py` is a
  no-op shim; ready to swap for an `httpx` POST chain to the
  cora `/procedures` endpoint when that endpoint lands.
- **Verification-only mode**. `--max-iterations 0` currently
  errors (loop bound). Should be valid as a "measure current
  slope and report, no calibration, no motion" check after the
  procedure runs.
- **The 8 stub procedures** (`item_003`–`item_010` in
  `2bm-docs`) need real bodies. They're the precondition graph
  for `detector_z_rail_alignment`; each is a separate alignment
  / configuration step.

### Acknowledgements

The reference for the camera-access pattern came from Francesco
De Carlo's `align-main` repository (notably the structure of
`detector.take_image()`). The colored-logging module is adapted
from the [`energy`](https://github.com/xray-imaging/energy)
package's `log.py`.
