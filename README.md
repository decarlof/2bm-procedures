# 2bm-procedures

Executable Python procedures for the APS 2-BM imaging beamline:
alignments, baselines, recoveries, calibrations, and other episodic
operational tasks.

**Current release: [v0.0.1](CHANGELOG.md) (2026-06-14)** —
`detector_z_rail_alignment` converges end-to-end on 2-BM-B
(|tilt| 431 → 26 µrad in 5 iterations, M condition number 1.1).
See [CHANGELOG.md](CHANGELOG.md) for the architecture, the
cora-process mapping, the bug fixes that got us here, and the
open follow-ups.

Each procedure is **one file** in [`procedures/`](procedures/), named
to match the corresponding [cora](https://github.com/xray-imaging/cora)
`Procedure` slug. The human specification for each procedure lives
in [`2bm-docs`](https://docs2bm.readthedocs.io/en/latest/source/procedures.html)
under `procedures/`; this repo carries the executable implementation.

## Why this lives outside `tomoscan`

`tomoscan` is shared across many beamlines (2-BM, 7-BM, 13-BM,
32-ID, 2-ID, 6-BM, plus streaming variants) and is intentionally
beamline-agnostic. 2-BM-specific operational procedures don't
belong upstream there — they belong here.

## Layout

```
2bm-procedures/
├── procedures/
│   ├── __init__.py
│   ├── _shared/                          # cross-procedure primitives
│   │   ├── centroid.py                   # COM + background-thresholded geometric centroid
│   │   ├── epics.py                      # PyEpics helpers, motion gates, restore primitives
│   │   ├── log.py                        # ANSI-coloured console logger
│   │   ├── cora_log.py                   # optional cora Procedure-record audit-spine hook
│   │   └── slits.py                      # B-station slit composite helpers
│   ├── detector_z_rail_alignment.py      # v0.0.1: field-tested on 2-BM-B
│   └── ...                               # one file per cora Procedure
├── CHANGELOG.md
└── tests/
```

## Invocation

Each procedure file is runnable as a module:

```bash
python -m procedures.detector_z_rail_alignment --z-near 50 --z-far 350
```

Operator-facing parameters are surfaced via `argparse`; defaults
match the values documented on the corresponding `2bm-docs`
procedure page.

## Relationship to cora

Each procedure here corresponds to one entry in
[`cora/docs/deployments/2-bm/procedures.md`](https://github.com/xmap/cora/blob/main/docs/deployments/2-bm/procedures.md).
The Python implementation is the executable body; cora is the
*audit spine* — every procedure run optionally opens a cora
`Procedure` record at start, appends per-step records as it goes,
and closes it (complete / abort / truncate) at end. The cora
integration is in [`procedures/_shared/cora_log.py`](procedures/_shared/cora_log.py);
when the cora server is unreachable, procedures still run, they
just don't log.

See [`docs2bm.readthedocs.io/.../procedures.html`](https://docs2bm.readthedocs.io/en/latest/source/procedures.html)
for the human walkthroughs.

## License

Apache-2.0; see [`LICENSE`](LICENSE).
