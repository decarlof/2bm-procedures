"""Executable procedures for the APS 2-BM imaging beamline.

One file per procedure; the file name matches the corresponding cora
``Procedure`` slug in ``cora/docs/deployments/2-bm/procedures.md``.

Shared primitives (centroid fits, EPICS helpers, optional cora-record
logging, slit composites) live in :mod:`procedures._shared`.

Per-procedure human specs live in ``2bm-docs/source/procedures/``
(rendered at https://docs2bm.readthedocs.io/en/latest/source/procedures.html).
"""

__version__ = "0.0.1"
