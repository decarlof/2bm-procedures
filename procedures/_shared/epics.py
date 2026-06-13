"""Thin PyEpics helpers used across procedures.

Kept minimal: caget / caput with timeout, cawait for ``.DMOV`` and
similar bool flags. Procedure modules import these directly rather
than each one re-implementing the same boilerplate.
"""

# TODO: wrap epics.caget / caput / camonitor with sensible timeouts
# and error handling for the recovery patterns the procedures expect.
