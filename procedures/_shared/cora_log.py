"""Optional cora audit-spine integration.

When the cora server is reachable, a procedure can open a
``Procedure`` record at start, append per-step records as it walks,
and close it (complete / abort / truncate) at end. When cora is not
reachable (or this module's optional dependency is not installed),
the helpers are no-ops and the procedure still runs.

The cora server URL is read from the ``CORA_URL`` environment
variable (default: ``http://localhost:8000``); the procedure slug is
the file name of the calling procedure module without the ``.py``.
"""

# TODO: thin httpx client for the cora /procedures endpoints
# TODO: open_procedure(slug, target_asset_ids, parameters) -> procedure_id
# TODO: append_step(procedure_id, step_index, payload)
# TODO: close_procedure(procedure_id, outcome="complete" | "abort" | "truncate")
