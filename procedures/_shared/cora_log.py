"""Optional cora audit-spine integration.

A procedure can optionally open a cora ``Procedure`` record at start,
append per-step records as it walks, and close it at end. When the
cora server is not reachable (or this module's optional ``httpx``
dependency is not installed), the helpers degrade to stdout logging
and the procedure runs unchanged.

cora server URL is read from the ``CORA_URL`` environment variable
(default: ``http://localhost:8000``). The cora REST endpoints are
not yet stable — this module ships as a no-op shim that the cora
team can populate against the real API when it lands.
"""

from __future__ import annotations

import logging
import os
import time
import uuid

log = logging.getLogger(__name__)


class CoraProcedureLog:
    """Procedure-log handle. Use as a context manager or call
    ``open()``/``append_step()``/``close()`` directly.

    The class always succeeds (no exceptions if cora is down or
    httpx is missing). Disconnect handling is intentionally soft —
    losing the audit trail must not bring down a procedure.
    """

    def __init__(self, slug: str, target_asset_ids: list[str] | None = None,
                 parameters: dict | None = None,
                 server_url: str | None = None) -> None:
        self.slug = slug
        self.target_asset_ids = list(target_asset_ids or [])
        self.parameters = dict(parameters or {})
        self.server_url = server_url or os.environ.get("CORA_URL", "http://localhost:8000")
        self.procedure_id: str | None = None
        self._step_index = 0
        self._client_ok = False

    def open(self) -> "CoraProcedureLog":
        """Try to register a Procedure on the cora server. Soft-fails
        to stdout if cora isn't reachable."""
        try:
            import httpx  # noqa: F401  (optional dep; real client lands here)
            # TODO: POST /procedures, capture returned id
            self._client_ok = True
        except ImportError:
            self._client_ok = False
        self.procedure_id = self.procedure_id or f"local:{uuid.uuid4()}"
        log.info("[cora_log] open procedure=%s id=%s targets=%s",
                 self.slug, self.procedure_id, self.target_asset_ids)
        return self

    def append_step(self, kind: str, payload: dict | None = None) -> None:
        self._step_index += 1
        log.info("[cora_log] step %d kind=%s payload=%s",
                 self._step_index, kind, payload or {})
        # TODO: POST /procedures/<id>/steps

    def close(self, outcome: str = "complete", note: str = "") -> None:
        log.info("[cora_log] close procedure=%s outcome=%s note=%s",
                 self.procedure_id, outcome, note)
        # TODO: PATCH /procedures/<id> with outcome

    # context manager sugar -------------------------------------------------

    def __enter__(self) -> "CoraProcedureLog":
        return self.open()

    def __exit__(self, exc_type, exc, tb):
        outcome = "complete" if exc is None else "abort"
        note = "" if exc is None else f"{exc_type.__name__}: {exc}"
        self.close(outcome=outcome, note=note)
        return False  # do not swallow exceptions
