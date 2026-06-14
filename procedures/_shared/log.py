"""ANSI-coloured console logging for procedures.

Adapted from https://github.com/xray-imaging/energy/blob/main/src/energy/log.py.
INFO -> green, WARNING -> yellow, ERROR / CRITICAL -> red.
DEBUG is left uncoloured so the procedure's debug noise doesn't
swamp the operator's eye.

Only the ``%(message)s`` portion of the record is wrapped in the
colour codes -- timestamp and level prefix stay plain so a piped
file is still grep-friendly.
"""

from __future__ import annotations

import logging
import sys


_GREEN = "\033[92m"
_YELLOW = "\033[33m"
_RED = "\033[91m"
_ENDC = "\033[0m"


class ColoredLogFormatter(logging.Formatter):
    """Wraps ``%(message)s`` in an ANSI colour per level."""

    def formatMessage(self, record: logging.LogRecord) -> str:
        if record.levelname == "INFO":
            record.message = _GREEN + record.message + _ENDC
        elif record.levelname == "WARNING":
            record.message = _YELLOW + record.message + _ENDC
        elif record.levelname in ("ERROR", "CRITICAL"):
            record.message = _RED + record.message + _ENDC
        return super().formatMessage(record)


def setup_console_logger(
    level: str = "INFO",
    fmt: str = "%(asctime)s  %(levelname)-7s  %(name)s: %(message)s",
    use_color: bool | None = None,
) -> None:
    """Configure the root logger with a single colour-aware stream handler.

    Replaces the procedure's previous ``logging.basicConfig`` call. Idempotent
    (clears any existing handlers on the root logger so re-running ``main``
    inside the same Python process doesn't duplicate output).

    ``use_color=None`` auto-detects: ANSI on when stdout is a TTY, off when
    piped to a file or non-tty. Pass ``True`` / ``False`` to force.
    """
    root = logging.getLogger()
    root.setLevel(getattr(logging, level))
    for h in list(root.handlers):
        root.removeHandler(h)

    handler = logging.StreamHandler(sys.stdout)
    if use_color is None:
        use_color = sys.stdout.isatty()
    handler.setFormatter(
        ColoredLogFormatter(fmt) if use_color else logging.Formatter(fmt)
    )
    root.addHandler(handler)
