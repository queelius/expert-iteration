"""Minimal logging helpers.

Stub for now. Returns a logger with a sensible default format so that
experiment scripts and tests can call ``get_logger(__name__)`` without
boilerplate. Real content (structured logs, run-id propagation, file
handlers) will be added when experiment tracking lands.
"""

from __future__ import annotations

import logging

_DEFAULT_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger.

    If the root logger has no handlers, attach a single ``StreamHandler``
    with the default format. Idempotent: repeated calls do not stack
    handlers.
    """
    root = logging.getLogger()
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT))
        root.addHandler(handler)
        root.setLevel(level)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger
