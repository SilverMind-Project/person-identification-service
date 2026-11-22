"""Deliberate ruff violations for tests/test_lint_canary.py (excluded from normal lint runs).

This module intentionally omits ``from __future__ import annotations`` (I002) and contains one
violation per guarded ruff rule so the canary test proves pyproject.toml still enforces every rule.
"""

import datetime
import logging

from . import config  # relative import (TID252)

logger = logging.getLogger(__name__)


def canary_violations(table: str) -> None:
    naive = datetime.datetime(2026, 1, 1, 12, 0)  # DTZ001
    x: int = "bad"  # type: ignore  # PGH003
    logger.info(f"table={table} naive={naive} x={x}")  # G004
    _sql = f"SELECT * FROM {table}"  # S608
    # unused_var = 42  # ERA001
    try:
        int(table)
    except Exception:  # BLE001
        return


async def canary_async_blocking(path: str) -> bytes:
    with open(path, "rb") as fh:  # ASYNC230
        return fh.read()
