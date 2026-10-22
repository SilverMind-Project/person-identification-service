"""Fail the run on any unexpected skip when ``PID_REQUIRE_FULL_SUITE=1``.

Database tests skip without ``TEST_DATABASE_URL``, and the calibration-fitter tests skip
without the ``calibration-tools`` extra, so a run missing either still reports green while
proving nothing about the repositories, the migration runner, or the fitter. CI sets the flag;
locally it is opt-in, so a quick run without a test database still works.
"""

from __future__ import annotations

import os

import pytest

REQUIRE_FULL_SUITE_ENV = "PID_REQUIRE_FULL_SUITE"

# Skip reasons that remain acceptable even in a full-suite run. Empty: every current skip
# means a missing database or extra.
ALLOWED_SKIP_REASONS: frozenset[str] = frozenset()

_unexpected_skips: list[str] = []


def _required() -> bool:
    return os.environ.get(REQUIRE_FULL_SUITE_ENV, "") == "1"


def _reason(report: pytest.CollectReport | pytest.TestReport) -> str:
    longrepr = report.longrepr
    if isinstance(longrepr, tuple) and len(longrepr) == 3:
        return str(longrepr[2]).removeprefix("Skipped: ")
    return str(longrepr)


def _record(report: pytest.CollectReport | pytest.TestReport) -> None:
    if report.skipped and _reason(report) not in ALLOWED_SKIP_REASONS:
        _unexpected_skips.append(f"{report.nodeid}: {_reason(report)}")


def pytest_collectreport(report: pytest.CollectReport) -> None:
    # A module-level pytest.importorskip skips at collection, not at test time.
    _record(report)


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    _record(report)


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    # A run whose only module was skipped at import ends NO_TESTS_COLLECTED, not OK; both are
    # "green" for a skip-only run. A real failure or error keeps its own status.
    passing = (pytest.ExitCode.OK, pytest.ExitCode.NO_TESTS_COLLECTED)
    if _required() and _unexpected_skips and exitstatus in passing:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED


def pytest_terminal_summary(terminalreporter: pytest.TerminalReporter) -> None:
    if _required() and _unexpected_skips:
        terminalreporter.section(f"{REQUIRE_FULL_SUITE_ENV}=1: unexpected skips")
        for line in _unexpected_skips:
            terminalreporter.line(line)
