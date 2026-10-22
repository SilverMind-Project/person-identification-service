"""The full-suite flag must turn a silent skip into a failing run."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests import skip_guard

_SUITE = """
import pytest

def test_runs():
    assert True

def test_needs_database():
    pytest.skip("Database not available")
"""

_IMPORT_SKIP_SUITE = """
import pytest

pytest.importorskip("a_module_that_is_not_installed")

def test_never_collected():
    assert True
"""


@pytest.fixture
def guarded(pytester: pytest.Pytester) -> pytest.Pytester:
    pytester.makeconftest(Path(skip_guard.__file__).read_text())
    return pytester


def test_a_skip_fails_the_run_under_the_flag(
    guarded: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(skip_guard.REQUIRE_FULL_SUITE_ENV, "1")
    guarded.makepyfile(_SUITE)

    result = guarded.runpytest()

    assert result.ret == pytest.ExitCode.TESTS_FAILED
    result.stdout.fnmatch_lines(
        ["*unexpected skips*", "*test_needs_database*Database not available*"]
    )


def test_a_module_level_import_skip_fails_the_run_under_the_flag(
    guarded: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(skip_guard.REQUIRE_FULL_SUITE_ENV, "1")
    guarded.makepyfile(_IMPORT_SKIP_SUITE)

    result = guarded.runpytest()

    assert result.ret == pytest.ExitCode.TESTS_FAILED


def test_a_skip_passes_without_the_flag(
    guarded: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(skip_guard.REQUIRE_FULL_SUITE_ENV, raising=False)
    guarded.makepyfile(_SUITE)

    result = guarded.runpytest()

    assert result.ret == pytest.ExitCode.OK


def test_a_real_failure_keeps_its_own_exit_status(
    guarded: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(skip_guard.REQUIRE_FULL_SUITE_ENV, "1")
    guarded.makepyfile("def test_fails():\n    assert False\n")

    result = guarded.runpytest()

    assert result.ret == pytest.ExitCode.TESTS_FAILED
