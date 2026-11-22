"""Non-vacuity canary for the ruff lint configuration in pyproject.toml.

Runs ruff against ``tests/fixtures/lint_canary/canary_violations.py`` using the repository's real
``pyproject.toml`` (without ``--select`` overrides) so removing or ignoring any guarded rule fails
this test.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PYPROJECT = _REPO_ROOT / "pyproject.toml"
_CANARY = _REPO_ROOT / "tests" / "fixtures" / "lint_canary" / "canary_violations.py"

_EXPECTED_CODES = frozenset(
    {
        "ASYNC230",
        "BLE001",
        "DTZ001",
        "ERA001",
        "G004",
        "I002",
        "PGH003",
        "S608",
        "TID252",
    }
)


def _ruff_codes(config_path: Path) -> set[str]:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ruff",
            "check",
            "--config",
            str(config_path),
            "--output-format=json",
            "--stdin-filename",
            "app/_lint_canary.py",
            "-",
        ],
        input=_CANARY.read_text(encoding="utf-8"),
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode in (0, 1), result.stderr
    findings = json.loads(result.stdout)
    return {item["code"] for item in findings}


def test_pyproject_enforces_all_guardrail_rules() -> None:
    reported = _ruff_codes(_PYPROJECT)
    missing = _EXPECTED_CODES - reported
    assert not missing, f"pyproject.toml did not report expected ruff codes: {sorted(missing)}"


def test_canary_detects_removed_rule(tmp_path: Path) -> None:
    stripped = _PYPROJECT.read_text(encoding="utf-8").replace('    "BLE",', "")
    tmp_config = tmp_path / "pyproject.toml"
    tmp_config.write_text(stripped, encoding="utf-8")

    reported = _ruff_codes(tmp_config)
    assert "BLE001" not in reported


def test_canary_fixture_is_excluded_from_normal_lint() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "ruff", "check", "--output-format=json", "."],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
