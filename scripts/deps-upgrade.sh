#!/usr/bin/env bash
# Check or perform coordinated PyPI dependency upgrades in uv.lock while preserving
# internal git pins, then run the quality gate and pip-audit.
#
# Usage:
#   scripts/deps-upgrade.sh --check   Read-only list of outdated Python packages
#   scripts/deps-upgrade.sh --apply   Upgrade PyPI packages in uv.lock and verify

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODE="${1:---check}"

case "$MODE" in
  --check)
    echo "=== Outdated Python packages (person-identification-service) ==="
    (cd "$ROOT" && uv pip list --python .venv/bin/python --outdated || true)
    ;;

  --apply)
    echo "=== 1. Upgrading PyPI packages in uv.lock (preserving internal git pins) ==="
    (
      cd "$ROOT"
      mapfile -t PYPI_ARGS < <(
        uv run --frozen --extra dev --extra calibration-tools python - <<'PY'
import tomllib
from pathlib import Path

lock = tomllib.loads(Path("uv.lock").read_text(encoding="utf-8"))
for pkg in lock.get("package", []):
    src = pkg.get("source", {})
    if isinstance(src, dict) and "registry" in src:
        print(f"--upgrade-package={pkg['name']}")
PY
      )
      uv lock "${PYPI_ARGS[@]}"
      uv sync --frozen --extra dev --extra calibration-tools
    )

    echo "=== 2. Running quality gate and dependency audit ==="
    (cd "$ROOT" && make check && make audit)
    ;;

  *)
    echo "usage: $0 [--check|--apply]" >&2
    exit 64
    ;;
esac
