#!/usr/bin/env bash
# Verify that Python runtime pins stay aligned across pyproject.toml, Dockerfile, and CI.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FAIL=0

check_eq() {
  local label="$1" expected="$2" actual="$3"
  if [ "$expected" != "$actual" ]; then
    echo "TOOLCHAIN DRIFT [$label]: expected '$expected', got '$actual'" >&2
    FAIL=1
  fi
}

PY_MINOR="$(grep -oE 'python_version = "[0-9]+\.[0-9]+"' "$ROOT/pyproject.toml" | head -n1 | sed -E 's/.*"([^"]+)"/\1/')"
PY_COMPACT="py${PY_MINOR/./}"

RUFF_TARGET="$(grep -oE 'target-version = "py[0-9]+"' "$ROOT/pyproject.toml" | head -n1 | sed -E 's/.*"([^"]+)"/\1/')"
check_eq "pyproject.toml ruff target-version" "$PY_COMPACT" "$RUFF_TARGET"

while IFS= read -r docker_py; do
  check_eq "Dockerfile python" "$PY_MINOR" "$docker_py"
done < <(grep -oE 'FROM python:[0-9]+\.[0-9]+' "$ROOT/Dockerfile" | cut -d: -f2)

CI_PY="$(grep -oE 'uv python install [0-9]+\.[0-9]+' "$ROOT/.github/workflows/ci.yml" | head -n1 | awk '{print $NF}')"
check_eq ".github/workflows/ci.yml uv python install" "$PY_MINOR" "$CI_PY"

if [ "$FAIL" -ne 0 ]; then
  exit 1
fi
echo "Toolchain pins aligned (Python ${PY_MINOR})."
