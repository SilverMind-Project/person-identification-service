# Person Identification Service: quality gate.
#
# `make check` is the gate, locally and in CI. Database tests need TEST_DATABASE_URL naming a
# database whose name ends in _test (tests/db.py refuses anything else), for example:
#
#   TEST_DATABASE_URL=postgresql://pid_user:change-me-pid-password@localhost:5432/person_identification_test make check
#
# Without it the database tests skip. PID_REQUIRE_FULL_SUITE=1 (set in CI) turns any skip into
# a failure; see tests/skip_guard.py.
#
# --frozen keeps uv from rewriting uv.lock; the calibration-tools extra is required because the
# fitter and quality-breakpoint tests import scikit-learn and scipy.

UV := uv run --frozen --extra dev --extra calibration-tools

.PHONY: help lint format-check typecheck test check audit hooks-test

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-14s %s\n", $$1, $$2}'

lint: ## Ruff lint
	$(UV) ruff check .

format-check: ## Ruff format check (no writes)
	$(UV) ruff format --check .

typecheck: ## Mypy over app/
	$(UV) mypy app/

test: ## Full pytest suite
	$(UV) pytest

hooks-test: ## Claude Code hook test table (.claude/hooks/tests/run.sh; needs jq)
	bash .claude/hooks/tests/run.sh

check: lint format-check typecheck test hooks-test ## The quality gate

# pip-audit must read the locked requirements: a bare `uvx pip-audit` audits the throwaway
# environment uvx builds for pip-audit itself and always reports clean. triton-shared is a git
# source with no PyPI advisories and no hash, so it is left out of the export.
audit: ## Audit the locked dependencies for published advisories
	@req=$$(mktemp); trap 'rm -f "$$req"' EXIT; \
	uv export --frozen --no-emit-project --no-emit-package triton-shared \
		--extra dev --extra calibration-tools --format requirements-txt > "$$req"; \
	uvx pip-audit --desc --disable-pip -r "$$req"
