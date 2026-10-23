---
paths: ["tests/**", "**/person-identification-service/tests/**"]
---

# Test database

- A database test gets its DSN only from `tests/db.py::safe_test_dsn()`: `TEST_DATABASE_URL`, a
  database whose name ends in `_test`, skipped when unset. Never read `DATABASE_URL`, never
  hardcode or default a DSN, and never mock asyncpg.
- Teardowns delete every row they touch. A test pointed at `person_identification` destroys the
  household enrollment; the guard hook and `safe_test_dsn()` exist because that happened.
- Run with `TEST_DATABASE_URL=postgresql://pid_user:...@localhost:5432/person_identification_test`;
  CI sets `PID_REQUIRE_FULL_SUITE=1`, so a skip fails the gate.
