# Person Identification Service: Claude guidance

Triton-backed face recognition and motion direction detection microservice for the Cognitive
Companion system: Python 3.12 FastAPI, Buffalo_L over Triton gRPC, PostgreSQL (TimescaleDB +
pgvector), MinIO. It owns face biometrics: detection, ArcFace embeddings, the enrollment
gallery, face confidence calibration, and visitor clustering.

**Architecture and API reference: [AGENTS.md](AGENTS.md)** (layout, lifecycle, settings,
services, endpoints, schema, testing patterns, common tasks). This file holds only what every
session needs.

## Skills to load

| Skill | Use when |
| --- | --- |
| `.claude/skills/engineering-standards/SKILL.md` | Every change |
| `.claude/skills/face-confidence-calibration/SKILL.md` | Similarity outputs, confidence calibration, recognition-state contracts, calibration datasets and artifacts, `/health` calibration status |
| `/home/sriram/code/nanai/.claude/skills/nanai-workspace/SKILL.md` | Any change spanning more than one repository |

Cross-cutting rules (markdown, comments, the Python environment, the test database) are in
`.claude/rules/` and load for matching files.

## Gate

```bash
export TEST_DATABASE_URL=postgresql://pid_user:...@localhost:5432/person_identification_test
make check                            # ruff, format check, mypy, full suite, hooks, markdown, doc refs
PID_REQUIRE_FULL_SUITE=1 make check   # what CI runs: a skipped test fails the gate
make install-hooks                    # once per clone: versioned git hooks, em-dash check, Git LFS
```

Other commands (dev server, calibration CLI, migrations, Docker) are in AGENTS.md section 4.

## Database safety

Database tests take their connection string only from `tests/db.py::safe_test_dsn()`: it reads
`TEST_DATABASE_URL`, refuses any database whose name does not end in `_test`, and the tests skip
without it. They never read `DATABASE_URL`. Their teardowns delete every row they touch: on
2026-10-20 an earlier default to the service database deleted the household's enrollment. Never
run tests against `person_identification`, never add a default connection string to a test, and
never mock the database instead.

## Claude Code guardrails

Start sessions in this directory (or `/cd` here): `.claude/settings.json` applies only from the
session's working directory. It sets `UV_FROZEN=1` and runs four hooks from `.claude/hooks/`:

- **Guard:** blocks any `pytest` run without `TEST_DATABASE_URL` naming a `*_test` database,
  because it would silently skip every database test. `make check` and `make test` are allowed,
  since they read the variable from the environment.
- **Post-edit:** an em-dash in an edited `.md` file or a `ruff check` failure comes back as
  feedback.
- **Stop:** flags added em-dashes and `uv.lock` churn. It blocks once and never loops.
- **Session start:** the gate command, the test database URL, and the live write-counter check.

Machine-wide guards (including write SQL against the live databases) live in `~/.claude/`. Run
any blocked command yourself with `!` if it is intended. `make hooks-test` is part of
`make check`.

## Non-negotiable invariants

- Triton is the only inference path; startup fails unless all five Buffalo_L models are ready.
- Services are built in the `app.main` lifespan and read from `request.app.state`; never
  instantiated in routers.
- Schema changes are new numbered `migrations/NNNN_name.up.sql` files with a matching
  `.down.sql`; `0001_initial_schema` is frozen; never drop the database to apply a change.
- Enrollment gains entries only through explicit enroll calls or explicit operator naming of a
  visitor cluster; calibration labels never change the gallery.
- `similarity` is a raw cosine score, never a probability; `calibrated_confidence` is null
  without a compatible artifact and never falls back to raw similarity.
- Logging is stdlib `logging.getLogger(__name__)` by design (other repositories differ); never
  `print()`, never log images, crops, or embeddings. No em-dashes in markdown.

## Cross-repository contracts

- **Face wire contract.** continuous-tracking calls `POST /api/v1/identify` with one person crop
  (`continuous-tracking/tracking-orchestrator/app/inference/face_id_client.py`); cognitive-companion uses
  `/identify-batch`, enrollment, visitors, and calibration review
  (`cognitive-companion/backend/integrations/person_id_client.py`). A field added to, renamed in,
  or removed from `FaceDetection` changes those clients in the same change set.
- `/identify` has side effects: unknown faces are offered to visitor clustering, and a request
  with `capture_context` may be sampled for calibration. Read-only measurement tools call
  `FaceEngine` directly.
