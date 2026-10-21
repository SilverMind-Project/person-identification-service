---
name: engineering-standards
description: Use for any code change in person-identification-service (routers, services, models, migrations, configuration, tests, metrics, wire contracts). Load face-confidence-calibration as well when touching similarity, confidence, calibration, or recognition state.
---

# Person Identification Service: Engineering Standards

This skill documents the service as it is and the rules every change must follow. Update it in
the same PR that changes a rule or a contract it describes. Load together with:

- `/home/sriram/code/nanai/person-identification-service/AGENTS.md` (architecture reference)
- `/home/sriram/code/nanai/person-identification-service/.claude/skills/face-confidence-calibration/SKILL.md`
  for any similarity, confidence, calibration, or recognition-state change
- `/home/sriram/code/nanai/continuous-tracking/.claude/skills/cts-identity-governance/SKILL.md`
  when a change alters what tracking-orchestrator receives
- `/home/sriram/code/nanai/.claude/skills/nanai-workspace/SKILL.md` for cross-repo changes

## What this service owns

Face biometrics for the household: SCRFD detection, ArcFace embeddings, the enrollment gallery,
the calibration artifact runtime, and visitor clustering. It is the only component that stores
face embeddings. Callers are tracking-orchestrator (`POST /api/v1/identify`, one person crop per
call) and cognitive-companion (`/identify-batch`, enrollment, visitor admin).

## Layering

```text
app/models/      Pydantic v2 wire models (request/response). Boundary validation only.
app/routers/     Thin FastAPI handlers: decode, call services from request.app.state, map to wire.
app/services/    Business logic and persistence (FaceEngine, EnrollmentStore, VisitorStore, ...).
                 Gallery code is split into a repository protocol with Postgres and in-memory
                 peers (enrollment_repository.py), pure policy (match_policy.py), and a thin
                 orchestrating service (enrollment_store.py). New gallery features follow it.
app/calibration/ Offline fitter (scikit-learn, optional extra) and stdlib-only runtime evaluator.
app/main.py      Lifespan wiring: the single place services are constructed.
```

Rules:

1. Routers never construct services and never run SQL; they read `request.app.state`.
2. Matching policy (thresholds, state assignment, margins) is a pure function over similarities,
   testable without a database. Keep SQL in the store, policy outside it. New code must follow
   this even where older code mixes them.
3. Wire models are Pydantic; internal carriers are dataclasses (`DetectedFace`,
   `IdentifyResult`). Never return a Pydantic model from a service.
4. The scikit-learn toolchain is never imported by `app/main.py` or any request path; runtime
   calibration is stdlib plus numpy.

## Wire contract discipline

- Responses are additive-only. A new field gets a default so older callers keep working; a
  removed field goes through a deprecation window recorded in this skill, with every consumer
  (tracking-orchestrator `app/inference/face_id_client.py`, cognitive-companion
  `backend/integrations/person_id_client.py`) migrated in the same change set.
- Name every similarity `similarity` or `raw_similarity`. Never name cosine similarity
  `confidence`, `probability`, or `certainty`. The face `confidence` field was removed on
  2026-10-20. Deprecation window open: `raw_similarity` is a computed alias of `similarity`, kept
  for one release so older consumers keep working; both consumers already read `raw_similarity`
  and fall back to `similarity`, so the alias can go in the next release that touches the
  identify contract.
- Every dimensional field carries its unit in the name: `_px`, `_deg`, `_s`.
- Face geometry is part of the identify contract: each face carries `face_width_px`,
  `face_height_px`, and nullable `interocular_px` (distance between the two SCRFD eye keypoints,
  computed only in `app/services/face_geometry.py`), measured in pixels of the submitted image;
  `IdentifyResponse` and each batch `FrameResult` carry `image_width_px` / `image_height_px`.
  A value that cannot be measured is `null`, never `0.0`. `DetectedFace.geometry` is derived in
  `__post_init__`, so every producer carries the same measurement.
- Face quality is part of the identify contract (`quality`, see the face-confidence-calibration
  skill); it is computed only in `app/services/face_quality.py`.
- Side effects are explicit. `/identify` currently writes visitor sightings for unknown faces on
  every call (`_ingest_unknown_faces` in `app/routers/identification.py`) and, with
  `calibration_capture.enabled` and a `capture_context`, queues sampled faces as calibration
  candidates (crop in MinIO under `calibration/`, row in `calibration_candidates`); any new endpoint that
  writes must say so in its docstring and OpenAPI description, and read-only measurement tools
  must call `FaceEngine` directly instead of `/identify`.

## Configuration

`app/config.py` loads `config/settings.yaml` with `${ENV_VAR:default}` interpolation. Read
values in constructors, not per request. Never hardcode a threshold. Secrets come from the
environment only. Every new key gets a comment stating unit, meaning, and the evidence behind
its default. Comments never cite milestone or decision numbers.

## Persistence and migrations

- PostgreSQL with TimescaleDB, pgvector, and pgvectorscale. The pool registers the pgvector codec
  (`_init_pool` in `app/main.py`); never open a raw connection without it.
- Migrations are incremental (owner decision 2026-10-20): every new change is a
  `migrations/NNNN_description.up.sql` applied by `app/db/migrate.py` at startup, and ships a
  matching `.down.sql` that exactly reverses it (`tests/test_migration_runner.py` fails on a
  missing one). The runner holds an advisory lock; `python -m app.db.migrate status|rollback
  --steps N` is the operator tool, and rollback refuses the baseline or any version without a down
  file. `0001_initial_schema` is frozen. Never edit
  an applied migration and never drop the database to apply a change: enrollment and visitor data
  are irreplaceable household data.
- Write idempotent DDL (`IF NOT EXISTS`). Parameterized SQL only (`$1..$N`).

## Logging, metrics, privacy

- stdlib `logging.getLogger(__name__)` with `key=value` message fields. Never `print`.
- Never log base64 images, embeddings, member names, or crop bytes.
- Prometheus metrics use the default registry and are exposed at `/metrics`. Label values must
  be bounded (states, reasons), never person ids.
- Face crops written to MinIO have a documented retention; any new stored biometric artifact
  needs a retention rule in the same change.

## Error handling

- Model or Triton failure on a request is an HTTP error with a structured detail; never a
  silent empty result that looks like "no face".
- Background loops catch, log with `logger.exception`, and continue; they never die silently.
- No bare `except:`; a broad `except Exception` must log and must not swallow programmer errors
  in tests.

## Testing

- `pytest` with `asyncio_mode = "auto"`. Tests live in `tests/`.
- Database-backed tests use a real asyncpg pool from `tests/db.py::safe_test_dsn()`: only
  `TEST_DATABASE_URL`, only a database named `*_test`, skipped when unset. Never read
  `DATABASE_URL` or hardcode a default DSN in a test: teardowns delete rows, and a default to the
  service database destroyed household enrollment once. Never mock asyncpg.
- A repository with an in-memory peer has a Postgres parity test running the same operations on
  both (`tests/test_enrollment_repository_parity.py` is the pattern).
- Triton is never called in tests; inject a fake implementing the client protocol, and a
  `_FakeFaceEngine` returning deterministic embeddings.
- Pure policy functions (matching, quality, calibration evaluation) get table-driven unit tests
  with no database.
- Every wire change gets a contract test asserting field names, types, and defaults, and a
  matching consumer test in each calling repository.
- Tests assert falsifiable outcomes (exact states, counts, thresholds at the boundary), never
  mere truthiness.

## Quality gate

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy app/
uv run pytest
```

All four must pass before a PR is declared done.

## Review checklist

- [ ] No similarity is named confidence or probability in new code or wire fields.
- [ ] Policy is pure and unit-tested; SQL lives only in stores.
- [ ] Wire changes are additive, tested, and mirrored in every consumer.
- [ ] Migration is a new numbered file with a correct down migration.
- [ ] No biometric data or names in logs; new stored artifacts have retention.
- [ ] AGENTS.md, CLAUDE.md, and this skill are updated when a rule or contract changed.
