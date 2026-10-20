# AGENTS.md

Load `/home/sriram/code/nanai/person-identification-service/.claude/skills/engineering-standards/SKILL.md`
for every change. For ArcFace similarity/confidence contracts or calibration work, also load
`/home/sriram/code/nanai/person-identification-service/.claude/skills/face-confidence-calibration/SKILL.md`.

Reference for AI coding agents working in `person-identification-service/`. This document is the canonical, deep guide. `CLAUDE.md` is a tight pointer aimed at the same audience; `README.md` is human-facing.

If a fact appears here, it traces to a file in this tree at the time of writing. Verify before relying on it: `git log` is authoritative for "what changed", and `grep` against `app/` is authoritative for "what exists".

---

## 1. Mission and scope

Person Identification Service is a Triton-backed face recognition and motion direction detection microservice for the Cognitive Companion system. It ingests base64-encoded camera frames via a REST API and returns identities, raw similarity and nullable calibrated confidence, face geometry (box size and interocular distance in pixels), bounding boxes, and motion direction classifications. It also owns face confidence calibration (offline fitter plus runtime evaluator) and visitor clustering of recurring unknown faces.

Callers: tracking-orchestrator sends one person crop per `POST /api/v1/identify`; cognitive-companion uses `/identify-batch`, enrollment, and the visitor review endpoints.

Two characteristics make this service non-trivial:

1. **Remote model lifecycle.** All five Buffalo_L graphs are served by Triton. Startup fails unless every configured model is ready.
2. **Vector similarity search.** Face identification uses nearest-centroid search against a pgvector DiskANN index, replacing the older in-memory `.npy` gallery approach. The database is the source of truth for enrollment state.

---

## 2. Tech stack

| Layer | Choice |
| --- | --- |
| Backend | Python 3.12, FastAPI, Pydantic 2, stdlib `logging` |
| Database | PostgreSQL with TimescaleDB + pgvector extensions; asyncpg driver with `pgvector` package for vector codec |
| Object storage | MinIO (S3-compatible) via `boto3` |
| Face AI | Buffalo_L: SCRFD, ArcFace, 2D/3D landmarks, gender/age |
| Inference runtime | NVIDIA Triton Inference Server over gRPC |
| Image processing | OpenCV (`opencv-python-headless`) |
| Package manager | `uv` (`uv.lock` is committed) |
| Lint and types | `ruff`; `mypy` gradual typing |

---

## 3. Repository layout

```text
person-identification-service/
├── app/
│   ├── main.py              # App factory + lifespan (service wiring source of truth)
│   ├── config.py            # YAML config loader with ${ENV_VAR:default} interpolation
│   ├── calibration/
│   │   ├── models.py        # CalibrationArtifact / manifest Pydantic models (JSON, never pickle)
│   │   ├── evaluator.py     # Runtime evaluator (stdlib + numpy only); health states
│   │   ├── fit.py           # Offline fitter (scikit-learn; calibration-tools extra)
│   │   ├── quality_breakpoints.py # Offline face quality anchor analysis (scipy; calibration-tools)
│   │   └── cli.py           # fit / evaluate / inspect / validate commands
│   ├── db/
│   │   └── migrate.py       # asyncpg migration runner (applies migrations/*.up.sql)
│   ├── models/
│   │   ├── enrollment.py    # EnrollRequest, EnrollResult, MemberInfo, MemberListResponse
│   │   ├── identification.py # FaceDetection, IdentifyRequest/Response, BatchIdentifyRequest/Response, PersonMotion, FrameResult
│   │   ├── motion.py        # MotionDetectionRequest/Response, PersonTrack, TrajectoryPoint
│   │   └── visitor.py       # Visitor cluster list/detail/name envelopes
│   ├── observability/
│   │   └── metrics.py       # Prometheus metrics (interocular and quality histograms, visitor counters)
│   ├── routers/
│   │   ├── health.py        # GET /health, GET /metrics (no /api/v1 prefix)
│   │   ├── enrollment.py    # POST /enroll, POST /enroll/upload/{id}, GET/DELETE /members/{id}
│   │   ├── identification.py # POST /identify, POST /identify-batch
│   │   ├── motion.py        # POST /detect-motion
│   │   └── visitors.py      # /visitors/clusters list, detail, name, dismiss, merge
│   └── services/
│       ├── face_engine.py     # Triton preprocessing, inference, and postprocessing
│       ├── face_models.py     # DetectedFace, IdentifyResult dataclasses
│       ├── face_geometry.py   # Face box size and interocular distance (pure)
│       ├── face_quality.py    # Face image quality scorer protocol + heuristic-v1 (pure)
│       ├── enrollment_store.py # pgvector gallery: enroll, identify, remove
│       ├── centroid.py        # Shared embedding insert + centroid recompute (enroll and visitor naming)
│       ├── visitor_store.py   # Visitor sightings, clustering, surfacing, naming, retention
│       ├── minio_client.py    # boto3 S3 wrapper for MinIO
│       ├── motion_detector.py # Cross-frame centroid tracking + direction classification
│       ├── guest_store.py     # MinIO uploads for unidentified-person frames
│       └── image_annotator.py # Bounding box + label drawing
├── config/
│   └── settings.yaml        # All application settings
├── schemas/
│   └── calibration_artifact.schema.json
├── migrations/
│   └── 0001_initial_schema.up.sql  # Frozen baseline; new changes are 0002+ (see section 8)
├── tests/                   # See section 9
├── pyproject.toml           # Build config, deps (extras: dev, model-tools, calibration-tools)
├── Dockerfile               # Accelerator-independent API build
├── docker-compose.yml       # API service definition
├── CLAUDE.md                # Agent quick-reference
└── README.md                # Human-facing documentation
```

`app/core/` and `app/utils/` exist as empty directories; there is no code there.

---

## 4. Commands

Run from the repository root unless noted.

```bash
# Development server
uv run uvicorn app.main:app --host 0.0.0.0 --port 8200 --reload

uv sync

# Offline conversion and validation tools
uv sync --extra model-tools

# Calibration toolchain (offline; never imported by the service)
uv run --extra calibration-tools python -m app.calibration.cli fit --pairs pairs.json --output artifact.json
uv run --extra calibration-tools python -m app.calibration.cli validate --artifact artifact.json

# Code quality
uv run ruff check .              # Lint
uv run ruff format .             # Format
uv run mypy app/                 # Type check

# Tests. Database tests run only against a database whose name ends in _test, and skip
# without TEST_DATABASE_URL; they never read DATABASE_URL (their teardown deletes rows).
export TEST_DATABASE_URL=postgresql://pid_user:...@localhost:5432/person_identification_test
uv run pytest                    # Full suite
uv run pytest tests/test_enrollment_store.py -v   # Targeted

# Migrations (startup applies pending ones; these are for operators)
uv run python -m app.db.migrate status
uv run python -m app.db.migrate rollback --steps 1

# Docker
docker build -t person-id-service .
docker run --gpus all -p 8200:8200 -v $(pwd)/data:/app/data person-id-service
docker compose up -d
```

---

## 5. Architecture

### 5.1 Backend layering

```text
app/models/          Pydantic wire models (request/response schemas)
app/routers/         FastAPI route handlers (thin: decode, call services, return)
app/services/        Business logic (FaceEngine, EnrollmentStore, MotionDetector, etc.)
                     Gallery work is split three ways: persistence behind the
                     EnrollmentRepository protocol (enrollment_repository.py, Postgres and
                     in-memory peers), the pure matching policy (match_policy.py), and the thin
                     EnrollmentStore that sequences engine, repository, and policy.
app/db/              Database migration runner
app/main.py          App factory + lifespan (wiring)
app/config.py        YAML + env-var configuration singleton
```

There is no `core/` layer in this service (unlike Cognitive Companion). The config module is loaded eagerly at import time, and services are constructed in the lifespan.

### 5.2 Data flow

```text
Camera frame (base64)
  → Router decodes to BGR numpy array
  → FaceEngine.detect_faces(image)
    → SCRFD detection → raw face crops
    → ArcFace embedding extraction → 512-dim vectors
  → EnrollmentStore.identify(embedding) or identify_all(faces)
    → pgvector cosine distance query (<=>) with LIMIT 1
    → centroid match or "unknown"/"Guest"
  → For batch: accumulate per-frame results
  → MotionDetector.detect_direction(frame_shapes, faces, identities)
    → Group by person_id, link unknowns via embedding similarity
    → Classify direction per track
  → Optionally: image_annotator.annotate_image() for bounding box output
  → Optionally: guest_store.save_guest_image() to MinIO
  → Return typed Pydantic response
```

### 5.3 Service lifecycle

In `app.main` lifespan (order matters):

1. `logging.basicConfig()` from config
2. `_init_pool(dsn)`: asyncpg pool with the `pgvector` codec registered (`min_size=1, max_size=5`)
3. `run_migrations(pool)`: applies pending `migrations/*.up.sql` files transactionally
4. `create_minio_client()`: MinIO client with bucket ensured
5. Open `TritonGrpcClient`, build `FaceEngine`, and require all five configured models to be ready
6. `EnrollmentStore(pool, face_engine)`, `MotionDetector()`, `GuestImageStore(pool, minio_client)`
7. `VisitorStore(pool, minio_client)` plus the daily retention task (`_retention_loop`, purges expired unnamed visitor clusters)
8. `CalibrationEvaluator.from_artifact_path(...)` with the expected ArcFace model, model profile, and preprocessing versions; a missing or incompatible artifact yields a degraded evaluator, never a startup failure
9. Store on `app.state`: `face_engine`, `triton_client`, `enrollment_store`, `motion_detector`, `guest_store`, `visitor_store`, `calibration_evaluator`
10. On shutdown: cancel the retention task, close Triton, then close the database pool

### 5.4 Configuration

`app/config.py` loads `config/settings.yaml` eagerly at module import time. Env-var interpolation uses `${VAR:default}` syntax.

```python
from app.config import config
threshold = config.get("recognition.threshold", 0.4)
```

Dot-notation keys map to nested dict paths. `config.reload()` clears the cache and re-reads the file (used only in tests; in production, restart the process).

All secrets (database DSN, MinIO credentials) come from environment variables. The config file contains only non-sensitive defaults.

---

## 6. Services in depth

### 6.1 FaceEngine (`app/services/face_engine.py`)

Implements the Buffalo_L client contract over Triton. There is no local ONNX Runtime path and no partial-model fallback.

```python
engine = FaceEngine(triton_client)
await engine.validate_models()
faces: list[DetectedFace] = await engine.detect_faces(image)
similarity: float = FaceEngine.compute_similarity(emb1, emb2)  # static
```

**Key details:**
- `det_size: [640, 640]` is required by the SCRFD Triton contract
- `det_threshold: 0.6` -- filters low-confidence detections
- Returns `DetectedFace` with bounding box, normalized embedding, landmarks, pose, and attributes
- SCRFD decode/NMS and ArcFace alignment remain client-side and are unit-tested
- `compute_similarity` is `np.dot(emb1, emb2)` (valid because embeddings are normalized)

### 6.2 EnrollmentStore (`app/services/enrollment_store.py`)

Orchestrates enrollment and identification. It holds no SQL and reads no configuration:
`EnrollmentStore(repository, face_engine, thresholds)` is built in the lifespan from a
`PostgresEnrollmentRepository(pool)` and `MatchThresholds` read from `recognition.*`.

- **Repository** (`app/services/enrollment_repository.py`): the `EnrollmentRepository` protocol
  (`nearest_centroids`, member and embedding CRUD, `recompute_centroid`), with
  `PostgresEnrollmentRepository` and `InMemoryEnrollmentRepository`. The centroid recompute SQL
  lives once, in `app/services/centroid.py`, because visitor naming runs it inside its own
  transaction; the Postgres repository delegates to it.
- **Policy** (`app/services/match_policy.py`): `classify_match(hits, thresholds)` turns the
  nearest centroids into a `MatchDecision`; pure and table-tested.

**Enroll flow:**
1. Check whether the member exists and how many embeddings it has (reported if every image fails)
2. For each image: await `engine.detect_faces()`, then pick the largest face by bbox area
3. `upsert_member` (reports "enrolled" or "updated"), `insert_embeddings`, `recompute_centroid`
   (mean of all the member's embeddings, L2-normalized, upserted into `centroids`)

**Identify flow:** `nearest_centroids(embedding, k=1)` (pgvector `<=>` cosine distance, DiskANN
index on `centroids.centroid`), then `classify_match`.

**Thresholds (raw cosine similarity to the nearest centroid):**
- `recognition.threshold` (default 0.4): at or above is `recognized` (person_id set)
- `recognition.unknown_threshold` (default 0.25): from here up to `threshold` is `candidate` (person_id `"unknown"`, `best_candidate_id` set); below is `unrecognized`

**Face geometry:** `identify_all` also fills `face_width_px`, `face_height_px`, and `interocular_px` from `DetectedFace.geometry`, which `DetectedFace.__post_init__` derives once with `face_geometry()` (`app/services/face_geometry.py`) from the five SCRFD keypoints on `DetectedFace.keypoints_5_px`.

**Face quality:** `FaceEngine._enrich_face` scores each face with the injected `FaceQualityScorer` (`app/services/face_quality.py`, `heuristic-v1`) from interocular distance, `det_score`, pitch, and the grayscale aligned crop; `identify_all` copies `DetectedFace.quality` and the routers serialize it as `FaceDetection.quality`. Interocular distance is the eye-keypoint distance in pixels of the submitted image, or `None` when the keypoints are unusable.

### 6.3 MotionDetector (`app/services/motion_detector.py`)

Cross-frame centroid tracking for direction classification. Designed for 3-5 frame batches from the Cognitive Companion event aggregator.

**Algorithm:**
1. Group face detections by `person_id` across frames
2. For unknown faces, link across frames by embedding cosine similarity (`cross_frame_similarity: 0.5`), creating synthetic track IDs (`unknown_0`, `unknown_1`, etc.). Only include tracks with 2+ entries.
3. For each track, compute trajectory from `(cx, cy, width, height)` per frame
4. Classify direction via `_classify_direction()`:
   - `dx_frac = total_horizontal_displacement / avg_frame_width`
   - `area_change = (final_area - initial_area) / initial_area`
   - Horizontal threshold: `min_displacement_fraction` (default 0.05 = 5% of frame width)
   - Depth threshold: 15% area change
   - Returns `(direction_str, confidence)` where confidence is `min(abs(dx_frac) * 10, abs(area_change) * 3, 0.99)`

**Direction values:** `left-to-right`, `right-to-left`, `towards-camera`, `away-from-camera`, `stationary`

### 6.4 GuestImageStore (`app/services/guest_store.py`)

Saves frames containing unidentified persons to MinIO and records metadata in PostgreSQL.

- Encodes BGR image as JPEG (quality 90) via `cv2.imencode()`
- Uploads to MinIO path: `guests/{YYYY-MM-DD}/{HHMMSS-ffffff}_f{frame_index}_{n}guests.jpg`
- INSERTs into `guest_visits` hypertable: `guest_count`, `object_name`, `frame_index`
- Returns object_name on success, None on failure (graceful degradation)

### 6.5 MinioClient (`app/services/minio_client.py`)

boto3 S3 client configured for MinIO compatibility:
- Path-style addressing (`s3_use_path_style = True`)
- Signature version s3v4
- HTTP or HTTPS depending on `minio.secure` setting
- `endpoint` is host:port only (scheme added by the client)

### 6.6 Image Annotator (`app/services/image_annotator.py`)

Draws bounding boxes and labels on a copy of the input image:
- Known persons: green box (`[0, 200, 0]` BGR) with `"Name 85%"` label
- Unknown persons: orange box (`[0, 165, 255]` BGR) with `"Guest 42%"` label
- All colors, scales, thicknesses configurable via `annotation.*` settings
- Returns a new numpy array; input image is not mutated

### 6.7 VisitorStore (`app/services/visitor_store.py`)

Manages the visitor clustering subsystem (M06) for recurring unknown visitors.

**Tables:**
- `visitor_clusters`: tracks `cluster_id`, `status` (`candidate`, `surfaced`, `named`, `dismissed`), `distinct_days`, and the running-mean centroid.
- `visitor_sightings`: hypertable for each observed face crop (MinIO key) and embedding.

**Config keys (`config.visitors.*`):**
- `clustering_enabled`: Master toggle.
- `min_detection_score` (0.6), `min_face_px` (64): Quality gates.
- `member_margin` (0.05): Member-margin fail-closed rule. A borderline match against the household (similarity > recognition.threshold - member_margin) is explicitly rejected from visitor clustering to prevent a known member from seeding a visitor cluster.
- `cluster_join_threshold` (0.55): Looser than recognition.
- `surface_min_days` (3), `surface_window_days` (30): Promotion criteria.
- `unnamed_retention_days` (60): Retention period.
- `enroll_top_k` (5): Embeddings copied on naming.

**Retention:**
An async daily task deletes unnamed (`candidate`/`surfaced`/`dismissed`) clusters whose `last_seen_at` is older than `unnamed_retention_days` (60), including sightings and MinIO crops. Named clusters are exempt.

**Naming is privileged:**
The API exposes `/api/v1/visitors/clusters/{id}/name`. Naming is the single privileged transition: it moves biometric data from the visitor dataset into the enrollment dataset. Enrollment gains entries ONLY via explicit enroll calls or explicit operator naming of a visitor cluster; never automatically.

---

## 7. API surface

Domain endpoints are under `/api/v1`. Health and metrics are at the root. There is no authentication (LAN-internal service; Cognitive Companion is the BFF and enforces auth upstream).

### Health and metrics

| Method | Path | Response |
| --- | --- | --- |
| `GET` | `/health` | `status`, `inference_backend`, `triton_endpoint`, `model_profile`, `enrolled_members`, `model`, `models`, and `calibration` (`status`, `artifact_version`, `arcface_model_version`, `preprocessing_version`) |
| `GET` | `/metrics` | Prometheus text format |

### Enrollment

| Method | Path | Request Body | Response |
| --- | --- | --- | --- |
| `POST` | `/api/v1/enroll` | `EnrollRequest` (base64 images) | `EnrollResult` |
| `POST` | `/api/v1/enroll/upload/{person_id}` | Multipart: `name` + `files` | `EnrollResult` |
| `GET` | `/api/v1/members` | -- | `MemberListResponse` |
| `GET` | `/api/v1/members/{person_id}` | -- | `MemberInfo` (404 if missing) |
| `DELETE` | `/api/v1/members/{person_id}` | -- | `{"deleted": true, "person_id": str}` (404 if missing) |

Enroll error handling:
- Returns 400 if no valid images provided
- Returns 422 if no faces detected in any image
- `EnrollResult.failed_images` lists indices of images where face detection failed
- `EnrollResult.status`: `"enrolled"` for new members, `"updated"` for existing

### Identification

| Method | Path | Request Body | Response |
| --- | --- | --- | --- |
| `POST` | `/api/v1/identify` | `IdentifyRequest` | `IdentifyResponse` |
| `POST` | `/api/v1/identify-batch` | `BatchIdentifyRequest` | `BatchIdentifyResponse` |

`IdentifyRequest` fields:
- `image`: base64-encoded JPEG/PNG (with or without `data:` URI prefix)
- `include_annotated_image`: bool (default false)
- `save_guest_images`: bool (default false)

`BatchIdentifyRequest` fields:
- `images`: list of base64 strings (min 1)
- `include_motion`: bool (default true)
- `include_annotated_image`: bool (default false)
- `save_guest_images`: bool (default false)

Each `FaceDetection` carries `person_id`, `name`, `recognition_state`, `best_candidate_id`, `similarity` (the raw cosine match score, never a probability) and its one-release alias `raw_similarity`, nullable `calibrated_confidence` with `calibration_status` and version fields, head pose, `det_score`, and face geometry (`face_width_px`, `face_height_px`, nullable `interocular_px`). `IdentifyResponse` and each `FrameResult` carry `image_width_px` and `image_height_px` of the decoded input (0 for an undecodable batch frame).

**Side effect:** every unknown face on either identify endpoint is offered to `VisitorStore.record_sighting` (gated by `visitors.clustering_enabled`), independent of `save_guest_images`. Read-only measurement tools must call `FaceEngine` directly instead of these endpoints.

Batch endpoint handles per-image errors gracefully: bad frames are skipped and the response carries an empty `faces` list for them.

### Visitors

| Method | Path | Response |
| --- | --- | --- |
| `GET` | `/api/v1/visitors/clusters` | Cluster list (optional `status` filter); 409 when clustering is disabled |
| `GET` | `/api/v1/visitors/clusters/{cluster_id}` | Cluster detail with recent sightings |
| `POST` | `/api/v1/visitors/clusters/{cluster_id}/name` | Names the cluster into an enrolled member (the only privileged transition) |
| `POST` | `/api/v1/visitors/clusters/{cluster_id}/dismiss` | Dismisses a cluster |
| `POST` | `/api/v1/visitors/clusters/{cluster_a}/merge/{cluster_b}` | Merges two clusters |

### Motion

| Method | Path | Request Body | Response |
| --- | --- | --- | --- |
| `POST` | `/api/v1/detect-motion` | `MotionDetectionRequest` (min 2 images) | `MotionDetectionResponse` |

Standalone motion detection without enrollment lookups. Includes full trajectory data per track.

---

## 8. Database schema

`migrations/0001_initial_schema.up.sql` is the frozen baseline. It creates the tables below plus the visitor clustering tables `visitor_clusters` and `visitor_sightings` (hypertable on `seen_at`). Migrations are incremental from 2026-10-20: every change is a new `NNNN_description.up.sql` with a matching `.down.sql`; never edit `0001` and never drop the database to apply a change.

Core tables:

```sql
CREATE EXTENSION IF NOT EXISTS vector;       -- pgvector
CREATE EXTENSION IF NOT EXISTS vectorscale;  -- pgvectorscale (DiskANN)
CREATE EXTENSION IF NOT EXISTS timescaledb;  -- hypertables

CREATE TABLE members (
    person_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE embeddings (
    id BIGINT GENERATED ALWAYS AS IDENTITY,
    person_id TEXT REFERENCES members(person_id) ON DELETE CASCADE,
    embedding vector(512),
    created_at TIMESTAMPTZ DEFAULT now()
);
SELECT create_hypertable('embeddings', 'created_at');

CREATE TABLE centroids (
    person_id TEXT PRIMARY KEY REFERENCES members(person_id) ON DELETE CASCADE,
    centroid vector(512),
    updated_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX idx_centroids_diskann ON centroids
    USING diskann (centroid vector_cosine_ops);

CREATE TABLE guest_visits (
    id BIGINT GENERATED ALWAYS AS IDENTITY,
    guest_count INT,
    object_name TEXT,
    frame_index INT DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT now()
);
SELECT create_hypertable('guest_visits', 'created_at');
```

Key design decisions:
- `embeddings` and `guest_visits` are TimescaleDB hypertables for automatic time-based partitioning.
- `centroids` uses a DiskANN index for approximate nearest-neighbor search (pgvectorscale).
- `members` is a regular table (expected to be small, < 20 rows for a household).
- CASCADE deletes: removing a member removes all their embeddings and centroid.

### Migration system

`app/db/migrate.py` is a lightweight migration runner (not Alembic-based):

1. Creates `alembic_version` tracking table if not present (Alembic-compatible convention).
2. Scans `migrations/` for `*.up.sql` files, sorted lexically.
3. Compares against applied versions in `alembic_version`.
4. Applies each pending migration in its own transaction.
5. Records the version number on success.

Each migration filename must start with a zero-padded sequence number (e.g., `0001_`).
Every run, rollback, and status read holds a Postgres advisory lock
(`pg_advisory_lock(hashtext('person_id_migrations'))`), so replicas starting together apply each
file once. Every migration after the frozen `0001_initial_schema` ships a `.down.sql`
(`tests/test_migration_runner.py` enforces it). `python -m app.db.migrate rollback --steps N`
reverts the newest N, checking every target first: it refuses the baseline and any version
without a down file, changing nothing. `status` also lists versions recorded in `alembic_version`
whose files no longer exist (for example the squashed `0002_visitor_clusters`).

---

## 9. Testing conventions

Framework: `pytest` + `pytest-asyncio` (`asyncio_mode = "auto"`).

Tests mirror functionality but live under `tests/` at the repo root (not `app/tests/`).

### Patterns

| What you test | Pattern |
| --- | --- |
| `EnrollmentStore` | Fast tests over `InMemoryEnrollmentRepository` (`tests/test_enrollment_store_inmemory.py`); Postgres behavior through a real pool (`tests/test_enrollment_store.py`); `tests/test_identify_characterization.py` freezes enroll and identify output in a golden file |
| Repository with an in-memory peer | A parity test runs the same operations on both and compares results (`tests/test_enrollment_repository_parity.py`) |
| Matching policy | Table-driven boundaries (`tests/test_match_policy.py`) |
| Migration runner | Probe migrations in a temporary directory against the test database (`tests/test_migration_runner.py`) |
| `GuestImageStore` | Real asyncpg pool; `_FakeMinioClient` (in-memory dict with `upload_bytes`, `generate_presigned_url`); skips if DB unavailable |
| Dependency compatibility | `importlib.metadata.version()` to check scikit-image version constraint |
| Pure policy and geometry (`face_geometry`, `face_quality`, calibration evaluator and fitter) | Table-driven unit tests with no database (`tests/test_face_geometry.py`, `tests/test_face_quality.py`, `tests/test_m10_calibration.py`) |
| Wire contracts | Router mounted on a bare `FastAPI` app with fakes on `app.state` and FastAPI's `TestClient` (`tests/test_identification_geometry_contract.py`); a real `EnrollmentStore` runs over an in-memory gallery seeded with one centroid (`tests/gallery.py::one_member_store`) |
| `VisitorStore` and visitor ingest | Real asyncpg pool (`tests/test_visitor_store.py`, `tests/test_identification_visitor_ingest.py`, `tests/test_visitors_router.py`) |

Test files: `test_confidence_contract_characterization.py`, `test_database_guard.py`, `test_dependencies.py`, `test_enrollment_repository_parity.py`, `test_enrollment_store.py`, `test_enrollment_store_inmemory.py`, `test_identify_characterization.py`, `test_match_policy.py`, `test_migration_runner.py`, `test_face_engine.py`, `test_face_geometry.py`, `test_face_quality.py`, `test_guest_store.py`, `test_identification_geometry_contract.py`, `test_identification_quality_contract.py`, `test_quality_breakpoints.py`, `test_identification_visitor_ingest.py`, `test_m10_calibration.py`, `test_m3_face_evidence.py`, `test_visitors_router.py`, `test_visitor_store.py`; calibration fixtures under `tests/fixtures/identity_integrity/`.

### Fixtures

There is no shared `conftest.py`. Database tests take their connection string only from
`tests/db.py::safe_test_dsn()`: it reads `TEST_DATABASE_URL` (never `DATABASE_URL`), returns
None when unset (the tests skip), and raises for any database whose name does not end in
`_test`. Database teardowns delete every row they touch; on 2026-10-20 an earlier default to the
service's own database deleted the household's enrollment.

```python
TEST_DSN = safe_test_dsn()

@pytest_asyncio.fixture
async def pool():
    if TEST_DSN is None:
        pytest.skip("TEST_DATABASE_URL not set")
    p = await asyncpg.create_pool(TEST_DSN, init=_register_vector)
    yield p
    await p.close()
```

### What NOT to do in tests

- Do not mock the database; use a real asyncpg pool against a test database.
- Never point database tests at a database that holds household data, and never add a default
  connection string to a test.
- Do not mutate class-level properties with `type(obj).prop = ...`; use local subclasses.
- Do not use the production MinIO client; use `_FakeMinioClient`.
- Do not load the real InsightFace model; use `_FakeFaceEngine` with synthetic embeddings.

---

## 10. Common tasks

### 10.1 Add an API endpoint

1. Add or extend Pydantic models in `app/models/<domain>.py`.
2. Add the route handler in `app/routers/<domain>.py`.
3. The router is already registered in `app/main.py` `create_app()`. If adding a new router file, register it there.
4. Add a test under `tests/`.

### 10.2 Add a configuration setting

1. Add the key to `config/settings.yaml` with a sensible default or `${ENV_VAR}` placeholder.
2. Access it via `config.get("dotted.path", default)` in the service that uses it.
3. If the setting has an env var override, document it in the config file comment and in README.md.

### 10.3 Add a database migration

1. Create a new SQL file in `migrations/` with the next sequence number (e.g., `0002_your_change.up.sql`) and a matching `0002_your_change.down.sql` that exactly reverses it.
2. Write idempotent DDL (use `IF NOT EXISTS`, `IF EXISTS` where appropriate).
3. The migration runs automatically at next startup via the lifespan. Never edit `0001_initial_schema` and never drop the database to apply a change: enrollment and visitor data are irreplaceable.

### 10.4 Upgrade the Buffalo_L models

1. Export the full models into `continuous-tracking/triton-models`.
2. Generate and validate the INT8 variants in `triton-models-jetson`.
3. The embedding dimension may change (buffalo_l uses 512-dim ArcFace). If it changes, you need a new migration to ALTER the vector column dimension and rebuild the DiskANN index.
4. Re-enroll all members (embeddings from the old model are incompatible).

---

## 11. External services

| Service | Env var | Required | Purpose |
| --- | --- | --- | --- |
| PostgreSQL (TimescaleDB + pgvector) | `DATABASE_URL` | Required | Face gallery, embeddings, centroids, guest visit records |
| MinIO (S3-compatible) | `MINIO_ENDPOINT`, `MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY`, `MINIO_BUCKET` | Required | Guest image object storage |
| Triton Inference Server | `TRITON_GRPC_URL` | Required | Buffalo_L model execution |

This service does not call other Cognitive Companion microservices. It is called BY the Cognitive Companion backend (as the BFF for person identification).

---

## 12. What NOT to do

**Logging and console.**
- Do not use `print()`. Use `logging.getLogger(__name__)`.
- Do not log base64 image data (will flood logs).

**Architecture and layering.**
- Do not instantiate services in routers. Read from `request.app.state`.
- Do not add a local inference path or silently skip an unavailable model.
- Do not store embeddings in local `.npy` files. Use the pgvector `embeddings` and `centroids` tables.
- Do not save guest images to the local filesystem. Use MinIO via `GuestImageStore`.

**Database.**
- Do not run migrations by hand. Migrations run automatically in the lifespan.
- Do not use raw `asyncpg` connections without the `pgvector` vector codec registered. The pool factory in `main.py` handles this.
- Do not embed secret values in SQL migration files.

**Config and secrets.**
- Do not hardcode thresholds. Use `config.get()`.
- Do not store secrets in `config/settings.yaml`. Use `${ENV_VAR}` interpolation.

**Dependencies.**
- Do not add a runtime dependency without updating `pyproject.toml` and running `uv lock`.
- Do not import `torch` or a local inference runtime into the production service.

**Tests.**
- Do not mock the database. Use a real asyncpg pool.
- Do not call a live Triton server in unit tests. Inject a protocol-compatible fake.

**Documentation.**
- Do not write em-dashes in `.md` files. Use colons, commas, semicolons.

---

## 13. Where to look when stuck

| You want to ... | Read |
| --- | --- |
| Understand startup wiring | `app/main.py` (lifespan) |
| Debug face detection | `app/services/face_engine.py` |
| Debug identification accuracy | `app/services/enrollment_store.py` (thresholds, centroid query) |
| Measure face resolution | `app/services/face_geometry.py`; per-camera reports come from continuous-tracking `scripts/face_resolution_report.py` |
| Debug calibration state | `app/calibration/evaluator.py`, `GET /health` (`calibration` block) |
| Debug visitor clustering | `app/services/visitor_store.py` |
| Debug motion detection | `app/services/motion_detector.py` (_classify_direction) |
| Debug guest image uploads | `app/services/guest_store.py` + `app/services/minio_client.py` |
| Find configuration values | `config/settings.yaml` |
| Understand the database schema | `migrations/0001_initial_schema.up.sql` |
| See how the pool is created | `app/main.py` (_init_pool) |
| See test patterns | `tests/test_enrollment_store.py` |
