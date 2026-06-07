# AGENTS.md

Reference for AI coding agents working in `person-identification-service/`. This document is the canonical, deep guide. `CLAUDE.md` is a tight pointer aimed at the same audience; `README.md` is human-facing.

If a fact appears here, it traces to a file in this tree at the time of writing. Verify before relying on it: `git log` is authoritative for "what changed", and `grep` against `app/` is authoritative for "what exists".

---

## 1. Mission and scope

Person Identification Service is a Triton-backed face recognition and motion direction detection microservice for the Cognitive Companion system. It ingests base64-encoded camera frames via a REST API and returns identities, confidence scores, bounding boxes, and motion direction classifications.

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
│   ├── db/
│   │   └── migrate.py       # asyncpg migration runner
│   ├── models/
│   │   ├── enrollment.py    # EnrollRequest, EnrollResult, MemberInfo, MemberListResponse
│   │   ├── identification.py # FaceDetection, IdentifyRequest/Response, BatchIdentifyRequest/Response, PersonMotion, FrameResult
│   │   └── motion.py        # MotionDetectionRequest/Response, PersonTrack, TrajectoryPoint
│   ├── routers/
│   │   ├── health.py        # GET /health
│   │   ├── enrollment.py    # POST /enroll, POST /enroll/upload/{id}, GET/DELETE /members/{id}
│   │   ├── identification.py # POST /identify, POST /identify-batch
│   │   └── motion.py        # POST /detect-motion
│   └── services/
│       ├── face_engine.py     # Triton preprocessing, inference, and postprocessing
│       ├── face_models.py     # DetectedFace, IdentifyResult dataclasses
│       ├── enrollment_store.py # pgvector gallery management: enroll, identify, remove
│       ├── minio_client.py    # boto3 S3 wrapper for MinIO
│       ├── motion_detector.py # Cross-frame centroid tracking + direction classification
│       ├── guest_store.py     # MinIO uploads for unidentified-person frames
│       └── image_annotator.py # Bounding box + label drawing
├── config/
│   └── settings.yaml        # All application settings
├── data/
│   └── models/buffalo_l/    # ONNX model files (committed via Git LFS)
├── migrations/
│   └── 0001_initial_schema.up.sql  # TimescaleDB + pgvector tables
├── tests/
│   ├── test_enrollment_store.py
│   ├── test_guest_store.py
│   └── test_dependencies.py
├── pyproject.toml           # Build config, deps, tool configs
├── Dockerfile               # Accelerator-independent API build
├── docker-compose.yml       # API service definition
├── CLAUDE.md                # Agent quick-reference
└── README.md                # Human-facing documentation
```

---

## 4. Commands

Run from the repository root unless noted.

```bash
# Development server
uv run uvicorn app.main:app --host 0.0.0.0 --port 8200 --reload

uv sync

# Offline conversion and validation tools
uv sync --extra model-tools

# Code quality
uv run ruff check .              # Lint
uv run ruff format .             # Format
uv run mypy app/                 # Type check

# Tests
uv run pytest                    # Full suite
uv run pytest tests/test_enrollment_store.py -v   # Targeted

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
2. `_init_pool(dsn)` -- asyncpg pool with `pgvector` vector codec registered (`min_size=1, max_size=5`)
3. `run_migrations(pool)` -- applies pending `migrations/*.up.sql` files transactionally
4. `create_minio_client()` -- MinIO client with bucket-ensured
5. Open `TritonGrpcClient` and require all five configured models to be ready
6. `EnrollmentStore(pool, face_engine)`, `MotionDetector()`, `GuestImageStore(pool, minio_client)`
7. Store on `app.state`: `face_engine`, `enrollment_store`, `motion_detector`, `guest_store`
8. Log Triton endpoint/profile and member count
9. On shutdown: close Triton, then close the database pool

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

Manages the face gallery in PostgreSQL with pgvector. Replaces the old SQLite + `.npy` file system.

**Enroll flow:**
1. Check if person_id already exists (for "enrolled" vs "updated" status)
2. For each image: await `engine.detect_faces()`, then pick the largest face by bbox area
3. INSERT/UPDATE `members` row
4. INSERT all new embeddings into `embeddings` (TimescaleDB hypertable)
5. Compute normalized centroid: mean of all embeddings for that person, then L2-normalize
6. UPSERT into `centroids` table (DiskANN-indexed)

**Identify flow:**
```sql
SELECT c.person_id, m.name, 1 - (c.centroid <=> $1) AS confidence
FROM centroids c
JOIN members m ON c.person_id = m.person_id
ORDER BY c.centroid <=> $1
LIMIT 1
```

Uses pgvector `<=>` (cosine distance). DiskANN index on `centroids.centroid` provides ANN speedup for large galleries.

**Thresholds:**
- `recognition.threshold` (default 0.4): below this = classify as "unknown"
- `recognition.unknown_threshold` (default 0.25): below this = "definitely unknown" (currently informational only)

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

---

## 7. API surface

All endpoints under `/api/v1` prefix. No authentication (this is a LAN-internal service; the Cognitive Companion backend is the BFF and enforces auth upstream).

### Health

| Method | Path | Request | Response |
| --- | --- | --- | --- |
| `GET` | `/health` | -- | Triton endpoint/profile, model names, and enrolled count |

### Enrollment

| Method | Path | Request Body | Response |
| --- | --- | --- | --- |
| `POST` | `/enroll` | `EnrollRequest` (base64 images) | `EnrollResult` |
| `POST` | `/enroll/upload/{person_id}` | Multipart: `name` + `files` | `EnrollResult` |
| `GET` | `/members` | -- | `MemberListResponse` |
| `GET` | `/members/{person_id}` | -- | `MemberInfo` (404 if missing) |
| `DELETE` | `/members/{person_id}` | -- | `{"deleted": true, "person_id": str}` (404 if missing) |

Enroll error handling:
- Returns 400 if no valid images provided
- Returns 422 if no faces detected in any image
- `EnrollResult.failed_images` lists indices of images where face detection failed
- `EnrollResult.status`: `"enrolled"` for new members, `"updated"` for existing

### Identification

| Method | Path | Request Body | Response |
| --- | --- | --- | --- |
| `POST` | `/identify` | `IdentifyRequest` | `IdentifyResponse` |
| `POST` | `/identify-batch` | `BatchIdentifyRequest` | `BatchIdentifyResponse` |

`IdentifyRequest` fields:
- `image`: base64-encoded JPEG/PNG (with or without `data:` URI prefix)
- `include_annotated_image`: bool (default false)
- `save_guest_images`: bool (default false)

`BatchIdentifyRequest` fields:
- `images`: list of base64 strings (min 1)
- `include_motion`: bool (default true)
- `include_annotated_image`: bool (default false)
- `save_guest_images`: bool (default false)

Batch endpoint handles per-image errors gracefully: bad frames are skipped and the response carries only successful frames. This is the primary endpoint used by the Cognitive Companion event aggregator.

### Motion

| Method | Path | Request Body | Response |
| --- | --- | --- | --- |
| `POST` | `/detect-motion` | `MotionDetectionRequest` (min 2 images) | `MotionDetectionResponse` |

Standalone motion detection without enrollment lookups. Includes full trajectory data per track.

---

## 8. Database schema

`migrations/0001_initial_schema.up.sql` creates:

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

---

## 9. Testing conventions

Framework: `pytest` + `pytest-asyncio` (`asyncio_mode = "auto"`).

Tests mirror functionality but live under `tests/` at the repo root (not `app/tests/`).

### Patterns

| What you test | Pattern |
| --- | --- |
| `EnrollmentStore` | Real asyncpg pool from `asyncpg.create_pool()`; `_FakeFaceEngine` class returning deterministic embeddings via `np.random.RandomState(42)`; cleanup fixture deletes test data after each test |
| `GuestImageStore` | Real asyncpg pool; `_FakeMinioClient` (in-memory dict with `upload_bytes`, `generate_presigned_url`); skips if DB unavailable |
| Dependency compatibility | `importlib.metadata.version()` to check scikit-image version constraint |

### Fixtures

There is no shared `conftest.py`. Each test file constructs its own pool and services:

```python
@pytest.fixture
async def pool():
    dsn = os.environ.get("TEST_DATABASE_URL", "postgresql://...")
    p = await asyncpg.create_pool(dsn)
    yield p
    await p.close()
```

### What NOT to do in tests

- Do not mock the database; use a real asyncpg pool against a test database.
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

1. Create a new SQL file in `migrations/` with the next sequence number (e.g., `0002_your_change.up.sql`).
2. Write idempotent DDL (use `IF NOT EXISTS`, `IF EXISTS` where appropriate).
3. The migration runs automatically at next startup via the lifespan.

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
| Debug motion detection | `app/services/motion_detector.py` (_classify_direction) |
| Debug guest image uploads | `app/services/guest_store.py` + `app/services/minio_client.py` |
| Find configuration values | `config/settings.yaml` |
| Understand the database schema | `migrations/0001_initial_schema.up.sql` |
| See how the pool is created | `app/main.py` (_init_pool) |
| See test patterns | `tests/test_enrollment_store.py` |
