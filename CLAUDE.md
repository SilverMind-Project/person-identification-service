# Person Identification Service -- Claude guidance

GPU-accelerated face recognition and motion direction detection microservice for the Cognitive Companion system. Python 3.12 FastAPI, InsightFace (SCRFD + ArcFace), ONNX Runtime with CUDA, PostgreSQL (TimescaleDB + pgvector), MinIO.

---

## Read before editing

1. [AGENTS.md](AGENTS.md): canonical reference (architecture, data flow, testing conventions, common tasks, anti-patterns).
2. `app/main.py` lifespan: source of truth for service initialization and `app.state` keys.
3. `config/settings.yaml`: every tunable, plus env-var-backed secrets.
4. `migrations/0001_initial_schema.up.sql`: current database schema (TimescaleDB hypertables, pgvector DiskANN index).

---

## Commands

```bash
# Run (development)
uv run uvicorn app.main:app --host 0.0.0.0 --port 8200 --reload

# GPU build (default)
uv sync

# CPU-only (development/testing without GPU)
uv sync --extra cpu

# Lint
uv run ruff check .

# Format
uv run ruff format .

# Type check
uv run mypy app/

# Tests
uv run pytest

# Targeted iteration
uv run pytest tests/test_enrollment_store.py -v
uv run pytest tests/test_guest_store.py -v

# Docker
docker build -t person-id-service .
docker run --gpus all -p 8200:8200 -v $(pwd)/data:/app/data person-id-service

# Docker Compose
docker compose up -d
```

---

## Architecture

```text
app/
  main.py              # App factory, lifespan (pool, FaceEngine, EnrollmentStore, MotionDetector, GuestImageStore init)
  config.py            # YAML config loader with ${ENV_VAR:default} interpolation; dot-notation access
  db/
    migrate.py         # asyncpg migration runner (scans migrations/*.up.sql, tracks in alembic_version)
  models/
    enrollment.py      # EnrollRequest, EnrollResult, MemberInfo, MemberListResponse
    identification.py  # FaceDetection, IdentifyRequest/Response, BatchIdentifyRequest/Response, PersonMotion, FrameResult
    motion.py          # MotionDetectionRequest/Response, PersonTrack, TrajectoryPoint
  routers/
    health.py          # GET /health (GPU status, enrolled count, model name)
    enrollment.py      # POST /enroll, POST /enroll/upload/{id}, GET/DELETE /members/{id}
    identification.py  # POST /identify, POST /identify-batch
    motion.py          # POST /detect-motion
  services/
    face_engine.py     # InsightFace FaceAnalysis wrapper: detect_faces(), compute_similarity()
    face_models.py     # DetectedFace, IdentifyResult dataclasses (no InsightFace dependency)
    enrollment_store.py # pgvector gallery: enroll(), identify(), identify_all(), list_members(), remove_member()
    minio_client.py    # boto3 S3 wrapper: upload_bytes(), generate_presigned_url(), delete_object()
    motion_detector.py # Cross-frame centroid tracking + direction classification
    guest_store.py     # Saves unidentified-person frames to MinIO + guest_visits hypertable
    image_annotator.py # Draws bounding boxes + name/confidence labels on images
config/
  settings.yaml        # All settings (${ENV_VAR} interpolation)
migrations/
  0001_initial_schema.up.sql  # TimescaleDB + pgvector schema
```

**Service lifecycle** (in `app.main` lifespan):

1. Configure logging from `config.get("logging.level")`
2. Create asyncpg pool from `config.get("database.dsn")` with pgvector codec registered
3. Run pending migrations via `app.db.migrate.run_migrations(pool)`
4. Create MinIO client, ensure bucket exists
5. Initialize `FaceEngine` in a threadpool (InsightFace `buffalo_l` model loaded to GPU)
6. Initialize `EnrollmentStore(pool, face_engine)`, `MotionDetector()`, `GuestImageStore(pool, minio_client)`
7. Store all services on `app.state` (face_engine, enrollment_store, motion_detector, guest_store)
8. On shutdown, close the asyncpg pool

**Service injection**: Services are attached to `app.state` in the lifespan. Routers access them via `request.app.state.<service>`.

---

## Key Services

### FaceEngine

Wraps InsightFace `FaceAnalysis` with `CUDAExecutionProvider` only.

- `detect_faces(image) -> list[DetectedFace]`: SCRFD detection + ArcFace 512-dim embedding extraction. Filters by `face_engine.det_threshold` (default 0.6).
- `compute_similarity(emb1, emb2) -> float`: static method, cosine similarity via `np.dot`.
- GPU availability detected via `onnxruntime.get_available_providers()`.

### EnrollmentStore

Manages face gallery in PostgreSQL with pgvector.

- `enroll(person_id, name, images) -> EnrollResult`: Detects largest face per image via threadpool, INSERTs member + embeddings, recomputes normalized centroid (mean of all embeddings), UPSERTs into `centroids` table.
- `identify(embedding) -> IdentifyResult`: Nearest-centroid query via pgvector `<=>` (cosine distance), `LIMIT 1`. Returns "unknown"/"Guest" if below `recognition.threshold` (default 0.4).
- `identify_all(faces) -> list[IdentifyResult]`: Batch identify, carries forward bbox.
- `list_members() -> list[MemberInfo]`: LEFT JOIN with embedding count.
- `remove_member(person_id) -> bool`: DELETE from members (CASCADE handles embeddings + centroids).
- `member_count() -> int`: Simple COUNT.

### MotionDetector

Tracks persons across frames by computing trajectory from face centroid positions.

- `detect_direction(frame_shapes, frame_faces, frame_identities) -> list[PersonTrackResult]`: Groups faces by person_id, links unknown faces across frames via embedding similarity, classifies direction.
- `_classify_direction(trajectory, frame_shape) -> tuple[str, float]`: Uses horizontal displacement (5% threshold) and face area change (15% threshold). Returns `left-to-right`, `right-to-left`, `towards-camera`, `away-from-camera`, or `stationary`.
- `_link_unknowns(entries) -> dict`: Cross-frame unknown-face linking via cosine similarity (`cross_frame_similarity: 0.5`).

### GuestImageStore

Saves unidentified-person frames to MinIO and records visits in the `guest_visits` hypertable.

- `save_guest_image(image, guest_count, frame_index) -> str | None`: Encodes frame as JPEG (quality 90), uploads to MinIO under `guests/{YYYY-MM-DD}/{HHMMSS-ffffff}_f{idx}_{n}guests.jpg`, INSERTs guest visit record. Returns object name on success, None on failure.

### MinioClient

boto3 S3 wrapper with path-style addressing and s3v4 signatures.

- `upload_bytes(data, object_name, content_type) -> str`: Uploads via `BytesIO`, returns presigned URL.
- `generate_presigned_url(object_name, expiration=3600) -> str`: Presigned GET URL.
- `delete_object(object_name)`: Deletes single object.
- Factory: `create_minio_client()` reads MinIO settings from config, creates client, ensures bucket exists.

---

## Database

PostgreSQL with TimescaleDB and pgvector extensions. Shared instance with other Cognitive Companion services.

Tables:

| Table | Type | Purpose |
| --- | --- | --- |
| `members` | Regular | person_id (PK), name, created_at |
| `embeddings` | Hypertable | id, person_id (FK), embedding vector(512), created_at |
| `centroids` | Regular | person_id (PK FK), centroid vector(512), updated_at; DiskANN index for ANN search |
| `guest_visits` | Hypertable | id, guest_count, object_name, frame_index, created_at |

Schema changes: SQL files in `migrations/` applied by `app.db.migrate.run_migrations()`. Uses an `alembic_version` tracking table (Alembic-compatible convention). Each migration runs in its own transaction.

In tests: create a real asyncpg pool connected to a test DB. The `EnrollmentStore` and `GuestImageStore` constructors accept the pool directly.

---

## Configuration

YAML config with environment variable overrides:

```python
from app.config import config
model = config.get("face_engine.model_name")  # "buffalo_l"
threshold = config.get("recognition.threshold", 0.4)
```

Key settings:

| Setting | Env var | Default | Description |
| --- | --- | --- | --- |
| `face_engine.model_name` | `PERSON_ID_MODEL` | `buffalo_l` | InsightFace model pack |
| `face_engine.ctx_id` | `CUDA_DEVICE_ID` | `0` | GPU device index (-1 = CPU) |
| `face_engine.det_threshold` | `DETECTION_THRESHOLD` | `0.6` | Face detection confidence |
| `recognition.threshold` | `RECOGNITION_THRESHOLD` | `0.4` | Cosine similarity for positive ID |
| `recognition.unknown_threshold` | -- | `0.25` | Below this = definitely unknown |
| `motion.min_displacement_fraction` | -- | `0.05` | Min displacement for movement |
| `motion.cross_frame_similarity` | -- | `0.5` | Unknown-face cross-frame linking |
| `database.dsn` | `DATABASE_URL` | -- | PostgreSQL connection string (required) |
| `minio.endpoint` | `MINIO_ENDPOINT` | `localhost:9000` | MinIO/S3 endpoint |
| `minio.access_key` | `MINIO_ACCESS_KEY` | `minioadmin` | MinIO access key |
| `minio.secret_key` | `MINIO_SECRET_KEY` | `minioadmin` | MinIO secret key |
| `minio.bucket` | `MINIO_BUCKET` | `cognitive-companion` | S3 bucket name |
| `minio.secure` | -- | `false` | Use HTTPS |

---

## API Endpoints

All under `/api/v1` prefix:

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/health` | Health check (GPU status, enrolled count, model) |
| `POST` | `/enroll` | Enroll member with base64 images |
| `POST` | `/enroll/upload/{person_id}` | Enroll via multipart file upload |
| `GET` | `/members` | List all enrolled members |
| `GET` | `/members/{person_id}` | Get member details |
| `DELETE` | `/members/{person_id}` | Remove member |
| `POST` | `/identify` | Identify faces in single image |
| `POST` | `/identify-batch` | Batch identify + optional motion detection |
| `POST` | `/detect-motion` | Standalone motion direction detection |

---

## Data Storage

```text
data/
  models/
    buffalo_l/            # ONNX model files (341 MB total)
      det_10g.onnx        # SCRFD face detection
      w600k_r50.onnx      # ArcFace recognition (512-dim)
      2d106det.onnx       # 2D landmark detection
      1k3d68.onnx         # 3D landmark estimation
      genderage.onnx      # Gender/age estimation
```

Face embeddings and centroids are stored in PostgreSQL (pgvector), not on disk. Guest images are uploaded to MinIO, not the local filesystem. The `data/` volume is only needed for ONNX model files.

---

## Code Style

- **Python**: ruff with `E`, `F`, `I`, `W`, `UP`, `B`, `SIM`, `RUF`, `PIE`, `PT`, `C4`, `T20` rules. mypy for type checking. 100-char line length.
- **No em-dashes** in any `.md` file. Use colons, periods, semicolons, or commas instead.
- Use stdlib `logging.getLogger()`. Never `print()`.
- Follow existing patterns; no premature abstractions.
- **Pydantic for wire models** (`BaseModel`), dataclasses for internal data transfer (`DetectedFace`, `IdentifyResult`, `PersonTrackResult`).

---

## Testing

Tests use `pytest` + `pytest-asyncio` (`asyncio_mode = auto`).

Patterns:

| What you test | Pattern |
| --- | --- |
| `EnrollmentStore` | Real asyncpg pool (skip if DB unavailable); `_FakeFaceEngine` returning synthetic 512-dim embeddings via `RandomState(42)` |
| `GuestImageStore` | Real asyncpg pool; `_FakeMinioClient` (in-memory dict) |
| Dependencies | `importlib.metadata.version()` to verify compatible versions |

Do not mock the database. Tests needing a database create a real asyncpg pool and skip if unavailable.

---

## Do NOT

- **Use `print()`** -- use `logging.getLogger()`
- **Hardcode thresholds** -- read from config
- **Import torch** -- uses ONNX Runtime, not PyTorch at inference time
- **Store embeddings in local files** -- embeddings live in PostgreSQL pgvector; guest images in MinIO
- **Use em-dashes in documentation** -- use colons, commas, or semicolons
- **Use bare `except:`** -- log and return a zero value or re-raise as an HTTP exception
- **Instantiate services in routers** -- read from `request.app.state`
- **Run migrations by hand** -- migrations run automatically in the lifespan via `run_migrations()`

---

## External Dependencies

| Service | Env var | Required |
| --- | --- | --- |
| PostgreSQL (TimescaleDB + pgvector) | `DATABASE_URL` | Required |
| MinIO (S3-compatible) | `MINIO_ENDPOINT`, `MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY`, `MINIO_BUCKET` | Required |
| NVIDIA GPU + CUDA 12+ | -- | Required for GPU inference |
