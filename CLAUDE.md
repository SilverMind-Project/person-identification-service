# Person Identification Service -- Claude guidance

GPU-accelerated face recognition and motion direction detection microservice for the Cognitive Companion system. Python 3.12 FastAPI, InsightFace (SCRFD + ArcFace), ONNX Runtime with CUDA.

---

## Commands

```bash
# Run (development)
uv run uvicorn app.main:app --host 0.0.0.0 --port 8100 --reload

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

# Docker
docker build -t person-id-service .
docker run --gpus all -p 8100:8100 -v $(pwd)/data:/app/data person-id-service

# Docker Compose
docker compose up -d
```

---

## Architecture

```
app/
  main.py              # App factory, lifespan (FaceEngine, EnrollmentStore, MotionDetector init)
  config.py            # YAML config loader with ${ENV_VAR} interpolation; dot-notation access
  models/
    enrollment.py      # EnrollRequest, EnrollResult, MemberInfo, MemberListResponse
    identification.py  # FaceDetection, IdentifyRequest/Response, BatchIdentifyRequest/Response, PersonMotion
    motion.py          # MotionDetectionRequest/Response, PersonTrack, TrajectoryPoint
  routers/
    health.py          # GET /health
    enrollment.py      # POST /enroll, POST /enroll/upload/{id}, GET/DELETE /members
    identification.py  # POST /identify, POST /identify-batch
    motion.py          # POST /detect-motion
  services/
    face_engine.py     # InsightFace FaceAnalysis wrapper: detect_faces(), compute_similarity()
    enrollment_store.py # SQLite metadata + .npy gallery: enroll(), identify(), remove_member()
    motion_detector.py # Cross-frame centroid tracking + direction classification
    guest_store.py     # Saves unidentified-person frames to disk
    image_annotator.py # Draws bounding boxes + name/confidence labels on images
config/
  settings.yaml        # All settings (${ENV_VAR} interpolation)
```

**Service lifecycle** (in `app.main` lifespan):
1. Initialize `FaceEngine` (loads InsightFace `buffalo_l` model into GPU memory)
2. Initialize `EnrollmentStore` (loads all centroids into memory for O(1) lookups)
3. Initialize `MotionDetector`, `GuestImageStore`
4. Store all in `app.state` for router access

**Service injection**: Services are attached to `app.state` in the lifespan. Routers access them via `request.app.state.<service>`.

---

## Key Services

### FaceEngine

Wraps InsightFace's `FaceAnalysis` for face detection (SCRFD) and 512-dim ArcFace embedding extraction.

- `detect_faces(image)`: returns list of `DetectedFace` with bbox, embedding, confidence, landmarks
- `compute_similarity(a, b)`: cosine similarity between two normalized embeddings
- Supports CUDA GPU with CPU fallback (`ctx_id=-1`)

### EnrollmentStore

Manages the face gallery: SQLite metadata + `.npy` embedding files.

- On startup, loads all centroids into memory
- `enroll(person_id, name, images)`: detects faces, saves embeddings as `.npy`, computes normalized centroid (mean of all embeddings)
- `identify(embedding)`: cosine similarity against each stored centroid; classified as "unknown" if below threshold (0.4 default)
- `remove_member(person_id)`: deletes all embeddings and centroid for a person

### MotionDetector

Tracks persons across frames by computing trajectory from face centroid positions.

- Classifies direction: `left-to-right`, `right-to-left`, `towards-camera`, `away-from-camera`, `stationary`
- For unknown faces, links across frames by embedding similarity (`cross_frame_similarity: 0.5`)
- Depth proxy uses face area change (15% threshold)
- Horizontal uses displacement fraction (5% threshold)

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
|---------|---------|---------|-------------|
| `face_engine.model_name` | `PERSON_ID_MODEL` | `buffalo_l` | InsightFace model pack |
| `face_engine.ctx_id` | `CUDA_DEVICE_ID` | `0` | GPU device index (-1 = CPU) |
| `face_engine.det_threshold` | `DETECTION_THRESHOLD` | `0.5` | Face detection confidence |
| `recognition.threshold` | `RECOGNITION_THRESHOLD` | `0.4` | Cosine similarity for positive ID |
| `recognition.unknown_threshold` | -- | `0.25` | Below this = definitely unknown |
| `storage.db_path` | -- | `data/face_db.sqlite` | SQLite database path |
| `storage.embeddings_dir` | -- | `data/embeddings` | Embedding storage directory |
| `storage.guest_images_dir` | -- | `data/guests` | Guest image storage |

---

## API Endpoints

All under `/api/v1` prefix:

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check (GPU status, enrolled count) |
| `POST` | `/enroll` | Enroll member with base64 images |
| `POST` | `/enroll/upload/{id}` | Enroll via multipart file upload |
| `GET` | `/members` | List all enrolled members |
| `GET` | `/members/{id}` | Get member details |
| `DELETE` | `/members/{id}` | Remove member |
| `POST` | `/identify` | Identify faces in single image |
| `POST` | `/identify-batch` | Batch identify + motion detection |
| `POST` | `/detect-motion` | Standalone motion direction detection |

---

## Data Storage

```text
data/
  face_db.sqlite              # Enrollment metadata
  embeddings/
    grandma/
      centroid.npy            # Mean embedding (fast identification)
      embedding_0.npy         # Individual embeddings
      ...
  guests/
    2026-03-23/
      143022-123456_f0_2guests.jpg
      ...
```

---

## Code Style

- **Python**: ruff with `E`, `F`, `I`, `W`, `UP`, `B`, `SIM`, `RUF`, `PIE`, `PT`, `C4`, `T20` rules. mypy for type checking. 100-char line length.
- **No em-dashes** ( - ) in any `.md` file. Use colons, periods, semicolons, or commas instead.
- Use `get_logger()` from Python stdlib. Never `print()`.
- Follow existing patterns; no premature abstractions.

---

## Do NOT

- **Use `print()`** -- use `logging`
- **Hardcode thresholds** -- read from config
- **Import torch** -- uses ONNX Runtime, not PyTorch at inference time
- **Store embeddings in SQLite** -- use `.npy` files for binary embedding data
- **Use em-dashes in documentation** -- use colons, commas, or semicolons
