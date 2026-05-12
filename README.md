# Person Identification Service

GPU-accelerated face recognition and motion direction detection microservice for the [Cognitive Companion](https://silvermind-project.github.io) system. Identifies household members in camera images and detects movement direction at doorways.

Documentation: [silvermind-project.github.io](https://silvermind-project.github.io). Agent reference: [AGENTS.md](AGENTS.md). Agent quick-start: [CLAUDE.md](CLAUDE.md).

---

## Architecture

- **Face Detection**: SCRFD (via InsightFace `buffalo_l` model pack)
- **Face Recognition**: ArcFace 512-dimensional embeddings with cosine similarity matching
- **Motion Detection**: Cross-frame centroid tracking with face re-identification
- **Runtime**: ONNX Runtime with CUDA execution provider
- **Storage**: PostgreSQL (TimescaleDB + pgvector) for face gallery; MinIO for guest images

```mermaid
flowchart LR
    CC["Cognitive Companion<br/>(BFF)"] -->|"POST /identify-batch"| PersonID["Person ID Service<br/>(FastAPI :8200)"]
    PersonID --> GPU["GPU<br/>InsightFace SCRFD + ArcFace"]
    PersonID --> DB["PostgreSQL<br/>TimescaleDB + pgvector"]
    PersonID --> MinIO["MinIO<br/>guest images"]
```

---

## Requirements

| Component | Purpose |
| --- | --- |
| NVIDIA GPU (10 GB+ VRAM) | Face detection and recognition inference |
| CUDA 12.x or later | GPU driver and libraries |
| Python 3.12 | Runtime |
| PostgreSQL (TimescaleDB + pgvector) | Face gallery, embeddings, centroids |
| MinIO (S3-compatible) | Guest image object storage |
| Docker + NVIDIA Container Toolkit | Containerized deployment |

---

## Quick Start

### Docker (recommended)

```bash
# Build
docker build -t person-id-service .

# Run with GPU access and model data volume
docker run --gpus all -p 8200:8200 \
  -e DATABASE_URL=postgresql://user:pass@host:5432/db \
  -e MINIO_ENDPOINT=host:9000 \
  -e MINIO_ACCESS_KEY=minioadmin \
  -e MINIO_SECRET_KEY=minioadmin \
  -e MINIO_BUCKET=cognitive-companion \
  -v $(pwd)/data:/app/data \
  person-id-service

# Verify
curl http://localhost:8200/health
```

### Docker Compose

```bash
cp .env.example .env   # set DATABASE_URL and MinIO credentials
docker compose up -d
```

### Local Development

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# GPU build
uv sync

# CPU-only (development/testing without a GPU)
uv sync --extra cpu

# Set required environment variables
export DATABASE_URL=postgresql://user:pass@localhost:5432/cognitive_companion
export MINIO_ENDPOINT=localhost:9000
export MINIO_ACCESS_KEY=minioadmin
export MINIO_SECRET_KEY=minioadmin

# Run
uv run uvicorn app.main:app --host 0.0.0.0 --port 8200 --reload
```

---

## Configuration

All settings in `config/settings.yaml` with `${ENV_VAR:default}` interpolation. Override at runtime via environment variables.

| Setting | Env var | Default | Description |
| --- | --- | --- | --- |
| `face_engine.model_name` | `PERSON_ID_MODEL` | `buffalo_l` | InsightFace model pack |
| `face_engine.ctx_id` | `CUDA_DEVICE_ID` | `0` | GPU device index (`-1` for CPU) |
| `face_engine.det_threshold` | `DETECTION_THRESHOLD` | `0.6` | Face detection confidence |
| `recognition.threshold` | `RECOGNITION_THRESHOLD` | `0.4` | Cosine similarity for positive ID |
| `recognition.unknown_threshold` | -- | `0.25` | Below this = definitely unknown |
| `database.dsn` | `DATABASE_URL` | -- | PostgreSQL connection string (required) |
| `minio.endpoint` | `MINIO_ENDPOINT` | `localhost:9000` | MinIO/S3 endpoint |
| `minio.access_key` | `MINIO_ACCESS_KEY` | `minioadmin` | MinIO access key |
| `minio.secret_key` | `MINIO_SECRET_KEY` | `minioadmin` | MinIO secret key |
| `minio.bucket` | `MINIO_BUCKET` | `cognitive-companion` | S3 bucket name |
| `logging.level` | `LOG_LEVEL` | `INFO` | Log verbosity |

Additional tuning options (edit `config/settings.yaml` directly):

| Setting | Default | Description |
| --- | --- | --- |
| `face_engine.det_size` | `[640, 640]` | Detection input resolution |
| `motion.min_displacement_fraction` | `0.05` | Min displacement (% of frame) for movement |
| `motion.cross_frame_similarity` | `0.5` | Unknown-face cross-frame similarity threshold |
| `annotation.box_color_known` | `[0, 200, 0]` | BGR color for known person boxes |
| `annotation.box_color_unknown` | `[0, 165, 255]` | BGR color for unknown person boxes |

---

## API Endpoints

All under `/api/v1`:

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/health` | Health check (GPU status, enrolled count) |
| `POST` | `/enroll` | Enroll member with base64 images |
| `POST` | `/enroll/upload/{person_id}` | Enroll via multipart file upload |
| `GET` | `/members` | List all enrolled members |
| `GET` | `/members/{person_id}` | Get member details |
| `DELETE` | `/members/{person_id}` | Remove member |
| `POST` | `/identify` | Identify faces in a single image |
| `POST` | `/identify-batch` | Batch identify + motion detection |
| `POST` | `/detect-motion` | Standalone motion direction detection |

Full reference: [silvermind-project.github.io/api/reference](https://silvermind-project.github.io/api/reference).

### Enroll a Member

```bash
IMG1=$(base64 -w0 photo1.jpg)
IMG2=$(base64 -w0 photo2.jpg)

curl -X POST http://localhost:8200/api/v1/enroll \
  -H "Content-Type: application/json" \
  -d "{\"person_id\": \"grandma\", \"name\": \"Grandma\", \"images\": [\"$IMG1\", \"$IMG2\"]}"
```

Response:

```json
{
  "person_id": "grandma",
  "name": "Grandma",
  "embedding_count": 2,
  "status": "enrolled",
  "failed_images": []
}
```

For file upload: `POST /api/v1/enroll/upload/{person_id}` with multipart `name` and `files` fields.

### Identify Faces in an Image

```bash
IMG=$(base64 -w0 camera_snapshot.jpg)

curl -X POST http://localhost:8200/api/v1/identify \
  -H "Content-Type: application/json" \
  -d "{\"image\": \"$IMG\"}"
```

Response:

```json
{
  "faces": [
    {"person_id": "grandma", "name": "Grandma", "confidence": 0.87, "bbox": [120, 80, 250, 310]}
  ],
  "annotated_image": null
}
```

### Batch Identification with Motion

This is the primary endpoint used by the Cognitive Companion backend:

```bash
curl -X POST http://localhost:8200/api/v1/identify-batch \
  -H "Content-Type: application/json" \
  -d "{\"images\": [\"$IMG1\", \"$IMG2\", \"$IMG3\"], \"include_motion\": true}"
```

Response includes per-frame detections, motion direction per person, and optional annotated images.

### Motion Direction Detection

| Direction | Meaning | Detection |
| --- | --- | --- |
| `left-to-right` | Moving rightward | Horizontal centroid displacement |
| `right-to-left` | Moving leftward | Horizontal centroid displacement |
| `towards-camera` | Approaching camera | Face area increasing |
| `away-from-camera` | Moving away | Face area decreasing |
| `stationary` | No movement | Below displacement threshold |

---

## Data Storage

```text
data/
  models/buffalo_l/
    det_10g.onnx            # SCRFD face detection (16.9 MB)
    w600k_r50.onnx          # ArcFace recognition 512-dim (174.4 MB)
    2d106det.onnx           # 2D landmark detection (5.0 MB)
    1k3d68.onnx             # 3D landmark estimation (143.6 MB)
    genderage.onnx          # Gender/age estimation (1.3 MB)
```

Face embeddings and centroids are stored in PostgreSQL (pgvector). Guest images are uploaded to MinIO. The `data/` directory contains only ONNX model files.

---

## Code Quality

```bash
uv run ruff check .               # Lint
uv run ruff format --check .      # Format check
uv run mypy app/                  # Type check
uv run pytest                     # Tests
```

---

## License

AGPL-3.0-or-later
