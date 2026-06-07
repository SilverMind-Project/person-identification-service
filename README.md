# Person Identification Service

Triton-backed face recognition and motion direction detection microservice for the [Cognitive Companion](https://silvermind-project.github.io) system. Identifies household members in camera images and detects movement direction at doorways.

Documentation: [silvermind-project.github.io](https://silvermind-project.github.io). Agent reference: [AGENTS.md](AGENTS.md). Agent quick-start: [CLAUDE.md](CLAUDE.md).

---

## Architecture

- **Face Detection**: SCRFD from the Buffalo_L model pack
- **Face Recognition**: ArcFace 512-dimensional embeddings with cosine similarity matching
- **Face Evidence**: 106-point landmarks, 68-point 3D landmarks, and gender/age attributes
- **Motion Detection**: Cross-frame centroid tracking with face re-identification
- **Runtime**: NVIDIA Triton Inference Server over gRPC
- **Storage**: PostgreSQL (TimescaleDB + pgvector) for face gallery; MinIO for guest images

```mermaid
flowchart LR
    CC["Cognitive Companion<br/>(BFF)"] -->|"POST /identify-batch"| PersonID["Person ID Service<br/>(FastAPI :8200)"]
    PersonID --> Triton["Triton<br/>Buffalo_L model pack"]
    PersonID --> DB["PostgreSQL<br/>TimescaleDB + pgvector"]
    PersonID --> MinIO["MinIO<br/>guest images"]
```

---

## Requirements

| Component | Purpose |
| --- | --- |
| Triton Inference Server | Serves all five Buffalo_L graphs |
| Python 3.12 | Runtime |
| PostgreSQL (TimescaleDB + pgvector) | Face gallery, embeddings, centroids |
| MinIO (S3-compatible) | Guest image object storage |
| Docker | Containerized API deployment |

---

## Quick Start

### Docker (recommended)

```bash
# Build
docker build -t person-id-service .

# Run against the default CTS Triton endpoint
docker run -p 8200:8200 \
  --network nanai \
  -e DATABASE_URL=postgresql://user:pass@host:5432/db \
  -e MINIO_ENDPOINT=host:9000 \
  -e MINIO_ACCESS_KEY=minioadmin \
  -e MINIO_SECRET_KEY=minioadmin \
  -e MINIO_BUCKET=cognitive-companion \
  -e TRITON_GRPC_URL=triton:8701 \
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

uv sync

# Offline model conversion and validation tools
uv sync --extra model-tools

# Set required environment variables
export DATABASE_URL=postgresql://user:pass@localhost:5432/cognitive_companion
export MINIO_ENDPOINT=localhost:9000
export MINIO_ACCESS_KEY=minioadmin
export MINIO_SECRET_KEY=minioadmin
export TRITON_GRPC_URL=localhost:8701

# Run
uv run uvicorn app.main:app --host 0.0.0.0 --port 8200 --reload
```

---

## Configuration

All settings in `config/settings.yaml` with `${ENV_VAR:default}` interpolation. Override at runtime via environment variables.

| Setting | Env var | Default | Description |
| --- | --- | --- | --- |
| `face_engine.triton_url` | `TRITON_GRPC_URL` | `triton:8701` | Triton gRPC endpoint |
| `face_engine.model_profile` | `PERSON_ID_MODEL_PROFILE` | `full` | Health metadata: `full` or `int8` |
| `face_engine.triton_timeout_ms` | `TRITON_TIMEOUT_MS` | `30000` | Per-request timeout |
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

## API endpoints

Application endpoints use `/api/v1`. The health endpoint is at `/health`.

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/health` | Triton endpoint, profile, model names, and enrolled count |
| `POST` | `/enroll` | Enroll member with base64 images |
| `POST` | `/enroll/upload/{person_id}` | Enroll via multipart file upload |
| `GET` | `/members` | List all enrolled members |
| `GET` | `/members/{person_id}` | Get member details |
| `DELETE` | `/members/{person_id}` | Remove member |
| `POST` | `/identify` | Identify faces in a single image |
| `POST` | `/identify-batch` | Batch identify + motion detection |
| `POST` | `/detect-motion` | Standalone motion direction detection |

Full reference: [silvermind-project.github.io/api/reference](https://silvermind-project.github.io/api/reference).

### Enroll a member

```bash
IMG1=$(base64 -w0 photo1.jpg)
IMG2=$(base64 -w0 photo2.jpg)

curl -X POST http://localhost:8200/api/v1/enroll \
  -H "Content-Type: application/json" \
  -d "{\"person_id\": \"resident-1\", \"name\": \"Resident\", \"images\": [\"$IMG1\", \"$IMG2\"]}"
```

Response:

```json
{
  "person_id": "resident-1",
  "name": "Resident",
  "embedding_count": 2,
  "status": "enrolled",
  "failed_images": []
}
```

For file upload: `POST /api/v1/enroll/upload/{person_id}` with multipart `name` and `files` fields.

### Identify faces in an image

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
    {"person_id": "resident-1", "name": "Resident", "confidence": 0.87, "bbox": [120, 80, 250, 310]}
  ],
  "annotated_image": null
}
```

### Batch identification with motion

This is the primary endpoint used by the Cognitive Companion backend:

```bash
curl -X POST http://localhost:8200/api/v1/identify-batch \
  -H "Content-Type: application/json" \
  -d "{\"images\": [\"$IMG1\", \"$IMG2\", \"$IMG3\"], \"include_motion\": true}"
```

Response includes per-frame detections, motion direction per person, and optional annotated images.

### Motion direction detection

| Direction | Meaning | Detection |
| --- | --- | --- |
| `left-to-right` | Moving rightward | Horizontal centroid displacement |
| `right-to-left` | Moving leftward | Horizontal centroid displacement |
| `towards-camera` | Approaching camera | Face area increasing |
| `away-from-camera` | Moving away | Face area decreasing |
| `stationary` | No movement | Below displacement threshold |

---

## Model profiles

The service uses one inference code path for both deployments. Change only the
Triton endpoint and profile label:

```bash
# Full-precision models served by continuous-tracking/triton-models
TRITON_GRPC_URL=triton:8701
PERSON_ID_MODEL_PROFILE=full

# Explicit-Q/DQ INT8 models served by continuous-tracking/triton-models-jetson
TRITON_GRPC_URL=jetson-hostname-or-ip:8701
PERSON_ID_MODEL_PROFILE=int8
```

The profile label is exposed in health metadata. It does not select a second
implementation or silently change model names.

| Triton model | Buffalo_L source | Role |
| --- | --- | --- |
| `face-detector-scrfd` | `det_10g.onnx` | Face detection and five-point landmarks |
| `face-recognition-arcface` | `w600k_r50.onnx` | 512-dimensional identity embedding |
| `face-landmark-2d106` | `2d106det.onnx` | 106-point facial landmarks |
| `face-landmark-3d68` | `1k3d68.onnx` | 68-point 3D landmarks and head pose |
| `face-attribute-genderage` | `genderage.onnx` | Gender and age estimates |

Startup fails unless Triton is reachable and all five models are ready. There
is no local ONNX Runtime fallback, partial-model mode, or automatic endpoint
failover. This keeps DGX and Jetson behavior on the same tested code path.

Verify the selected runtime after startup:

```bash
curl -s http://localhost:8200/health
```

The response includes the configured Triton endpoint, the `full` or `int8`
profile, and readiness state.

Face embeddings and centroids are stored in PostgreSQL (pgvector). Guest images
are uploaded to MinIO.

## Model provenance

The canonical full-precision files are in
`data/models/models/buffalo_l` and match `data/models/models/buffalo_l.zip`.
The files in `data/models/buffalo_l` were rewritten by ONNX tooling, but they
are also FP32. They contain no quantize or dequantize nodes. Their graph nodes
and initializer tensors match the canonical models, so they are used as the
calibration baseline for the explicit-Q/DQ Jetson exports.

The served FP32 and INT8 repositories are owned by
[`continuous-tracking`](https://github.com/SilverMind-Project/continuous-tracking):

- `triton-models`: canonical full-precision Buffalo_L models for the DGX
- `triton-models-jetson`: calibrated explicit-Q/DQ models and Jetson TensorRT
  build tooling

See [Jetson CTS deployment](https://silvermind-project.github.io/hardware/jetson-cts)
and [Model quantization](https://silvermind-project.github.io/hardware/model-quantization)
for qualification metrics and deployment constraints.

## Code quality

```bash
uv run ruff check .               # Lint
uv run ruff format --check .      # Format check
uv run mypy app/                  # Type check
uv run pytest                     # Tests
```

---

## License

AGPL-3.0-or-later
