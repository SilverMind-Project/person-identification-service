# Person Identification Service

GPU-accelerated face recognition and motion direction detection service for the Cognitive Companion system. Identifies household members in camera images and detects direction of movement at doorways.

## Architecture

- **Face Detection**: SCRFD (via InsightFace `buffalo_l` model pack)
- **Face Recognition**: ArcFace 512-dimensional embeddings with cosine similarity matching
- **Motion Detection**: Cross-frame centroid tracking with face re-identification
- **Runtime**: ONNX Runtime with CUDA execution provider

## Requirements

- NVIDIA GPU with 10 GB+ VRAM (RTX 3060 or better)
- CUDA 12.x drivers installed
- Docker with NVIDIA Container Toolkit (for containerized deployment)

## Quick Start

### Docker (recommended)

```bash
# Build
docker build -t person-id-service .

# Run with GPU access and persistent data volume
docker run --gpus all -p 8100:8100 -v $(pwd)/data:/app/data person-id-service

# Verify
curl http://localhost:8100/health
```

### Docker Compose

```bash
docker compose up -d
```

Configuration is driven by `config/settings.yaml`. Override individual values via environment variables — see the commented `environment:` block in `docker-compose.yml` for the full list.

### Local Development

```bash
# Install uv (https://docs.astral.sh/uv/)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv sync

# GPU build (default)
uv sync

# CPU-only (development/testing without a GPU)
uv sync --extra cpu

# Run
uv run uvicorn app.main:app --host 0.0.0.0 --port 8100 --reload
```

## Configuration

All settings live in `config/settings.yaml`. Values marked with `${VAR:default}` can be overridden at runtime by setting the named environment variable — no file edits required.

| Env var | settings.yaml key | Default | Description |
| --- | --- | --- | --- |
| `PERSON_ID_MODEL` | `face_engine.model_name` | `buffalo_l` | InsightFace model pack (`buffalo_s` for faster inference) |
| `CUDA_DEVICE_ID` | `face_engine.ctx_id` | `0` | GPU device index (`-1` for CPU fallback) |
| `DETECTION_THRESHOLD` | `face_engine.det_threshold` | `0.5` | Minimum face detection confidence |
| `RECOGNITION_THRESHOLD` | `recognition.threshold` | `0.4` | Cosine similarity threshold for positive ID |
| `LOG_LEVEL` | `logging.level` | `INFO` | Log verbosity (`DEBUG`, `INFO`, `WARNING`) |

Additional tunable settings (edit `config/settings.yaml` directly):

| Setting | Default | Description |
| --- | --- | --- |
| `face_engine.det_size` | `[640, 640]` | Detection input resolution |
| `recognition.unknown_threshold` | `0.25` | Below this = definitely unknown |
| `motion.min_displacement_fraction` | `0.05` | Min displacement (% of frame) to count as movement |
| `annotation.box_color_known` | `[0, 200, 0]` | BGR color for known person bounding boxes |
| `annotation.box_color_unknown` | `[0, 165, 255]` | BGR color for unknown person bounding boxes |
| `annotation.text_scale` | `0.7` | Font scale for name labels |
| `annotation.box_thickness` | `2` | Bounding box line thickness |

## Code Quality

```bash
# Lint
uv run ruff check .

# Format
uv run ruff format .

# Type check
uv run mypy app/

# Run all checks (lint + format + type check)
uv run ruff check . && uv run ruff format --check . && uv run mypy app/

# Tests
uv run pytest
```

## Member Management API

Full CRUD operations for managing enrolled household members. No model fine-tuning is needed; the pretrained ArcFace model generalizes to new faces. "Enrollment" simply means registering reference photos for each person.

### Best Practices for Enrollment Photos

1. **Capture 5-10 images per person** for robust recognition
2. **Vary lighting**: daylight, evening lamp, nightlight
3. **Vary angle**: front face, slight left/right turns, looking down
4. **Include accessories**: with/without glasses, different hairstyles
5. **Use actual cameras**: photos from the deployment cameras match the inference domain best
6. **Avoid group photos**: each image should contain only the target person's face clearly visible

### List All Members

`GET /api/v1/members`

Returns all enrolled household members with their metadata.

```bash
curl http://localhost:8100/api/v1/members
```

Response:

```json
{
  "members": [
    {
      "person_id": "grandma",
      "name": "Grandma",
      "embedding_count": 5,
      "created_at": "2026-03-20T14:30:00+00:00"
    },
    {
      "person_id": "grandpa",
      "name": "Grandpa",
      "embedding_count": 3,
      "created_at": "2026-03-20T14:35:00+00:00"
    }
  ],
  "total": 2
}
```

### Get a Single Member

`GET /api/v1/members/{person_id}`

Returns details for a specific enrolled member. Returns 404 if the member does not exist.

```bash
curl http://localhost:8100/api/v1/members/grandma
```

Response:

```json
{
  "person_id": "grandma",
  "name": "Grandma",
  "embedding_count": 5,
  "created_at": "2026-03-20T14:30:00+00:00"
}
```

### Enroll a New Member (base64 JSON)

`POST /api/v1/enroll`

Enroll a new household member by providing base64-encoded face images in a JSON payload. If the `person_id` already exists, the new images are added and the centroid embedding is recomputed (see "Add More Images" below).

```bash
# Encode images to base64
IMG1=$(base64 -w0 grandma_photo1.jpg)
IMG2=$(base64 -w0 grandma_photo2.jpg)
IMG3=$(base64 -w0 grandma_photo3.jpg)

# Enroll
curl -X POST http://localhost:8100/api/v1/enroll \
  -H "Content-Type: application/json" \
  -d "{
    \"person_id\": \"grandma\",
    \"name\": \"Grandma\",
    \"images\": [\"$IMG1\", \"$IMG2\", \"$IMG3\"]
  }"
```

Request body fields:

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `person_id` | string | Yes | Unique identifier (1-64 characters) |
| `name` | string | Yes | Display name (1-128 characters) |
| `images` | list[string] | Yes | Base64-encoded face images (JPEG/PNG), minimum 1 |

Response:

```json
{
  "person_id": "grandma",
  "name": "Grandma",
  "embedding_count": 3,
  "status": "enrolled",
  "failed_images": []
}
```

The `status` field will be `"enrolled"` for new members or `"updated"` when adding images to an existing member. The `failed_images` list contains the indices of any images where no face could be detected. If no faces are detected in any image, the endpoint returns HTTP 422.

### Enroll a New Member (file upload)

`POST /api/v1/enroll/upload/{person_id}`

A convenience endpoint for enrolling via multipart file upload, useful with curl or admin tools. Behaves identically to the base64 endpoint in terms of enrollment logic.

```bash
curl -X POST http://localhost:8100/api/v1/enroll/upload/grandma \
  -F "name=Grandma" \
  -F "files=@grandma_photo1.jpg" \
  -F "files=@grandma_photo2.jpg" \
  -F "files=@grandma_photo3.jpg"
```

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `person_id` | path param | Yes | Unique identifier for the person |
| `name` | form field | Yes | Display name |
| `files` | file(s) | Yes | One or more image files (JPEG/PNG) |

Returns the same `EnrollResult` response as the base64 endpoint.

### Add More Images to an Existing Member

To improve recognition accuracy, call either enroll endpoint again with the same `person_id`. New embeddings are appended to the existing set and the centroid is recomputed. The response `status` will be `"updated"` instead of `"enrolled"`.

```bash
# Add two more photos using the base64 endpoint
IMG4=$(base64 -w0 grandma_photo4.jpg)
IMG5=$(base64 -w0 grandma_photo5.jpg)

curl -X POST http://localhost:8100/api/v1/enroll \
  -H "Content-Type: application/json" \
  -d "{
    \"person_id\": \"grandma\",
    \"name\": \"Grandma\",
    \"images\": [\"$IMG4\", \"$IMG5\"]
  }"
```

### Delete a Member

`DELETE /api/v1/members/{person_id}`

Permanently removes an enrolled member and all their stored embeddings (individual embeddings and centroid). Returns 404 if the member does not exist.

```bash
curl -X DELETE http://localhost:8100/api/v1/members/grandma
```

Response:

```json
{
  "deleted": true,
  "person_id": "grandma"
}
```

## Inference (Identification)

### Single Image

```bash
IMG=$(base64 -w0 camera_snapshot.jpg)

curl -X POST http://localhost:8100/api/v1/identify \
  -H "Content-Type: application/json" \
  -d "{\"image\": \"$IMG\"}"
```

Response:

```json
{
  "faces": [
    {
      "person_id": "grandma",
      "name": "Grandma",
      "confidence": 0.87,
      "bbox": [120.5, 80.2, 250.3, 310.7]
    }
  ],
  "annotated_image": null
}
```

### Annotated Image Output

Set `include_annotated_image: true` to receive the image back with bounding boxes and name labels drawn on it. Known persons are drawn in green, unknown in orange.

```bash
curl -X POST http://localhost:8100/api/v1/identify \
  -H "Content-Type: application/json" \
  -d "{\"image\": \"$IMG\", \"include_annotated_image\": true}"
```

The response `annotated_image` field contains a base64-encoded JPEG with bounding boxes and `"Name 85%"` labels drawn over each detected face. This is used by the Cognitive Companion pipeline's `person_identification` step when `include_annotated_image` is enabled in the step config, providing downstream VLM steps with labeled frames for better contextual analysis.

### Batch Identification with Motion Detection

This is the primary endpoint used by the Cognitive Companion backend. Send a sequence of frames (typically 5 from the event aggregator) to identify faces and detect motion direction.

```bash
IMG1=$(base64 -w0 frame_001.jpg)
IMG2=$(base64 -w0 frame_002.jpg)
IMG3=$(base64 -w0 frame_003.jpg)

curl -X POST http://localhost:8100/api/v1/identify-batch \
  -H "Content-Type: application/json" \
  -d "{
    \"images\": [\"$IMG1\", \"$IMG2\", \"$IMG3\"],
    \"include_motion\": true,
    \"include_annotated_image\": false
  }"
```

Response:

```json
{
  "frames": [
    {"frame_index": 0, "faces": [{"person_id": "grandma", "name": "Grandma", "confidence": 0.85, "bbox": [100, 50, 200, 250]}]},
    {"frame_index": 1, "faces": [{"person_id": "grandma", "name": "Grandma", "confidence": 0.87, "bbox": [150, 55, 260, 260]}]},
    {"frame_index": 2, "faces": [{"person_id": "grandma", "name": "Grandma", "confidence": 0.86, "bbox": [210, 60, 320, 270]}]}
  ],
  "motion": [
    {"person_id": "grandma", "name": "Grandma", "direction": "left-to-right", "confidence": 0.92}
  ],
  "annotated_images": null
}
```

### Motion Direction Only

```bash
curl -X POST http://localhost:8100/api/v1/detect-motion \
  -H "Content-Type: application/json" \
  -d "{\"images\": [\"$IMG1\", \"$IMG2\", \"$IMG3\"]}"
```

## Motion Direction Detection

The service detects four directions of movement:

| Direction | Meaning | Detection Method |
| --- | --- | --- |
| `left-to-right` | Person moving rightward in frame | Horizontal centroid displacement |
| `right-to-left` | Person moving leftward in frame | Horizontal centroid displacement |
| `towards-camera` | Person approaching the camera | Face bounding box area increasing |
| `away-from-camera` | Person moving away from camera | Face bounding box area decreasing |
| `stationary` | No significant movement | Below displacement threshold |

### Doorway Configuration

For cameras at doorways, configure the camera sensor's `config_json` in the Cognitive Companion backend with a `door_direction` field to map left/right motion to entering/leaving semantics:

```json
{
  "door_direction": "left_is_inside"
}
```

## Guest/Unknown Handling

- Faces not matching any enrolled member are classified as `"unknown"` with `name: "Guest"`
- The `confidence` field shows how close the best match was
- Between `unknown_threshold` (0.25) and `threshold` (0.4): uncertain match
- Below `unknown_threshold`: definitely unknown

### Saving Guest Images

Both `/identify` and `/identify-batch` endpoints accept a `save_guest_images` flag (default: `false`). When enabled, the **full frame image** is saved to disk whenever unidentified guests are detected. This is useful for reviewing visitors, building enrollment datasets, or auditing false negatives.

```bash
# Single image - save if guests detected
curl -X POST http://localhost:8100/api/v1/identify \
  -H "Content-Type: application/json" \
  -d "{\"image\": \"$IMG\", \"save_guest_images\": true}"

# Batch - save frames containing guests
curl -X POST http://localhost:8100/api/v1/identify-batch \
  -H "Content-Type: application/json" \
  -d "{
    \"images\": [\"$IMG1\", \"$IMG2\", \"$IMG3\"],
    \"include_motion\": true,
    \"save_guest_images\": true
  }"
```

Images are saved to `data/guests/` organized by date:

```text
data/guests/
├── 2026-03-23/
│   ├── 143022-123456_f0_2guests.jpg
│   ├── 143022-234567_f1_1guests.jpg
│   └── ...
└── 2026-03-24/
    └── ...
```

The filename encodes: `{UTC time}_{frame index}_{guest count}.jpg`

Configure the storage directory in `config/settings.yaml`:

```yaml
storage:
  guest_images_dir: "data/guests"   # default
```

## Data Storage

```text
data/
├── face_db.sqlite              # Enrollment metadata (person_id, name, counts)
├── embeddings/
│   ├── grandma/
│   │   ├── centroid.npy         # Mean embedding (used for fast identification)
│   │   ├── embedding_0.npy     # Individual face embeddings
│   │   ├── embedding_1.npy
│   │   └── ...
│   └── grandpa/
│       ├── centroid.npy
│       └── ...
└── guests/                      # Saved guest images (when save_guest_images=true)
    ├── 2026-03-23/
    │   └── *.jpg
    └── ...
```

Mount the `data/` directory as a persistent volume to retain enrollment and guest images across container restarts.

## Improving Accuracy

If recognition accuracy is low for a specific person:

1. **Add more reference photos** via the enrollment endpoint (different angles, lighting)
2. **Lower the recognition threshold** via `RECOGNITION_THRESHOLD` env var or `recognition.threshold` in `config/settings.yaml` (e.g., `0.35`) — increases true positives but may increase false positives
3. **Use photos from the actual cameras** — domain-matched images work best
4. **Check face visibility** — ensure cameras capture faces clearly (adequate lighting, appropriate angle)

## Kubernetes Deployment

Kubernetes manifests are in `kubernetes/`:

```text
kubernetes/
├── base/                  # Environment-agnostic manifests
│   ├── deployment.yaml    # GPU deployment (nvidia.com/gpu: 1)
│   ├── service.yaml       # ClusterIP on port 8100
│   └── pvc.yaml           # 5 Gi persistent volume for data/
└── local/                 # Local cluster overlay
    └── deployment.yaml    # localhost:32000 registry image
```

The service is deployed as `person-id-svc` on port 8100 and is accessed by the Cognitive Companion backend via `PERSON_ID_SERVICE_URL`.
