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

1. **Remote model lifecycle.** All five graphs configured by the active `RecognizerProfile` (`face_engine.active_profile`, default `buffalo_l`) are served by Triton. Startup fails unless every configured model is ready.
2. **Vector similarity search.** Face identification uses nearest-centroid or pose-binned prototype search against pgvector, scoped by `recognizer_version`. The database is the source of truth for enrollment state.

---

## 2. Tech stack

| Layer | Choice |
| --- | --- |
| Backend | Python 3.12, FastAPI, Pydantic 2 for wire models (dataclasses such as `DetectedFace`, `IdentifyResult`, `PersonTrackResult` for internal data), stdlib `logging` |
| Database | PostgreSQL with TimescaleDB + pgvector extensions; asyncpg driver with `pgvector` package for vector codec |
| Object storage | MinIO (S3-compatible) via `boto3` |
| Face AI | Buffalo_L: SCRFD, ArcFace, 2D/3D landmarks, gender/age |
| Inference runtime | NVIDIA Triton Inference Server over gRPC |
| Image processing | OpenCV (`opencv-python-headless`) |
| Package manager | `uv` (`uv.lock` is committed) |
| Lint and types | `ruff` (`E`, `F`, `I`, `W`, `UP`, `B`, `SIM`, `RUF`, `PIE`, `PT`, `C4`, `T20`; 100-char lines); `mypy` gradual typing over `app/` |

**Logging is stdlib `logging.getLogger(__name__)` with `key=value` text, by design.** cognitive-companion uses its own `backend.core.logging` wrapper and continuous-tracking uses `structlog`; each choice is per project. Do not "fix" this service to match another repository.

---

## 3. Repository layout

```text
person-identification-service/
├── app/
│   ├── main.py              # App factory + lifespan (service wiring source of truth)
│   ├── config.py            # YAML config loader with ${ENV_VAR:default} interpolation
│   ├── calibration/
│   │   ├── models.py        # CalibrationArtifact (schema 1) and CalibrationArtifactV2 (schema 2) models (JSON, never pickle)
│   │   ├── episodes.py      # Group captures into walks (one label, one camera, gaps under 300 s)
│   │   ├── manifest.py      # Score manifest builder: household and external records (offline)
│   │   ├── evaluator.py     # Runtime evaluator (stdlib + numpy only); health states
│   │   ├── fit.py           # Offline fitter (scikit-learn; calibration-tools extra)
│   │   ├── quality_breakpoints.py # Offline face quality anchor analysis (scipy; calibration-tools)
│   │   ├── recognizer_eval.py # Deterministic household comparison harness for recognizer swaps (Part B)
│   │   └── cli.py           # fit / evaluate / inspect / validate / build-manifest / fit-v2 commands
│   ├── db/
│   │   └── migrate.py       # asyncpg migration runner (applies migrations/*.up.sql)
│   ├── models/
│   │   ├── enrollment.py    # EnrollRequest, EnrollResult, MemberInfo, MemberListResponse
│   │   ├── identification.py # FaceDetection, IdentifyRequest/Response, BatchIdentifyRequest/Response, PersonMotion, FrameResult
│   │   ├── motion.py        # MotionDetectionRequest/Response, PersonTrack, TrajectoryPoint
│   │   ├── visitor.py       # Visitor cluster list/detail/name envelopes
│   │   └── calibration_review.py # Calibration candidate list, label, and summary envelopes
│   ├── observability/
│   │   └── metrics.py       # Prometheus metrics (pid_face_interocular_px, pid_face_quality_score, visitor counters)
│   ├── routers/
│   │   ├── health.py        # GET /health, GET /metrics (no /api/v1 prefix)
│   │   ├── enrollment.py    # POST /enroll, POST /enroll/upload/{id}, GET/DELETE /members/{id}, /members/{id}/prototypes*
│   │   ├── identification.py # POST /identify, POST /identify-batch
│   │   ├── motion.py        # POST /detect-motion
│   │   ├── visitors.py      # /visitors/clusters list, detail, name, dismiss, merge
│   │   └── calibration.py   # /calibration/candidates list, label, label-batch, reject, delete, promote; /calibration/summary; /calibration/attestations (wearable-attested faces, audit)
│   ├── services/
│   │   ├── recognizer_profile.py # Frozen RecognizerProfile registry (Triton models, alignment, normalization, licence)
│   │   ├── face_engine.py     # Triton preprocessing, inference, and postprocessing (profile-driven)
│   │   ├── face_models.py     # DetectedFace, IdentifyResult dataclasses
│   │   ├── face_geometry.py   # Face box size and interocular distance (pure)
│   │   ├── face_quality.py    # Face image quality scorer protocol + heuristic-v1 (pure)
│   │   ├── prototype_builder.py # Pose binning, quality weighting, prototype clustering, promotion rules (pure)
│   │   ├── enrollment_store.py # Orchestration: enroll(), identify(), identify_all(), promote_candidate(), members (no SQL, no config)
│   │   ├── enrollment_repository.py # EnrollmentRepository protocol + Postgres and in-memory peers
│   │   ├── match_policy.py    # classify_match: nearest identities + margin to recognition state (pure)
│   │   ├── centroid.py        # Shared embedding insert + centroid recompute (enroll and visitor naming)
│   │   ├── visitor_store.py   # Visitor sightings, clustering, surfacing, naming, retention
│   │   ├── calibration_candidates.py # Calibration capture repository (Postgres + in-memory peers)
│   │   ├── calibration_capture.py # Stratified capture policy, background crop upload, retention
│   │   ├── minio_client.py    # boto3 S3 wrapper for MinIO
│   │   ├── motion_detector.py # Cross-frame centroid tracking + direction classification
│   │   ├── guest_store.py     # MinIO uploads for unidentified-person frames
│   │   └── image_annotator.py # Bounding box + label drawing
│   └── tools/
│       ├── rebuild_prototypes.py # CLI to backfill pose bins and rebuild face_prototypes for all members
│       └── reembed.py       # CLI to re-embed retained enrollment crops under a target RecognizerProfile
├── config/
│   └── settings.yaml        # All application settings
├── scripts/
│   ├── calibrate_face_quality.py # Offline: suggest face quality ramp anchors from labeled images
│   ├── evaluate_recognizer_swap.py # Offline: deterministic household comparison of candidate recognizers
│   └── check-no-em-dash.sh  # Markdown em-dash gate (make check-md)
├── schemas/
│   ├── calibration_artifact.schema.json
│   ├── calibration_artifact_v2.schema.json
│   └── recognizer_comparison_report.schema.json
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
# Schema 2 from household labels (read-only DB access; manifests hold private data)
uv run --extra calibration-tools python -m app.calibration.cli build-manifest --output manifest.jsonl
uv run --extra calibration-tools python -m app.calibration.cli fit-v2 --manifest manifest.jsonl --output artifact.json --allow-household-only
# Re-embed retained enrollment crops for a candidate RecognizerProfile (parallel gallery)
DATABASE_URL=... TRITON_GRPC_URL=triton:8701 uv run python -m app.tools.reembed --profile adaface_ir50_webface4m
# Deterministic household comparison of candidate recognizers
uv run --extra calibration-tools python scripts/evaluate_recognizer_swap.py --baseline-scores base.json --candidate-scores cand.json --candidate-profile adaface_ir50_webface4m --output report.json

# Quality gate: `make check` runs ruff check, ruff format --check, mypy app/, and the full
# suite with the dev and calibration-tools extras (uv run --frozen, so uv.lock never churns).
# Database tests run only against a database whose name ends in _test, and skip without
# TEST_DATABASE_URL; they never read DATABASE_URL (their teardown deletes rows).
# PID_REQUIRE_FULL_SUITE=1 (set in CI, .github/workflows/ci.yml) turns any skip into a failure.
export TEST_DATABASE_URL=postgresql://pid_user:...@localhost:5432/person_identification_test
make check
PID_REQUIRE_FULL_SUITE=1 make check
make audit                       # pip-audit over the locked dependencies
uv run --frozen ruff format .    # Format (writes)
uv run --frozen --extra dev pytest tests/test_enrollment_store.py -v   # Targeted

# Migrations (startup applies pending ones; these are for operators)
uv run python -m app.db.migrate status
uv run python -m app.db.migrate rollback --steps 1

# Docker
docker build -t person-id-service .
docker run -p 8200:8200 -e TRITON_GRPC_URL=triton:8701 person-id-service
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

1. `logging.basicConfig()` from `logging.level`; build the face quality scorer from `face_quality` (an unknown scorer fails startup here, before any connection)
2. `_init_pool(dsn)`: asyncpg pool with the `pgvector` codec registered (`min_size=1, max_size=5`)
3. `run_migrations(pool)`: applies pending `migrations/*.up.sql` files transactionally
4. `create_minio_client()`: MinIO client with bucket ensured
5. Load the active `RecognizerProfile` from `face_engine.active_profile` and `face_engine.profiles` (`app/services/recognizer_profile.py`), open `TritonGrpcClient`, build `FaceEngine(triton_client, profile=..., quality_scorer=...)`, and require all five profile-configured models to be ready
6. `EnrollmentStore(PostgresEnrollmentRepository(pool, recognizer_version=profile.recognizer_version), face_engine, MatchThresholds(...), minio_client=minio_client, on_member_removed=...)`, `MotionDetector()`, `GuestImageStore(pool, minio_client)`
7. `VisitorStore(pool, minio_client, recognizer_version=profile.recognizer_version)`, `CalibrationCaptureService` (started; seeds today's caps and runs its worker only when `calibration_capture.enabled`), plus the daily retention task (`_retention_loop`, purges expired unnamed visitor clusters and expired calibration captures)
8. `CalibrationEvaluator.from_artifact_path(...)` with the expected `recognizer_version` (defaulting to the active profile), `model_profile`, `preprocessing_version`, and profile-scoped `quality_version`; a missing or incompatible artifact yields a degraded evaluator, never a startup failure. A second evaluator loads `calibration.template_artifact_path` for the `template` profile, bound to the template aggregator's version; `TemplateMatchService` (the same profile-scoped `PostgresEnrollmentRepository`, `QualityWeightedMeanAggregator`, and `MatchThresholds` from `recognition.template_*`) serves `/match-template`
9. Store on `app.state`: `face_engine`, `triton_client`, `enrollment_store`, `template_matcher`, `face_quality_version`, `motion_detector`, `guest_store`, `visitor_store`, `calibration_candidates`, `calibration_capture`, `calibration_evaluator`, `template_calibration_evaluator`
10. On shutdown: cancel the retention task, stop the capture worker, close Triton, then close the database pool

### 5.4 Configuration

`app/config.py` loads `config/settings.yaml` eagerly at module import time. Env-var interpolation uses `${VAR:default}` syntax.

```python
from app.config import config

model = config.get("face_engine.model_name")  # "buffalo_l"
threshold = config.get("recognition.threshold", 0.4)
```

Dot-notation keys map to nested dict paths. `config.reload()` clears the cache and re-reads the file (used only in tests; in production, restart the process).

All secrets (database DSN, MinIO credentials) come from environment variables. The config file contains only non-sensitive defaults.

Key settings:

| Setting | Env var | Default | Description |
| --- | --- | --- | --- |
| `face_engine.triton_url` | `TRITON_GRPC_URL` | `triton:8701` | Triton gRPC endpoint |
| `face_engine.active_profile` | `PERSON_ID_RECOGNIZER_PROFILE` | `buffalo_l` | Active `RecognizerProfile` key under `face_engine.profiles` |
| `face_engine.profiles` | -- | `buffalo_l`, `adaface_ir50_webface4m` | Named recognizer profiles (Triton models, alignment template, color space, normalization, tensor names, licence) |
| `face_engine.model_profile` | `PERSON_ID_MODEL_PROFILE` | `full` | Health metadata |
| `face_engine.triton_timeout_ms` | `TRITON_TIMEOUT_MS` | `30000` | Request timeout |
| `face_engine.det_threshold` | `DETECTION_THRESHOLD` | `0.6` | Face detection confidence |
| `recognition.threshold` | `RECOGNITION_THRESHOLD` | `0.4` | Cosine similarity for positive ID |
| `recognition.unknown_threshold` | -- | `0.25` | Lower bound of the `candidate` band; below is `unrecognized` |
| `recognition.template_threshold` | -- | `0.4` | The recognition threshold for `/match-template` templates; starts equal to the single-face value until a template fit shows otherwise |
| `recognition.template_unknown_threshold` | -- | `0.25` | The `candidate` lower bound for templates |
| `calibration.artifact_path` | `CALIBRATION_ARTIFACT_PATH` | empty | Calibration artifact JSON (schema 1 or 2); empty means `degraded_missing`. Schema 2 binds to the quality scorer version and covers only its listed identities |
| `calibration.template_artifact_path` | `CALIBRATION_TEMPLATE_ARTIFACT_PATH` | empty | Schema 2 artifact with `profile: template` (built with `build-manifest --profile template`); calibrates `/match-template` only, and binds to the template aggregator version |
| `face_quality.scorer` | -- | `heuristic-v1` | Face quality scorer; unknown names fail startup |
| `face_quality.heuristic.*` | -- | see settings | Ramp anchors and geometric-mean weights; a change is a new scorer version |
| `calibration_capture.enabled` | -- | `false` | Sample faces from tracker `/identify` calls into the calibration set |
| `calibration_capture.per_bucket_per_day` | -- | `40` | Cap per (quality decile, nearest identity, camera) per UTC day |
| `calibration_capture.unlabeled_retention_days` | -- | `30` | Unlabeled and rejected captures (and crops) purged after this |
| `calibration_capture.queue_size` | -- | `64` | Background capture queue; a full queue drops the offer |
| `calibration_capture.attestation_enabled` | -- | `false` | Hold faces of `/identify` calls that name a detection, in memory, for wearable attestation |
| `calibration_capture.attestation_max_items` | -- | `2000` | Faces held at once (aligned 112x112 crops, about 38 KB each) |
| `calibration_capture.attestation_max_age_s` | -- | `600` | A held face older than this (capture time) can no longer be attested |
| `calibration_capture.attestation_min_quality` | -- | `0.35` | heuristic-v1 quality floor for holding a face |
| `calibration_capture.attestation_min_interocular_px` | -- | `24.0` | Eye-distance floor for holding a face |
| `calibration_capture.attestation_per_bucket_per_day` | -- | `40` | Attested faces stored per (quality decile, attested identity, camera) per UTC day |
| `visitors.clustering_enabled` | -- | `true` | Persist and cluster unknown-face sightings |
| `visitors.min_face_quality` | -- | `0.7` | Minimum face quality score for a visitor sighting (0.0 disables) |
| `visitors.max_abs_yaw_deg` | -- | `30` | Maximum absolute head yaw for a visitor sighting (180 disables) |
| `motion.min_displacement_fraction` | -- | `0.05` | Min displacement for movement |
| `motion.cross_frame_similarity` | -- | `0.5` | Unknown-face cross-frame linking |
| `database.dsn` | `DATABASE_URL` | -- | PostgreSQL connection string (required) |
| `minio.endpoint` | `MINIO_ENDPOINT` | `localhost:9000` | MinIO/S3 endpoint |
| `minio.access_key` | `MINIO_ACCESS_KEY` | `minioadmin` | MinIO access key |
| `minio.secret_key` | `MINIO_SECRET_KEY` | `minioadmin` | MinIO secret key |
| `minio.bucket` | `MINIO_BUCKET` | `cognitive-companion` | S3 bucket name |
| `minio.secure` | -- | `false` | Use HTTPS |

---

## 6. Services in depth

### 6.1 FaceEngine (`app/services/face_engine.py`)

Implements the active `RecognizerProfile` (`app/services/recognizer_profile.py`, default `buffalo_l`) contract over Triton. There is no local ONNX Runtime path and no partial-model fallback.

```python
engine = FaceEngine(triton_client, profile=profile, quality_scorer=scorer)
await engine.validate_models()
faces: list[DetectedFace] = await engine.detect_faces(image)
embedding: np.ndarray = await engine.embed_aligned_crop(aligned_crop_bgr)
similarity: float = FaceEngine.compute_similarity(emb1, emb2)  # static
```

**Key details:**
- `det_size: [640, 640]` is required by the SCRFD Triton contract
- `det_threshold: 0.6` -- filters low-confidence detections
- Returns `DetectedFace` with bounding box, normalized embedding, landmarks, pose, attributes, and `aligned_crop_bgr` (112x112 aligned BGR crop, excluded from repr/compare, used by enrollment crop retention)
- Alignment template, input size, color space (`rgb` or `bgr`), normalization mean/std, and recognition input/output tensor names all come from `RecognizerProfile`; `landmark_3d_model` stays independent of `recognition_model` so head-pose Euler angles stay stable across recognizer swaps
- Quality version is scoped via `profile.scoped_quality_version(base_version)`: `buffalo_l` keeps `"heuristic-v1"`, while non-default profiles emit `"heuristic-v1@<recognizer_version>"` so quality consumers fail closed on a switch
- `embed_aligned_crop(aligned_crop_bgr)` embeds an already-aligned crop for offline re-embedding (`app/tools/reembed.py`)
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
3. `upsert_member` (reports "enrolled" or "updated"), `insert_embeddings` (returns generated `BIGINT` ids), encode each `aligned_crop_bgr` as PNG and upload to MinIO under `enrollment/<person_id>/<embedding_id>.png` (`set_embedding_crop_key`), `recompute_centroid` (mean of the member's embeddings for the active `recognizer_version`, L2-normalized, upserted into `centroids`), and rebuild pose-binned prototypes

**Members:** `list_members()` (with embedding counts for the active `recognizer_version`), `remove_member(person_id)` (CASCADE removes embeddings, centroids, and prototypes; also deletes `enrollment/<person_id>/` in MinIO and runs `on_member_removed` to purge calibration candidates/crops), and `member_count()`. Promoting a calibration candidate copies its crop to `enrollment/<person_id>/<embedding_id>.png` so promoted exemplars survive calibration purges and participate in `reembed`.

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
- `upload_bytes(data, object_name, content_type) -> str` uploads via `BytesIO` and returns a presigned URL; `generate_presigned_url(object_name, expiration=3600)`; `delete_object(object_name)`
- Factory: `create_minio_client()` reads the MinIO settings, creates the client, and ensures the bucket exists

### 6.6 Image Annotator (`app/services/image_annotator.py`)

Draws bounding boxes and labels on a copy of the input image:
- Known persons: green box (`[0, 200, 0]` BGR) with `"Name 85%"` label
- Unknown persons: orange box (`[0, 165, 255]` BGR) with `"Guest 42%"` label
- All colors, scales, thicknesses configurable via `annotation.*` settings
- Returns a new numpy array; input image is not mutated

### 6.7 VisitorStore (`app/services/visitor_store.py`)

Manages the visitor clustering subsystem for recurring unknown visitors.

**Tables:**
- `visitor_clusters`: tracks `cluster_id`, `status` (`candidate`, `surfaced`, `named`, `dismissed`, `attributed`), `distinct_days`, the centroid, and for reviewed clusters `attributed_person_id`, `reviewed_by`, and `reviewed_at` (migration 0004).
- `visitor_sightings`: hypertable for each observed face crop (MinIO key) and embedding.

**Config keys (`config.visitors.*`):**
- `clustering_enabled`: Master toggle.
- `min_detection_score` (0.6), `min_face_px` (64): Detection and size gates.
- `min_face_quality` (0.7 in settings; code default 0.0 disables) and `max_abs_yaw_deg` (30; code default 180 disables): a sighting needs a face quality score and a near-frontal pose. Faces no reviewer could judge split one person into many clusters. A face without a quality score is rejected. Rejections are counted in `pid_visitor_sightings_rejected_total{reason}`.
- `member_margin` (0.05): Member-margin fail-closed rule. A borderline match against the household (similarity > recognition.threshold - member_margin) is explicitly rejected from visitor clustering to prevent a known member from seeding a visitor cluster.
- `cluster_join_threshold` (0.55): Looser than recognition.
- `surface_min_days` (3), `surface_window_days` (30): Promotion criteria.
- `unnamed_retention_days` (60): Retention period.
- `enroll_top_k` (5): Embeddings copied on naming.

**Open and closed clusters** (pure policy in `app/services/visitor_policy.py`):
`candidate` and `surfaced` clusters are open: their centroid is the running mean of their sightings. `dismissed` and `attributed` clusters are closed by an operator: they keep absorbing matching sightings (so the same face does not come back as a new cluster) but their centroid is frozen at review time, so they never widen toward faces nobody reviewed. Each such quiet join is counted in `pid_visitor_sightings_absorbed_total{status}`: a reviewed cluster swallowing a face it was not reviewed for (a real visitor who resembles a member) shows up there and nowhere else. `named` clusters absorb nothing.

**Review without enrollment:**
`POST /api/v1/visitors/clusters/review` attributes clusters to an existing, enrolled member (`attributed`) or dismisses them, per cluster, recording the actor. Attribution never adds an embedding to the member: it says whose faces these are and stops them surfacing. Any cluster except a named one can be reviewed again. Deleting a member reverts its attributed clusters to `candidate` in the same transaction (`PostgresEnrollmentRepository.remove_member`; the foreign key has no delete action, so a missed path fails). An attributed cluster cannot be merged.

**Retention:**
An async daily task deletes clusters in any state but `named` whose `last_seen_at` is older than `unnamed_retention_days` (60), including sightings and MinIO crops. A closed cluster that keeps absorbing would never age out that way, so its individual sightings older than the window are deleted too, while the cluster row and frozen centroid stay. Named clusters are exempt.

**Calibration captures are not enrollment:**
`CalibrationCaptureService` (`app/services/calibration_capture.py`) samples faces from tracker `/identify` calls into `calibration_candidates` (migration 0003), capped per (quality decile, nearest identity or `none`, camera) per UTC day. Capture runs in a bounded background queue and never fails or slows `/identify`. A label never changes the gallery; only explicit promotion (`POST /api/v1/calibration/candidates/{candidate_id}/promote`), an operator action gated on member label, active quality scorer version, `prototypes.min_promotion_quality`, and `prototypes.min_promotion_interocular_px`, adds the embedding (`source = deployment_capture`, `source_candidate_id` recorded for leave-episode-out calibration) and increments `members.gallery_revision`. Unlabeled and rejected rows are purged after `unlabeled_retention_days` (30) with their crops.

Wearable attestation (migration 0006, revocation in migration 0007) is a second path through the same storage. With `calibration_capture.attestation_enabled`, faces of `/identify` calls whose `capture_context` names a `detection_id` are held in memory (`AttestationCaptureBuffer`, nothing written). `POST /api/v1/calibration/attestations` from tracking-orchestrator stores the held faces of attested detections as `label_state = 'attested'` captures (`label_source = sensor_attested`, `labeled_by = wearable:<association id>`, `attestation_grade` room or trajectory). They never appear in the operator queue (`list_page` with no state lists only operator states) or in `list_labeled`. An operator audits a uniform random sample (`/attestations/audit-sample`, `/attestations/{id}/audit`); `app/calibration/attestation_audit.py` computes the Wilson 95% lower bound, and the manifest builder (`build-manifest --include-attested`) refuses attested captures unless at least 200 are audited with a bound of 0.99 or better, and drops captures audited wrong or revoked. An operator may promote an attested capture (never one audited wrong or revoked). `POST /api/v1/calibration/attestations/revoke` stamps `revoked_at` and `revoked_by` on every attested capture from a wearable association segment, returning already-promoted prototypes for manual review without removing them; `GET /api/v1/calibration/attestations/counts` reports active, revoked, and promoted counts per segment. Unaudited, audited-wrong, and revoked attested captures purge on the same 30-day schedule (even if previously audited `correct`); unrevoked audited-correct ones stay. Down migrations 0006 and 0007 refuse while attested or revoked rows exist.

**Naming and promotion are privileged:**
The API exposes `/api/v1/visitors/clusters/{id}/name` and `/api/v1/calibration/candidates/{id}/promote`. Enrollment gains entries ONLY via explicit enroll calls, explicit operator naming of a visitor cluster, or explicit operator promotion of a labeled capture; never automatically.

### 6.8 CalibrationEvaluator (`app/calibration/evaluator.py`)

Maps raw similarity to `calibrated_confidence` from a versioned JSON artifact. Missing, invalid, or incompatible artifacts give `calibrated_confidence = null` and a degraded `/health` calibration status; it never falls back to raw similarity. The contract is the face-confidence-calibration skill.

---

## 7. API surface

Domain endpoints are under `/api/v1`. Health and metrics are at the root. There is no authentication (LAN-internal service; Cognitive Companion is the BFF and enforces auth upstream).

### Health and metrics

| Method | Path | Response |
| --- | --- | --- |
| `GET` | `/health` | `status`, `inference_backend`, `triton_endpoint`, `model_profile`, `active_profile`, `recognizer_version`, `licence`, `enrolled_members`, `model`, `models`, `calibration` (`status`, `artifact_version`, `recognizer_version`, `arcface_model_version`, `preprocessing_version`, `profile`, `quality_version`, `covered_identities`), and `template_calibration` (the same for the template artifact, plus `aggregation`) |
| `GET` | `/metrics` | Prometheus text format |

### Enrollment

| Method | Path | Request Body | Response |
| --- | --- | --- | --- |
| `POST` | `/api/v1/enroll` | `EnrollRequest` (base64 images) | `EnrollResult` |
| `POST` | `/api/v1/enroll/upload/{person_id}` | Multipart: `name` + `files` | `EnrollResult` |
| `GET` | `/api/v1/members` | -- | `MemberListResponse` |
| `GET` | `/api/v1/members/{person_id}` | -- | `MemberInfo` (404 if missing) |
| `DELETE` | `/api/v1/members/{person_id}` | -- | `{"deleted": true, "person_id": str}` (404 if missing); also removes calibration candidates and crops associated with the member |
| `GET` | `/api/v1/members/{person_id}/prototypes` | -- | `MemberPrototypesResponse` (prototypes per pose bin, gallery revision) |
| `POST` | `/api/v1/members/{person_id}/prototypes/rebuild` | -- | `RebuildPrototypesResponse` (rebuilds pose-binned prototypes from stored embeddings) |
| `DELETE` | `/api/v1/members/{person_id}/prototypes/{prototype_id}` | -- | Removes prototype and its underlying embedding(s), rebuilds centroid and prototypes, bumps revision (409 if last embedding) |

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
| `POST` | `/api/v1/match-template` | `MatchTemplateRequest` | `MatchTemplateResponse` |

`IdentifyRequest` fields:
- `image`: base64-encoded JPEG/PNG (with or without `data:` URI prefix)
- `include_annotated_image`: bool (default false)
- `save_guest_images`: bool (default false)
- `return_embedding`: bool (default false); each face then carries `embedding_f16_b64` (512 little-endian float16 values, base64) and `embedding_model_version`, for a later `/match-template` call. Callers must not log or persist it

`MatchTemplateRequest` carries 2 to 16 `members` (`embedding_f16_b64` and the `quality` object from the same `/identify` response) and `embedding_model_version`. The service aggregates the members with `QualityWeightedMeanAggregator` (`qwmean-v1`: quality-weighted mean of normalized embeddings, renormalized, `app/services/template_aggregation.py`), queries the two nearest centroids, and classifies the template with `classify_match` and the `recognition.template_*` thresholds. The response carries `raw_similarity`, `second_best_similarity`, `member_count`, `aggregator_version`, `template_quality` (means over the three best members by quality score, the same rule the template manifest uses), and `calibrated_confidence` from the template artifact only. A wrong `embedding_model_version`, a member quality from another scorer version, an undecodable embedding, or members that cannot aggregate is a 422 with a `detail.code`. It has no side effects (no image is received).

`BatchIdentifyRequest` fields:
- `images`: list of base64 strings (min 1)
- `include_motion`: bool (default true)
- `include_annotated_image`: bool (default false)
- `save_guest_images`: bool (default false)

Each `FaceDetection` carries `person_id`, `name`, `recognition_state`, `best_candidate_id`, `similarity` (the raw cosine match score, never a probability) and its one-release alias `raw_similarity`, nullable `calibrated_confidence` with `calibration_status` and version fields, head pose, `det_score`, and face geometry (`face_width_px`, `face_height_px`, nullable `interocular_px`). `IdentifyResponse` and each `FrameResult` carry `image_width_px` and `image_height_px` of the decoded input (0 for an undecodable batch frame).

**Side effects:** every unknown face on either identify endpoint is offered to `VisitorStore.record_sighting` (gated by `visitors.clustering_enabled`), independent of `save_guest_images`. On `/identify` only, a request with a `capture_context` (sent by tracking-orchestrator) offers its faces to `CalibrationCaptureService` (gated by `calibration_capture.enabled`), which keeps a stratified sample as calibration candidates in a background worker, and, when the context names a `detection_id` and `calibration_capture.attestation_enabled` is on, holds them in memory for a later wearable attestation. Read-only measurement tools must call `FaceEngine` directly instead of these endpoints.

Batch endpoint handles per-image errors gracefully: bad frames are skipped and the response carries an empty `faces` list for them.

### Visitors

| Method | Path | Response |
| --- | --- | --- |
| `GET` | `/api/v1/visitors/clusters` | Cluster list (optional `status` filter); 409 when clustering is disabled |
| `GET` | `/api/v1/visitors/clusters/{cluster_id}` | Cluster detail with recent sightings |
| `POST` | `/api/v1/visitors/clusters/{cluster_id}/name` | Names the cluster into an enrolled member (the only privileged transition) |
| `POST` | `/api/v1/visitors/clusters/review` | Attributes clusters to an enrolled member, or dismisses them; per-cluster results, records the actor; never enrolls |
| `POST` | `/api/v1/visitors/clusters/{cluster_a}/merge/{cluster_b}` | Merges two clusters |

Calibration review (operator labeling of sampled faces; cognitive-companion proxies it and
presigns the crop keys):

| Method | Path | Response |
| --- | --- | --- |
| `GET` | `/api/v1/calibration/candidates` | Calibration captures, newest first (`state`, `limit`, `cursor`) |
| `POST` | `/api/v1/calibration/candidates/{candidate_id}/label` | Label a capture: member, not enrolled, or not a face |
| `POST` | `/api/v1/calibration/candidates/label-batch` | Label several captures `not_enrolled` or `not_a_face`, one result each; member labels are refused (one face at a time) |
| `POST` | `/api/v1/calibration/candidates/{candidate_id}/promote` | Promote a labeled member capture (meeting quality and interocular floors) into enrollment prototypes and bump `gallery_revision` |
| `POST` | `/api/v1/calibration/candidates/{candidate_id}/reject` | Mark a capture unusable (purged with unlabeled ones) |
| `POST` | `/api/v1/calibration/attestations` | tracking-orchestrator: store the held faces of wearable-attested detections as `attested` captures; returns moved, missed, capped |
| `GET` | `/api/v1/calibration/attestations` | Attested captures, best quality first (`identity_id`, `audited`, `limit`) |
| `GET` | `/api/v1/calibration/attestations/audit-sample` | Uniform random sample of unaudited attested captures (`n`, `seed`) |
| `POST` | `/api/v1/calibration/attestations/{candidate_id}/audit` | Record an audit verdict (`correct`, `wrong`, `not_a_face`) |
| `GET` | `/api/v1/calibration/attestations/audit-summary` | Audit counts, Wilson lower bound, and whether the bar is met |
| `POST` | `/api/v1/calibration/attestations/revoke` | tracking-orchestrator: idempotently revoke all attested captures for one `association_id`; returns `revoked` count and `already_promoted` refs for manual review |
| `GET` | `/api/v1/calibration/attestations/counts` | Per-`association_id` counts (`total`, `active`, `revoked`, `promoted`) of attested captures |
| `DELETE` | `/api/v1/calibration/candidates/{candidate_id}` | Delete a capture and its crop |
| `GET` | `/api/v1/calibration/summary` | Labeling progress per identity, kind, quality decile, camera |

### Motion

| Method | Path | Request Body | Response |
| --- | --- | --- | --- |
| `POST` | `/api/v1/detect-motion` | `MotionDetectionRequest` (min 2 images) | `MotionDetectionResponse` |

Standalone motion detection without enrollment lookups. Includes full trajectory data per track.

---

## 8. Database schema

PostgreSQL (TimescaleDB + pgvector) is the instance shared with the other Cognitive Companion services. `migrations/0001_initial_schema.up.sql` is the frozen baseline. It creates the tables below plus the visitor clustering tables `visitor_clusters` and `visitor_sightings` (hypertable on `seen_at`). Migrations are incremental from 2026-10-20: every change is a new `NNNN_description.up.sql` with a matching `.down.sql`; never edit `0001` and never drop the database to apply a change.

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

Tables added on top of the core tables:

| Table | Type | Purpose |
| --- | --- | --- |
| `visitor_clusters` | Regular | Visitor cluster lifecycle (candidate, surfaced, named, dismissed, attributed), centroid, and review record |
| `visitor_sightings` | Hypertable | Unknown-face sightings (crop key, embedding) per cluster |
| `calibration_candidates` | Regular | Sampled faces for calibration: crop key, embedding, quality features, operator label (0003); wearable-attested state, grade, association, and audit (0006); revocation timestamps and actor (0007) |
| `face_prototypes` | Regular | Pose-binned face prototypes (`frontal`, `left`, `right`, `down`) per member and `recognizer_version`, exact search, `ON DELETE CASCADE` (0005, 0008) |

Migration `0008_recognizer_version` adds `recognizer_version TEXT NOT NULL DEFAULT 'buffalo_l'` to all six embedding-bearing tables (`embeddings`, `centroids` with composite `PRIMARY KEY (person_id, recognizer_version)`, `face_prototypes`, `calibration_candidates`, `visitor_clusters`, and `visitor_sightings`). Its down migration refuses rollback while any table contains rows with `recognizer_version <> 'buffalo_l'`.

List every migration with `ls migrations`.

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

Test files: `test_confidence_contract_characterization.py`, `test_database_guard.py`, `test_dependencies.py`, `test_enrollment_repository_parity.py`, `test_enrollment_store.py`, `test_enrollment_store_inmemory.py`, `test_identify_characterization.py`, `test_match_policy.py`, `test_migration_runner.py`, `test_face_engine.py`, `test_face_geometry.py`, `test_face_quality.py`, `test_guest_store.py`, `test_identification_geometry_contract.py`, `test_identification_quality_contract.py`, `test_quality_breakpoints.py`, `test_identification_visitor_ingest.py`, `test_m10_calibration.py`, `test_m3_face_evidence.py`, `test_visitors_router.py`, `test_visitor_store.py`, `test_visitor_policy.py`, `test_calibration_capture_policy.py`, `test_calibration_candidates_repo.py`, `test_calibration_router.py`, `test_calibration_evaluator_v2.py`, `test_calibration_manifest.py`, `test_calibration_fit_v2.py`, `test_skip_guard.py`, `tests/test_wearable_attestation.py`, `tests/test_attested_label_migration.py`, `tests/test_attestation_revocation.py`, `tests/test_recognizer_profile.py`, `tests/test_enrollment_crop_retention_and_reembed.py`, `tests/test_recognizer_eval.py`; calibration fixtures under `tests/fixtures/identity_integrity/`. `tests/conftest.py` loads `tests/skip_guard.py`, which fails the run on any skipped test when `PID_REQUIRE_FULL_SUITE=1` (set in CI).

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

1. Create a new SQL file in `migrations/` with the next sequence number (e.g., `NNNN_your_change.up.sql`) and a matching `NNNN_your_change.down.sql` that exactly reverses it.
2. Write idempotent DDL (use `IF NOT EXISTS`, `IF EXISTS` where appropriate).
3. The migration runs automatically at next startup via the lifespan. Never edit `0001_initial_schema` and never drop the database to apply a change: enrollment and visitor data are irreplaceable.

### 10.4 Upgrade or swap the face recognizer profile

1. Export the ONNX graph and Triton model config into `continuous-tracking/triton-models/` (for AdaFace IR-50, use `continuous-tracking/triton-models/scripts/export_adaface_onnx.py`; generate and validate Jetson INT8 variants in `triton-models-jetson` when targeting Jetson).
2. Add or verify the profile entry under `face_engine.profiles.<name>` in `config/settings.yaml` (all five Triton model names, `input_size_px`, `alignment_template`, `embedding_dim`, `color_space`, `normalization_mean`, `normalization_std`, `recognition_input_name`, `recognition_output_name`, `preprocessing_version`, and `licence`). Keeping `landmark_3d_model` unchanged preserves validated head-pose Euler angles.
3. If `embedding_dim` differs from 512, add a new migration for the new vector column dimension before running `reembed`.
4. Populate a parallel gallery from retained aligned enrollment crops without disturbing the active recognizer:
   `DATABASE_URL=... TRITON_GRPC_URL=triton:8701 uv run python -m app.tools.reembed --profile <name>`
   Check the reported `skipped_without_crop` count and re-enroll any legacy member who predated crop retention.
5. Evaluate on household labeled holdout captures with `scripts/evaluate_recognizer_swap.py` before switching.
6. Switch `face_engine.active_profile` to `<name>`. Existing calibration artifacts automatically report `degraded_incompatible` (both `recognizer_version` and the profile-scoped `quality_version` change), and Cognitive Companion's Identity Setup `artifact` step blocks until you refit and redeploy single-frame and template calibration artifacts for `<name>` and update `face_id.expected_recognizer_version` and `face_id.expected_quality_version` in `continuous-tracking/tracking-orchestrator/config/settings.yaml`.
7. To roll back, restore `face_engine.active_profile: buffalo_l`, the previous calibration artifacts, and CTS's expected versions; the original `buffalo_l` gallery rows remain intact in Postgres.

---

## 11. External services

| Service | Env var | Required | Purpose |
| --- | --- | --- | --- |
| PostgreSQL (TimescaleDB + pgvector) | `DATABASE_URL` | Required | Face gallery, embeddings, centroids, guest visit records |
| MinIO (S3-compatible) | `MINIO_ENDPOINT`, `MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY`, `MINIO_BUCKET` | Required | Guest image object storage |
| Triton Inference Server | `TRITON_GRPC_URL` | Required | Buffalo_L model execution |

Runtime ONNX and TensorRT artifacts live in the continuous-tracking Triton repositories; the service container holds no model weights. Face embeddings and centroids are stored in PostgreSQL; guest images in MinIO.

This service does not call other Cognitive Companion microservices. It is called BY the Cognitive Companion backend (as the BFF for person identification).

---

## 12. What NOT to do

**Logging and console.**
- Do not use `print()`. Use `logging.getLogger(__name__)`.
- Do not log base64 image data (will flood logs).

**Architecture and layering.**
- Do not instantiate services in routers. Read from `request.app.state`.
- Do not use bare `except:`. Log and return a zero value, or re-raise as an HTTP exception.
- Follow existing patterns; no premature abstractions.
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
| Measure face resolution | `app/services/face_geometry.py`; per-camera reports come from `continuous-tracking/tracking-orchestrator/scripts/face_resolution_report.py` |
| Debug calibration state | `app/calibration/evaluator.py`, `GET /health` (`calibration` block) |
| Debug visitor clustering | `app/services/visitor_store.py` |
| Debug motion detection | `app/services/motion_detector.py` (_classify_direction) |
| Debug guest image uploads | `app/services/guest_store.py` + `app/services/minio_client.py` |
| Find configuration values | `config/settings.yaml` |
| Understand the database schema | `migrations/0001_initial_schema.up.sql` |
| See how the pool is created | `app/main.py` (_init_pool) |
| See test patterns | `tests/test_enrollment_store.py` |
