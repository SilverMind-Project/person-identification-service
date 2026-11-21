"""Tests for RecognizerProfile, FaceEngine profile wiring, wire aliases, and /health licence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.calibration.evaluator import CalibrationEvaluator
from app.models.identification import FaceDetection
from app.routers import health
from app.services.face_engine import FaceEngine
from app.services.face_quality import HeuristicFaceQualityScorer, HeuristicQualityParams
from app.services.recognizer_profile import (
    ADAFACE_WEBFACE4M_LICENCE,
    BUFFALO_L_PROFILE,
    DEFAULT_ARCFACE_TEMPLATE,
    RecognizerProfile,
    get_recognizer_profile,
    load_active_recognizer_profile,
)

FloatArray = npt.NDArray[np.float32]
_SCORER = HeuristicFaceQualityScorer(HeuristicQualityParams())


def _empty_scrfd_outputs() -> dict[str, FloatArray]:
    return {
        "448": np.zeros((12800, 1), dtype=np.float32),
        "471": np.zeros((3200, 1), dtype=np.float32),
        "494": np.zeros((800, 1), dtype=np.float32),
        "451": np.zeros((12800, 4), dtype=np.float32),
        "474": np.zeros((3200, 4), dtype=np.float32),
        "497": np.zeros((800, 4), dtype=np.float32),
        "454": np.zeros((12800, 10), dtype=np.float32),
        "477": np.zeros((3200, 10), dtype=np.float32),
        "500": np.zeros((800, 10), dtype=np.float32),
    }


class _RecordingTritonClient:
    def __init__(self, *, recognition_model: str, output_name: str) -> None:
        self.recognition_model = recognition_model
        self.output_name = output_name
        self.calls: list[tuple[str, str, FloatArray]] = []

    async def is_model_ready(self, model_name: str) -> bool:
        return True

    async def infer(
        self,
        model_name: str,
        inputs: list[tuple[str, FloatArray]],
        output_names: list[str],
    ) -> dict[str, FloatArray]:
        self.calls.append((model_name, inputs[0][0], inputs[0][1].copy()))
        if model_name == "face-detector-scrfd":
            outputs = _empty_scrfd_outputs()
            outputs["448"][0, 0] = 0.95
            outputs["451"][0] = np.array([0.0, 0.0, 2.0, 2.0], dtype=np.float32)
            outputs["454"][0] = np.array(
                [0.5, 0.5, 1.2, 0.5, 0.8, 0.9, 0.5, 1.3, 1.1, 1.3],
                dtype=np.float32,
            )
            return {name: outputs[name] for name in output_names}
        if model_name == self.recognition_model:
            return {self.output_name: np.ones((1, 512), dtype=np.float32)}
        if model_name == "face-landmark-2d106":
            return {"fc1": np.zeros((1, 212), dtype=np.float32)}
        if model_name == "face-landmark-3d68":
            return {"fc1": np.zeros((1, 3309), dtype=np.float32)}
        if model_name == "face-attribute-genderage":
            return {"fc1": np.array([[0.1, 0.9, 0.42]], dtype=np.float32)}
        raise AssertionError(f"Unexpected model: {model_name}")


def test_unknown_active_profile_fails_startup() -> None:
    cfg = {
        "active_profile": "does_not_exist",
        "profiles": {
            "buffalo_l": {
                "recognizer_version": "buffalo_l",
                "licence": "non-commercial",
            }
        },
    }
    with pytest.raises(ValueError, match="Unknown face recognizer profile 'does_not_exist'"):
        load_active_recognizer_profile(cfg)


def test_invalid_alignment_template_shape_is_rejected() -> None:
    cfg = {
        "active_profile": "bad_tpl",
        "profiles": {
            "bad_tpl": {
                "recognizer_version": "bad_tpl",
                "licence": "test",
                "alignment_template": [[10.0, 20.0], [30.0, 40.0]],
            }
        },
    }
    with pytest.raises(ValueError, match="alignment_template"):
        get_recognizer_profile(cfg, "bad_tpl")


def test_non_finite_template_or_invalid_color_space_is_rejected() -> None:
    with pytest.raises(ValueError, match="alignment_template"):
        RecognizerProfile(
            name="nan_tpl",
            recognizer_version="nan_tpl",
            detector_model="d",
            recognition_model="r",
            landmark_2d_model="l2",
            landmark_3d_model="l3",
            genderage_model="g",
            alignment_template=(
                (float("nan"), 51.0),
                (73.0, 51.0),
                (56.0, 71.0),
                (41.0, 92.0),
                (70.0, 92.0),
            ),
            licence="test",
        )
    with pytest.raises(ValueError, match="color_space"):
        RecognizerProfile(
            name="bad_cs",
            recognizer_version="bad_cs",
            detector_model="d",
            recognition_model="r",
            landmark_2d_model="l2",
            landmark_3d_model="l3",
            genderage_model="g",
            color_space="hsv",  # type: ignore[arg-type]
            licence="test",
        )


def test_scoped_quality_version_preserves_buffalo_l_and_qualifies_new_profiles() -> None:
    assert BUFFALO_L_PROFILE.scoped_quality_version("heuristic-v1") == "heuristic-v1"
    adaface = RecognizerProfile(
        name="adaface_ir50_webface4m",
        recognizer_version="adaface_ir50_webface4m",
        detector_model="face-detector-scrfd",
        recognition_model="face-recognition-adaface-ir50",
        landmark_2d_model="face-landmark-2d106",
        landmark_3d_model="face-landmark-3d68",
        genderage_model="face-attribute-genderage",
        color_space="bgr",
        recognition_input_name="input",
        recognition_output_name="embedding",
        licence=ADAFACE_WEBFACE4M_LICENCE,
    )
    assert adaface.scoped_quality_version("heuristic-v1") == "heuristic-v1@adaface_ir50_webface4m"
    assert adaface.templates_compatible(BUFFALO_L_PROFILE) is True


async def test_face_engine_honors_bgr_profile_and_scopes_quality_version() -> None:
    adaface = RecognizerProfile(
        name="adaface_ir50_webface4m",
        recognizer_version="adaface_ir50_webface4m",
        detector_model="face-detector-scrfd",
        recognition_model="face-recognition-adaface-ir50",
        landmark_2d_model="face-landmark-2d106",
        landmark_3d_model="face-landmark-3d68",
        genderage_model="face-attribute-genderage",
        alignment_template=DEFAULT_ARCFACE_TEMPLATE,
        color_space="bgr",
        recognition_input_name="input",
        recognition_output_name="embedding",
        licence=ADAFACE_WEBFACE4M_LICENCE,
    )
    client = _RecordingTritonClient(
        recognition_model="face-recognition-adaface-ir50", output_name="embedding"
    )
    engine = FaceEngine(client, _SCORER, adaface)
    assert engine.quality_version == "heuristic-v1@adaface_ir50_webface4m"

    # Solid BGR image (B=255, G=0, R=0) so BGR vs RGB channel ordering is directly falsifiable.
    img = np.zeros((640, 640, 3), dtype=np.uint8)
    img[:, :, 0] = 255
    [face] = await engine.detect_faces(img)
    assert face.quality is not None
    assert face.quality.version == "heuristic-v1@adaface_ir50_webface4m"
    assert face.aligned_crop_bgr is not None
    assert face.aligned_crop_bgr.shape == (112, 112, 3)

    rec_calls = [c for c in client.calls if c[0] == "face-recognition-adaface-ir50"]
    assert len(rec_calls) == 1
    _, input_name, tensor = rec_calls[0]
    assert input_name == "input"
    assert tensor.shape == (1, 3, 112, 112)
    # Channel 0 in BGR is Blue (255 -> +1.0); channel 2 is Red (0 -> -1.0).
    assert float(tensor[0, 0].max()) == pytest.approx(1.0)
    assert float(tensor[0, 2].max()) == pytest.approx(-1.0)


async def test_embed_aligned_crop_validates_shape_and_runs_only_recognition_model() -> None:
    client = _RecordingTritonClient(recognition_model="face-recognition-arcface", output_name="683")
    engine = FaceEngine(client, _SCORER, BUFFALO_L_PROFILE)

    with pytest.raises(ValueError, match="Expected aligned BGR crop"):
        await engine.embed_aligned_crop(np.zeros((64, 64, 3), dtype=np.uint8))

    vec = await engine.embed_aligned_crop(np.full((112, 112, 3), 127, dtype=np.uint8))
    assert vec.shape == (512,)
    assert np.linalg.norm(vec) == pytest.approx(1.0)
    assert [name for name, _, _ in client.calls] == ["face-recognition-arcface"]


def test_face_detection_syncs_recognizer_version_and_arcface_alias() -> None:
    from_canonical = FaceDetection(
        person_id="alice",
        name="Alice",
        bbox=[0.0, 0.0, 10.0, 10.0],
        recognizer_version="adaface_ir50_webface4m",
    ).model_dump()
    assert from_canonical["recognizer_version"] == "adaface_ir50_webface4m"
    assert from_canonical["arcface_model_version"] == "adaface_ir50_webface4m"

    from_legacy = FaceDetection(
        person_id="alice",
        name="Alice",
        bbox=[0.0, 0.0, 10.0, 10.0],
        arcface_model_version="buffalo_l",
    ).model_dump()
    assert from_legacy["recognizer_version"] == "buffalo_l"
    assert from_legacy["arcface_model_version"] == "buffalo_l"


def test_calibration_evaluator_binds_to_recognizer_version_and_reads_legacy_key(
    tmp_path: Path,
) -> None:
    artifact: dict[str, Any] = {
        "schema_version": 1,
        "artifact_version": "v-rec-1",
        "method": "logistic",
        "recognizer_version": "adaface_ir50_webface4m",
        "model_profile": "full",
        "preprocessing_version": "v1",
        "fitted_at": "2026-11-21T00:00:00Z",
        "coef": 4.0,
        "intercept": -1.5,
    }
    path = tmp_path / "adaface_cal.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")

    ok = CalibrationEvaluator.from_artifact_path(
        path, expected_recognizer_version="adaface_ir50_webface4m"
    )
    assert ok.health() == "ready"
    assert ok.recognizer_version() == "adaface_ir50_webface4m"
    assert ok.arcface_model_version() == "adaface_ir50_webface4m"

    mismatch = CalibrationEvaluator.from_artifact_path(
        path, expected_recognizer_version="buffalo_l"
    )
    assert mismatch.health() == "degraded_incompatible"
    assert mismatch.predict({"similarity": 0.9}, "alice") is None


def test_health_endpoint_exposes_active_profile_recognizer_version_and_licence() -> None:
    class _Store:
        gallery_representation_mode = "shadow"
        active_representation = "centroid"

        async def member_count(self) -> int:
            return 2

    class _Engine:
        endpoint = "triton:8701"
        model_profile = "full"
        profile = BUFFALO_L_PROFILE
        recognizer_version = BUFFALO_L_PROFILE.recognizer_version
        licence = BUFFALO_L_PROFILE.licence
        model_names = BUFFALO_L_PROFILE.model_names

    app = FastAPI()
    app.include_router(health.router)
    app.state.face_engine = _Engine()
    app.state.enrollment_store = _Store()
    app.state.calibration_evaluator = CalibrationEvaluator(None, "degraded_missing")
    app.state.template_calibration_evaluator = CalibrationEvaluator(None, "degraded_missing")

    body = TestClient(app).get("/health").json()
    assert body["active_profile"] == "buffalo_l"
    assert body["recognizer_version"] == "buffalo_l"
    assert "non-commercial research purposes only" in body["licence"]
    assert "recognizer_version" in body["calibration"]
    assert "arcface_model_version" in body["calibration"]
    assert "recognizer_version" in body["template_calibration"]
