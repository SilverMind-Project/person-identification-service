"""Unit tests for the Triton-backed Buffalo_L face engine."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from app.services.face_engine import FaceEngine

FloatArray = npt.NDArray[np.float32]


class _FakeTritonClient:
    def __init__(
        self,
        unavailable_model: str | None = None,
        *,
        read_only_outputs: bool = False,
    ) -> None:
        self.unavailable_model = unavailable_model
        self.read_only_outputs = read_only_outputs
        self.calls: list[str] = []

    async def is_model_ready(self, model_name: str) -> bool:
        return model_name != self.unavailable_model

    async def infer(
        self,
        model_name: str,
        inputs: list[tuple[str, FloatArray]],
        output_names: list[str],
    ) -> dict[str, FloatArray]:
        self.calls.append(model_name)
        assert inputs[0][1].dtype == np.float32

        if model_name == "face-detector-scrfd":
            outputs = _empty_scrfd_outputs()
            outputs["448"][0, 0] = 0.95
            outputs["451"][0] = np.array([0.0, 0.0, 2.0, 2.0], dtype=np.float32)
            outputs["454"][0] = np.array(
                [0.5, 0.5, 1.2, 0.5, 0.8, 0.9, 0.5, 1.3, 1.1, 1.3],
                dtype=np.float32,
            )
            result = {name: outputs[name] for name in output_names}
            if self.read_only_outputs:
                for output in result.values():
                    output.setflags(write=False)
            return result
        if model_name == "face-recognition-arcface":
            embedding = np.arange(1, 513, dtype=np.float32)[None]
            if self.read_only_outputs:
                embedding.setflags(write=False)
            return {"683": embedding}
        if model_name == "face-landmark-2d106":
            output = np.zeros((1, 212), dtype=np.float32)
            if self.read_only_outputs:
                output.setflags(write=False)
            return {"fc1": output}
        if model_name == "face-landmark-3d68":
            output = np.zeros((1, 3309), dtype=np.float32)
            if self.read_only_outputs:
                output.setflags(write=False)
            return {"fc1": output}
        if model_name == "face-attribute-genderage":
            output = np.array([[0.1, 0.9, 0.42]], dtype=np.float32)
            if self.read_only_outputs:
                output.setflags(write=False)
            return {"fc1": output}
        raise AssertionError(f"Unexpected model: {model_name}")


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


@pytest.mark.asyncio
async def test_detect_faces_runs_complete_buffalo_pack() -> None:
    client = _FakeTritonClient()
    engine = FaceEngine(client)

    faces = await engine.detect_faces(np.zeros((640, 640, 3), dtype=np.uint8))

    assert len(faces) == 1
    assert np.linalg.norm(faces[0].embedding) == pytest.approx(1.0)
    assert faces[0].landmarks is not None
    assert faces[0].landmarks.shape == (106, 2)
    assert faces[0].gender == 1
    assert faces[0].age == 42
    assert set(client.calls) == {
        "face-detector-scrfd",
        "face-recognition-arcface",
        "face-landmark-2d106",
        "face-landmark-3d68",
        "face-attribute-genderage",
    }


@pytest.mark.asyncio
async def test_detect_faces_accepts_read_only_triton_outputs() -> None:
    client = _FakeTritonClient(read_only_outputs=True)
    engine = FaceEngine(client)

    faces = await engine.detect_faces(np.zeros((640, 640, 3), dtype=np.uint8))

    assert len(faces) == 1
    assert faces[0].embedding.flags.writeable
    assert np.linalg.norm(faces[0].embedding) == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_validate_models_fails_when_any_model_is_unavailable() -> None:
    client = _FakeTritonClient(unavailable_model="face-landmark-3d68")
    engine = FaceEngine(client)

    with pytest.raises(RuntimeError, match="face-landmark-3d68"):
        await engine.validate_models()


def test_invalid_model_profile_fails_at_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _config_get(key: str, default: object | None = None) -> object | None:
        if key == "face_engine.model_profile":
            return "automatic"
        return default

    monkeypatch.setattr("app.services.face_engine.config.get", _config_get)

    with pytest.raises(ValueError, match="model_profile"):
        FaceEngine(_FakeTritonClient())
