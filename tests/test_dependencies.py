"""Dependency tests enforcing system stability and API compatibility."""

import importlib.metadata

import pytest


def test_insightface_compatibility():
    """Ensure our environment maintains a version of scikit-image compatible with InsightFace's APIs.

    InsightFace 0.7.3 relies on skimage's `estimate()` method (deprecated in >= 0.26).
    This test verifies that we are locked to a version strictly less than 0.26
    to prevent FutureWarnings and future removal breakages.
    """
    try:
        skimage_version = importlib.metadata.version("scikit-image")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("scikit-image not found in current non-Docker testing environment.")
        return

    # Split version string, e.g. "0.25.0" -> ["0", "25", "0"]
    major, minor, *_rest = skimage_version.split(".")

    assert int(major) == 0, (
        f"scikit-image major version {major} is not compatible with InsightFace 0.7.3"
    )
    assert int(minor) < 26, (
        f"scikit-image version {skimage_version} yields FutureWarnings in InsightFace 0.7.3. Must be < 0.26"
    )
