"""Dependency tests for the production Triton inference path."""

from __future__ import annotations

import importlib.metadata

from triton_shared.client import TritonGrpcClient


def _major_minor(distribution: str) -> tuple[int, int]:
    version = importlib.metadata.version(distribution)
    major, minor, *_ = version.split(".")
    return int(major), int(minor)


def test_tritonclient_version_supports_async_grpc() -> None:
    """The shared client requires Triton's maintained asyncio gRPC API."""
    assert _major_minor("tritonclient") >= (2, 51)


def test_triton_shared_is_installed() -> None:
    """Production must use the shared Triton client contract."""
    assert importlib.metadata.version("triton-shared")


def test_triton_shared_exports_grpc_client() -> None:
    """The installed Git revision must expose the client used at startup."""
    assert TritonGrpcClient.__module__ == "triton_shared.client.grpc"
