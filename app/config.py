"""YAML-based configuration with ${ENV_VAR} interpolation."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

_CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
_ENV_PATTERN = re.compile(r"\$\{(\w+)(?::([^}]*))?\}")

_settings: dict[str, Any] | None = None


def _interpolate(value: Any) -> Any:
    if isinstance(value, str):
        def _replace(m: re.Match) -> str:
            env_var = m.group(1)
            default = m.group(2)
            return os.environ.get(env_var, default if default is not None else m.group(0))
        return _ENV_PATTERN.sub(_replace, value)
    if isinstance(value, dict):
        return {k: _interpolate(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate(v) for v in value]
    return value


def _load() -> dict[str, Any]:
    global _settings
    path = _CONFIG_DIR / "settings.yaml"
    if path.exists():
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        _settings = _interpolate(raw)
    else:
        _settings = {}
    return _settings


def get(key: str, default: Any = None) -> Any:
    """Dot-notation config access: ``get("face_engine.model_name")``."""
    cfg = _settings if _settings is not None else _load()
    parts = key.split(".")
    node: Any = cfg
    for part in parts:
        if isinstance(node, dict):
            node = node.get(part)
        else:
            return default
        if node is None:
            return default
    return node


def reload() -> None:
    global _settings
    _settings = None
    _load()


# Eager load on import
_load()
