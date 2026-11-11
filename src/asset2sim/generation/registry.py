"""Lazy adapter registration keeps model environments independent."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import Any

from asset2sim.generation.base import Generator

MODELS = {"sf3d": "asset2sim.generation.sf3d"}


def _module(name: str) -> ModuleType:
    if name not in MODELS:
        raise ValueError(f"Unknown model {name!r}; available models: {', '.join(MODELS)}")
    return import_module(MODELS[name])


def create_generator(name: str, parameters: dict[str, Any]) -> Generator:
    return _module(name).create_generator(parameters)


def fetch_assets(name: str, parameters: dict[str, Any]) -> Path:
    return _module(name).fetch_assets(parameters)
