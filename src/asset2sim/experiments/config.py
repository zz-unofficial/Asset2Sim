"""Resolve model defaults and experiment-specific inputs and overrides."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def load_parameters(path: Path, _parents: tuple[Path, ...] = ()) -> dict[str, Any]:
    path = path.resolve()
    if path in _parents:
        raise ValueError(f"Configuration inheritance cycle: {path}")
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ValueError(f"Invalid YAML in {path}: {error}") from error
    if not isinstance(value, dict):
        raise TypeError(f"Expected a YAML mapping: {path}")
    parent = value.pop("extends", None)
    if parent is not None and not isinstance(parent, str):
        raise TypeError("extends must be a YAML file path")
    inherited = (
        load_parameters(path.parent / parent, (*_parents, path)) if parent is not None else {}
    )
    return inherited | value


@dataclass
class ExperimentConfig:
    name: str
    model: str
    model_config: Path
    manifest: Path
    overrides: dict[str, Any] = field(default_factory=dict)
    sample_ids: list[str] | None = None

    @classmethod
    def load(cls, path: Path) -> ExperimentConfig:
        path = path.resolve()
        values = load_parameters(path)
        for key in ("name", "model", "model_config", "manifest"):
            if not isinstance(values.get(key), str) or not values[key]:
                raise ValueError(f"{path}: {key} must be a nonempty string")
        allowed = {"name", "model", "model_config", "manifest", "overrides", "sample_ids"}
        if extra := values.keys() - allowed:
            raise ValueError(f"Unknown experiment fields: {sorted(extra)}")
        if not isinstance(values.get("overrides", {}), dict):
            raise TypeError("overrides must be a mapping")
        selection = values.get("sample_ids")
        if selection is not None and (
            not isinstance(selection, list) or not all(isinstance(v, str) for v in selection)
        ):
            raise ValueError("sample_ids must be a list of strings")
        for key in ("model_config", "manifest"):
            values[key] = (path.parent / values[key]).resolve()
        return cls(**values)

    def parameters(self) -> dict[str, Any]:
        return load_parameters(self.model_config) | self.overrides
