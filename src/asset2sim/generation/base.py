"""Contract implemented by each single-image generator adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol


class Generator(Protocol):
    model_name: str

    @property
    def parameters(self) -> dict[str, Any]: ...

    def load(self) -> None: ...

    def generate(self, image_path: Path, output_dir: Path) -> dict[str, Any]:
        """Write input.png and mesh.glb; return measurements in the shared units."""
        ...

    def metadata(self) -> dict[str, Any]:
        """Return weight version, runtime environment and raw coordinate conventions."""
        ...

    def recover(self) -> None:
        """Release failed-sample resources before continuing the batch."""
        ...
