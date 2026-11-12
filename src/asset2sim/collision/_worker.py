"""Run the native CoACD library in a separate process so failures are recorded."""

import json
import sys
from pathlib import Path

import coacd
import numpy as np


def main() -> None:
    directory = Path(sys.argv[1])
    with np.load(directory / "input.npz") as data:
        mesh = coacd.Mesh(data["vertices"], data["faces"])
    settings = json.loads((directory / "settings.json").read_text())
    coacd.set_log_level("error")
    components = coacd.run_coacd(mesh, **settings)
    arrays = {
        f"{kind}_{i}": array
        for i, part in enumerate(components)
        for kind, array in zip(("vertices", "faces"), part)
    }
    np.savez_compressed(directory / "output.npz", count=len(components), **arrays)


if __name__ == "__main__":
    main()
