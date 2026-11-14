#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
conda run --no-capture-output -n asset2sim-sf3d \
    python -m asset2sim.experiments.benchmark "$@"
