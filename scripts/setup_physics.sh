#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_BASE="$(conda info --base)"
set +u
source "$CONDA_BASE/etc/profile.d/conda.sh"
if ! conda run -n asset2sim-physics python --version >/dev/null 2>&1; then
    conda env create -f "$REPO_ROOT/environments/physics/environment.yml"
fi
conda activate asset2sim-physics
set -u
python -m pip install -r "$REPO_ROOT/environments/physics/requirements.txt"
python -m pip install -e "$REPO_ROOT"
