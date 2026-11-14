#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
git submodule update --init --recursive third_party/TRELLIS.2
if ! command -v conda >/dev/null; then
    echo "Conda is required on the GPU server" >&2
    exit 2
fi
CONDA_BASE="$(conda info --base)"
set +u
source "$CONDA_BASE/etc/profile.d/conda.sh"
if ! conda run -n trellis2 python --version >/dev/null 2>&1; then
    conda env create -f "$REPO_ROOT/environments/trellis2/environment.yml"
fi
conda activate trellis2
export CUDA_HOME="$CONDA_PREFIX"
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r "$REPO_ROOT/environments/trellis2/requirements.txt"
cd "$REPO_ROOT/third_party/TRELLIS.2"
source ./setup.sh --flash-attn --nvdiffrast --nvdiffrec --cumesh --o-voxel --flexgemm
set -u
cd "$REPO_ROOT"
python -m pip install -e .
python -m pip freeze > "$CONDA_PREFIX/asset2sim-installed-packages.txt"
echo "TRELLIS.2 server environment ready: conda activate trellis2"
