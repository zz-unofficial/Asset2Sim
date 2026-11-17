#!/usr/bin/env bash
set -euo pipefail
# Run explicitly on the selected NVIDIA server; no weights are downloaded here.
if [[ "${1:-}" != "--server" ]]; then
    echo "Usage on the GPU server: bash scripts/setup_physx_omni.sh --server" >&2
    exit 2
fi
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
command -v conda >/dev/null
if [[ -n "${CUDA_HOME:-}" ]]; then
    export PATH="$CUDA_HOME/bin:$PATH"
fi
command -v nvcc >/dev/null
command -v nvidia-smi >/dev/null
nvidia-smi
if git -C "$REPO_ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git submodule update --init --recursive third_party/PhysX-Omni
elif [[ ! -f "$REPO_ROOT/third_party/PhysX-Omni/1vlm_demo.py" ]]; then
    echo "PhysX-Omni source is missing; provide the upstream checkout first." >&2
    exit 2
fi
CONDA_BASE="$(conda info --base)"
set +u
source "$CONDA_BASE/etc/profile.d/conda.sh"
if ! conda run -n asset2sim-physx-omni python --version >/dev/null 2>&1; then
    conda env create -f environments/physx_omni/environment.yml
fi
conda activate asset2sim-physx-omni
# PyTorch CUDA wheels keep library headers under their individual packages.
for include_dir in "$CONDA_PREFIX"/lib/python*/site-packages/nvidia/*/include; do
    [[ -d "$include_dir" ]] && export CPATH="$include_dir${CPATH:+:$CPATH}"
done
# Candidate stack: CUDA Toolkit 12.1 must be provided by the server image.
nvcc --version | grep -q 'release 12.1' || {
    echo "Select a CUDA Toolkit 12.1 server image before installation." >&2
    exit 2
}
if [[ "${PHYSX_BASE_READY:-0}" != 1 ]]; then
    python -m pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121
fi
cd "$REPO_ROOT/third_party/PhysX-Omni"
if [[ -n "${PHYSX_EXTENSIONS_ROOT:-}" ]]; then
    for source_dir in nvdiffrast diffoctreerast mip-splatting/submodules/diff-gaussian-rasterization; do
        [[ -f "$PHYSX_EXTENSIONS_ROOT/$source_dir/setup.py" ]] || {
            echo "Missing prepared extension source: $source_dir" >&2
            exit 2
        }
    done
    if [[ "${PHYSX_BASE_READY:-0}" == 1 ]]; then
        python -c 'import torch, torchvision, xformers, flash_attn, spconv.pytorch, kaolin'
    else
        source ./setup.sh --basic --xformers --flash-attn --spconv --kaolin
    fi
    python -m pip install --no-build-isolation "$PHYSX_EXTENSIONS_ROOT/nvdiffrast"
    python -m pip install --no-build-isolation "$PHYSX_EXTENSIONS_ROOT/diffoctreerast"
    python -m pip install --no-build-isolation "$PHYSX_EXTENSIONS_ROOT/mip-splatting/submodules/diff-gaussian-rasterization"
else
    source ./setup.sh --basic --xformers --flash-attn --diffoctreerast --spconv --mipgaussian --kaolin --nvdiffrast
fi
set -eu
cd "$REPO_ROOT"
python -m pip install -r environments/physx_omni/requirements.txt
python -m pip install -e .
python -m pip check
python -m pip freeze > "$CONDA_PREFIX/asset2sim-installed-packages.txt"
echo "Environment installed. Run the first object pilot before scheduling a batch."
