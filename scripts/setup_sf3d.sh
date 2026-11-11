#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_BASE="$(conda info --base)"
# Some Conda CUDA activation hooks read optional, unset shell variables.
set +u
source "$CONDA_BASE/etc/profile.d/conda.sh"
if ! conda run -n asset2sim-sf3d python --version >/dev/null 2>&1; then
    conda env create -f "$REPO_ROOT/environments/sf3d/environment.yml"
fi
conda activate asset2sim-sf3d
set -u

git -C "$REPO_ROOT" submodule update --init third_party/stable-fast-3d
python -m pip install torch==2.8.0 torchvision==0.23.0 \
    --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r "$REPO_ROOT/environments/sf3d/requirements.txt"

export CUDA_HOME="$CONDA_PREFIX"
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export MAX_JOBS="${MAX_JOBS:-4}"
# Use the active GPU instead of inheriting an unrelated all-architectures build list.
export TORCH_CUDA_ARCH_LIST="${SF3D_CUDA_ARCH_LIST:-$(python -c 'import torch; print("%d.%d" % torch.cuda.get_device_capability())')}"
# PyTorch's CUDA wheels already contain these headers; do not download duplicate SDK libraries.
SF3D_CUDA_HEADERS="$(python -c 'import glob, sysconfig; print(":".join(glob.glob(sysconfig.get_paths()["purelib"] + "/nvidia/*/include")))')"
export CPATH="$SF3D_CUDA_HEADERS${CPATH:+:$CPATH}"
cd "$REPO_ROOT/third_party/stable-fast-3d"
python -m pip install --no-build-isolation ./texture_baker ./uv_unwrapper
cd "$REPO_ROOT"
python -m pip install --no-build-isolation -e .

