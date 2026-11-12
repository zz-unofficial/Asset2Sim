#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_BASE="$(conda info --base)"
set +u
source "$CONDA_BASE/etc/profile.d/conda.sh"
if ! conda run -n asset2sim-spar3d python --version >/dev/null 2>&1; then
    conda env create -f "$REPO_ROOT/environments/spar3d/environment.yml"
fi
conda activate asset2sim-spar3d
set -u
git -C "$REPO_ROOT" submodule update --init third_party/stable-point-aware-3d
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install --no-build-isolation -r "$REPO_ROOT/environments/spar3d/requirements.txt"
export CUDA_HOME="$CONDA_PREFIX" CC=/usr/bin/gcc CXX=/usr/bin/g++ MAX_JOBS="${MAX_JOBS:-4}"
export TORCH_CUDA_ARCH_LIST="${SPAR3D_CUDA_ARCH_LIST:-$(python -c 'import torch; print("%d.%d" % torch.cuda.get_device_capability())')}"
SPAR3D_CUDA_HEADERS="$(python -c 'import glob, sysconfig; print(":".join(glob.glob(sysconfig.get_paths()["purelib"] + "/nvidia/*/include")))')"
export CPATH="$SPAR3D_CUDA_HEADERS${CPATH:+:$CPATH}"
cd "$REPO_ROOT/third_party/stable-point-aware-3d"
python -m pip install --no-build-isolation ./texture_baker ./uv_unwrapper
cd "$REPO_ROOT"
python -m pip install --no-build-isolation -e .
