#!/usr/bin/env bash
set -eo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
RUN_ID="${1:-run01}"
DATASET="${2:-all}"
[[ "$RUN_ID" =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]*$ ]] || exit 1
[[ "$DATASET" == all || "$DATASET" == gso || "$DATASET" == omni ]] || exit 1
if command -v conda >/dev/null 2>&1; then
    SPAR3D_CONDA_BASE="$(conda info --base)"
else
    SPAR3D_CONDA_BASE="${ASSET2SIM_CONDA_BASE:-$HOME/miniconda3}"
fi
source "$SPAR3D_CONDA_BASE/etc/profile.d/conda.sh"
conda activate asset2sim-spar3d
set -u
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
for dataset in gso omni; do
    [[ "$DATASET" == all || "$DATASET" == "$dataset" ]] || continue
    for output in "data/generated/spar3d/paper_${dataset}/$RUN_ID" \
        "results/raw/spar3d_paper_${dataset}/$RUN_ID" \
        "results/metrics/spar3d_${dataset}_comparison/$RUN_ID"; do
        [[ ! -e "$output" ]] || { printf 'Output exists: %s\n' "$output" >&2; exit 1; }
    done
    test -f "results/raw/sf3d_paper_${dataset}/run02/evaluation.json"
done
SUFFIX=""
[[ "$DATASET" == all ]] || SUFFIX="_${DATASET}"
LAUNCH="results/raw/spar3d_paper/${RUN_ID}${SUFFIX}_launcher"
mkdir -p "$(dirname "$LAUNCH")"
mkdir "$LAUNCH"
exec > >(tee "$LAUNCH/pipeline.log") 2>&1
finish() {
    local code=$?
    if (( code == 0 )); then printf 'completed\n' > "$LAUNCH/status.txt"
    else printf 'failed: exit %s\n' "$code" > "$LAUNCH/status.txt"; fi
}
trap finish EXIT
printf 'fetch_weights\n' > "$LAUNCH/status.txt"
asset2sim fetch --model spar3d --config configs/models/spar3d.yaml
for dataset in gso omni; do
    [[ "$DATASET" == all || "$DATASET" == "$dataset" ]] || continue
    printf 'generate_%s\n' "$dataset" > "$LAUNCH/status.txt"
    asset2sim run --config "configs/experiments/spar3d_${dataset}_public.yaml" \
        --output "data/generated/spar3d/paper_${dataset}/$RUN_ID"
    printf 'evaluate_%s\n' "$dataset" > "$LAUNCH/status.txt"
    asset2sim evaluate --run "data/generated/spar3d/paper_${dataset}/$RUN_ID" \
        --config configs/evaluation/sf3d_paper_public.yaml \
        --output "results/raw/spar3d_paper_${dataset}/$RUN_ID"
    printf 'report_%s\n' "$dataset" > "$LAUNCH/status.txt"
    asset2sim paper-report "results/raw/sf3d_paper_${dataset}/run02" \
        "results/raw/spar3d_paper_${dataset}/$RUN_ID" \
        --output "results/metrics/spar3d_${dataset}_comparison/$RUN_ID"
done
