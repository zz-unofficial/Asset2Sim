#!/usr/bin/env bash
# Run the public OmniObject3D experiment and combine it with the completed GSO run.
set -eo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
RUN_ID="${1:-run01}"
if [[ ! "$RUN_ID" =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]*$ ]]; then
    printf 'Use a simple run ID, for example run01.\n' >&2
    exit 1
fi

if command -v conda >/dev/null 2>&1; then
    SF3D_CONDA_BASE="$(conda info --base)"
else
    SF3D_CONDA_BASE="${ASSET2SIM_CONDA_BASE:-$HOME/miniconda3}"
fi
source "$SF3D_CONDA_BASE/etc/profile.d/conda.sh"
conda activate asset2sim-sf3d
set -u
export PYTHONUNBUFFERED=1

GENERATED="data/generated/sf3d/paper_omni/$RUN_ID"
EVALUATED="results/raw/sf3d_paper_omni/$RUN_ID"
REPORT="results/metrics/sf3d_paper_public/$RUN_ID"
LAUNCH="results/raw/sf3d_paper_omni/${RUN_ID}_launcher"
GSO="results/raw/sf3d_paper_gso/$RUN_ID"
GSO_RUN="data/generated/sf3d/paper_gso/$RUN_ID"
for target in "$GENERATED" "$EVALUATED" "$REPORT" "$LAUNCH"; do
    if [[ -e "$target" ]]; then
        printf 'Output already exists: %s. Choose a new run ID.\n' "$target" >&2
        exit 1
    fi
done
for category in bed bumbag battery book; do
    test -f "data/raw/omniobject3d/$category.tar.gz" || {
        printf 'Missing archive: data/raw/omniobject3d/%s.tar.gz\n' "$category" >&2
        exit 1
    }
done
test -f "$GSO_RUN/run.json"
mkdir -p "$LAUNCH"
exec > >(tee -a "$LAUNCH/pipeline.log") 2>&1

finish() {
    local task_exit=$?
    if (( task_exit == 0 )); then
        printf 'completed\n' > "$LAUNCH/status.txt"
        printf '\nComplete. Combined report: %s/comparison.md\n' "$REPORT"
    else
        printf 'failed: exit %s\n' "$task_exit" > "$LAUNCH/status.txt"
        printf '\nStopped with exit %s. See %s/pipeline.log\n' "$task_exit" "$LAUNCH" >&2
    fi
}
trap finish EXIT

printf 'prepare_data\n' > "$LAUNCH/status.txt"
printf '[1/4] Prepare 24 OmniObject3D objects and 16 views per object\n'
if [[ -f data/manifests/sf3d_omni_public.jsonl ]]; then
    printf 'Using the existing prepared OmniObject3D dataset.\n'
else
    asset2sim prepare-paper-data --config configs/datasets/sf3d_omni_public.yaml \
        --output data/processed/omniobject3d \
        --manifest data/manifests/sf3d_omni_public.jsonl
fi

printf 'generate\n' > "$LAUNCH/status.txt"
printf '\n[2/4] Generate textured meshes with SF3D\n'
asset2sim run --config configs/experiments/sf3d_omni_public.yaml --output "$GENERATED"

printf 'evaluate\n' > "$LAUNCH/status.txt"
printf '\n[3/4] Measure CD, F-scores, PSNR, SSIM and LPIPS\n'
asset2sim evaluate --run "$GENERATED" \
    --config configs/evaluation/sf3d_paper_public.yaml --output "$EVALUATED"
if [[ ! -e "$GSO" ]]; then
    asset2sim evaluate --run "$GSO_RUN" \
        --config configs/evaluation/sf3d_paper_public.yaml --output "$GSO"
fi

printf 'report\n' > "$LAUNCH/status.txt"
printf '\n[4/4] Combine GSO and OmniObject3D results and export figures\n'
asset2sim paper-report "$GSO" "$EVALUATED" --output "$REPORT"
