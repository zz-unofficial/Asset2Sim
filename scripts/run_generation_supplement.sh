#!/usr/bin/env bash
# Local D2 and matched low-memory SPAR3D seed experiments.
set -euo pipefail
cd "$(dirname "$0")/.."
conda_bin="${CONDA_EXE:-$HOME/miniconda3/bin/conda}"
export PYOPENGL_PLATFORM=egl PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
generation_status=0
for model in sf3d spar3d; do
  "$conda_bin" run --no-capture-output -n "asset2sim-$model" asset2sim run \
    --config "configs/experiments/${model}_d2_local_20261116.yaml" \
    --output "data/generated/$model/generation_core/d2_no_cache_local_20261116" || generation_status=1
done
for seed in 0 1 2; do
  "$conda_bin" run --no-capture-output -n asset2sim-spar3d asset2sim run \
    --config "configs/experiments/spar3d_seed${seed}_local_20261116.yaml" \
    --output "data/generated/spar3d/generation_core/seed${seed}_local_20261116" || generation_status=1
done
"$conda_bin" run --no-capture-output -n asset2sim-sf3d \
  python -m asset2sim.evaluation.benchmark \
  --manifest data/manifests/benchmark_d2_local_20261116.jsonl \
  --run sf3d=data/generated/sf3d/generation_core/d2_no_cache_local_20261116 \
  --run spar3d=data/generated/spar3d/generation_core/d2_no_cache_local_20261116 \
  --config configs/evaluation/generation_benchmark_local.yaml \
  --output results/raw/generation_benchmark/d2_no_cache_local_20261116
for seed in 0 1 2; do
  "$conda_bin" run --no-capture-output -n asset2sim-sf3d \
    python -m asset2sim.evaluation.benchmark \
    --manifest data/manifests/gso_seed8_local_20261116.jsonl \
    --run "spar3d=data/generated/spar3d/generation_core/seed${seed}_local_20261116" \
    --config configs/evaluation/generation_benchmark_local.yaml \
    --output "results/raw/generation_benchmark/seed${seed}_local_20261116"
done

exit "$generation_status"
