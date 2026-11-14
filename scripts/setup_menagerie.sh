#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
MENAGERIE_DIR="$REPO_ROOT/.local/mujoco_menagerie"
if [[ ! -e "$MENAGERIE_DIR" ]]; then
    git clone --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git "$MENAGERIE_DIR"
    git -C "$MENAGERIE_DIR" sparse-checkout set universal_robots_ur5e robotiq_2f85
    git -C "$MENAGERIE_DIR" checkout 370c43965323c0944b433b202aac5058b5f907a9
fi
