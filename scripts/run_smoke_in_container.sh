#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_DIR="${1:-/fast/jyeon/gsae/artifacts/smoke_condor}"

exec "${ROOT}/scripts/run_in_container.sh" \
  python -m gsae.cli smoke \
    --work-dir "${WORK_DIR}" \
    --device cpu \
    --steps 20 \
    --hidden-dim 32 \
    --latent-dim 64
