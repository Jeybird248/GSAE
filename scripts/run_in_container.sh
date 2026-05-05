#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONTAINER="${GSAE_CONTAINER:-/fast/jyeon/gsae/containers/gsae.sif}"

if [ ! -f "${CONTAINER}" ]; then
  echo "Container not found: ${CONTAINER}" >&2
  echo "Build it with: bash ${ROOT}/scripts/build_container.sh gsae" >&2
  exit 2
fi

USE_NV="${USE_NV:-1}"
NV_ARGS=()
if [ "${USE_NV}" = "1" ]; then
  NV_ARGS=(--nv)
fi

exec apptainer exec "${NV_ARGS[@]}" \
  --bind /home:/home \
  --bind /lustre:/lustre \
  --bind /fast:/fast \
  --env PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}" \
  --pwd "${ROOT}" \
  "${CONTAINER}" "$@"
