#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NAME="${1:-gsae}"
DEF="${ROOT}/containers/${NAME}.def"
OUT_DIR="${GSAE_CONTAINERS_DIR:-/fast/jyeon/gsae/containers}"
OUT="${OUT_DIR}/${NAME}.sif"

mkdir -p "${OUT_DIR}" "${APPTAINER_CACHEDIR:-/fast/jyeon/apptainer_cache}" "${APPTAINER_TMPDIR:-/home/jyeon/apptainer_tmp}"
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-/fast/jyeon/apptainer_cache}"
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-/home/jyeon/apptainer_tmp}"

if [ ! -f "${DEF}" ]; then
  echo "Missing definition file: ${DEF}" >&2
  exit 2
fi

apptainer build --force "${OUT}" "${DEF}"
echo "${OUT}"
