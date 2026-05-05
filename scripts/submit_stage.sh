#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGE=""
BID=100
DRY_RUN=0
SUBFILE="${ROOT}/condor/run_stage.sub"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage) shift; STAGE="${1:?missing stage}" ;;
    --bid) shift; BID="${1:?missing bid}" ;;
    --subfile) shift; SUBFILE="${1:?missing subfile}" ;;
    --dry-run) DRY_RUN=1 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done

if [ -z "${STAGE}" ]; then
  echo "--stage is required" >&2
  exit 2
fi

submit=(condor_submit_bid "${BID}" -a "stage=${STAGE}" "${SUBFILE}")
if [ "${DRY_RUN}" -eq 1 ]; then
  printf '%q ' "${submit[@]}"
  printf '\n'
else
  "${submit[@]}"
fi
