#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGE="${1:?stage required: collect|train|bank|gate|generate|smoke}"
CONFIG="${2:-${ROOT}/configs/default.yaml}"

LAYERS="${GSAE_LAYERS:-6,8,10,12}"
ARTIFACT_DIR="${GSAE_ARTIFACT_DIR:-/fast/jyeon/gsae/artifacts/default}"
MODEL="${GSAE_MODEL:-meta-llama/Meta-Llama-3-8B}"
TRAIN_PROMPTS="${GSAE_TRAIN_PROMPTS:-${ROOT}/data/examples/prompts.jsonl}"
VAL_PROMPTS="${GSAE_VAL_PROMPTS:-${ROOT}/data/examples/prompts.jsonl}"
GEN_PROMPTS="${GSAE_GEN_PROMPTS:-${VAL_PROMPTS}}"
DEVICE="${GSAE_DEVICE:-auto}"

case "${STAGE}" in
  smoke)
    exec "${ROOT}/scripts/run_in_container.sh" python -m gsae.cli smoke --work-dir "${ARTIFACT_DIR}/smoke" --device cpu
    ;;
  collect_train)
    exec "${ROOT}/scripts/run_in_container.sh" python -m gsae.cli collect-activations \
      --model "${MODEL}" --prompts "${TRAIN_PROMPTS}" --output "${ARTIFACT_DIR}/activations/train" \
      --layers "${LAYERS}" --device "${DEVICE}" --dtype "${GSAE_DTYPE:-auto}" ${GSAE_TRUST_REMOTE_CODE:+--trust-remote-code}
    ;;
  collect_val)
    exec "${ROOT}/scripts/run_in_container.sh" python -m gsae.cli collect-activations \
      --model "${MODEL}" --prompts "${VAL_PROMPTS}" --output "${ARTIFACT_DIR}/activations/val" \
      --layers "${LAYERS}" --device "${DEVICE}" --dtype "${GSAE_DTYPE:-auto}" ${GSAE_TRUST_REMOTE_CODE:+--trust-remote-code}
    ;;
  train)
    exec "${ROOT}/scripts/run_in_container.sh" python -m gsae.cli train \
      --config "${CONFIG}" --activations "${ARTIFACT_DIR}/activations/train" \
      --output "${ARTIFACT_DIR}/gsae" --layers "${LAYERS}" --device "${DEVICE}"
    ;;
  bank)
    exec "${ROOT}/scripts/run_in_container.sh" python -m gsae.cli build-bank \
      --config "${CONFIG}" --trained-dir "${ARTIFACT_DIR}/gsae" \
      --val-activations "${ARTIFACT_DIR}/activations/val" \
      --output "${ARTIFACT_DIR}/bank.pt" --layers "${LAYERS}" --device "${DEVICE}"
    ;;
  gate)
    exec "${ROOT}/scripts/run_in_container.sh" python -m gsae.cli train-gate \
      --config "${CONFIG}" --trained-dir "${ARTIFACT_DIR}/gsae" \
      --activations "${ARTIFACT_DIR}/activations/val" \
      --output "${ARTIFACT_DIR}/gate.joblib" --layers "${LAYERS}" --device "${DEVICE}"
    ;;
  generate)
    exec "${ROOT}/scripts/run_in_container.sh" python -m gsae.cli generate \
      --model "${MODEL}" --prompts "${GEN_PROMPTS}" --output "${ARTIFACT_DIR}/generations.jsonl" \
      --trained-dir "${ARTIFACT_DIR}/gsae" --bank "${ARTIFACT_DIR}/bank.pt" \
      --gate "${ARTIFACT_DIR}/gate.joblib" --layers "${LAYERS}" --device "${DEVICE}" \
      --dtype "${GSAE_DTYPE:-auto}" ${GSAE_TRUST_REMOTE_CODE:+--trust-remote-code}
    ;;
  *)
    echo "Unknown stage: ${STAGE}" >&2
    exit 2
    ;;
esac
