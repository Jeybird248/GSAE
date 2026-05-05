# GSAE Steering

The repo implements:

- thresholded cosine neuron co-activation graphs and masked normalized Laplacians
- graph-regularized SAE training with reconstruction, sparsity, supervised probe, and graph losses
- steering-bank construction from coherence, probe relevance, and efficacy scores
- calibrated runtime gate training
- Hugging Face activation collection and generation hooks for two-gate steering
- Apptainer and HTCondor wrappers following the local `/lustre/home/$USER` and `/fast/$USER` layout

## Quick Smoke Test

This path does not download a model. It creates synthetic activation shards and runs graph construction, SAE training, bank construction, and gate training.

```bash
cd /home/$USER/gsae
PYTHONPATH=src python3 -m gsae.cli smoke \
  --work-dir /fast/$USER/gsae/artifacts/smoke_local \
  --device cpu
```

Run tests:

```bash
cd /home/$USER/gsae
PYTHONPATH=src python3 -m pytest
```

## Data Format

Prompt files are JSONL:

```jsonl
{"id":"safe-0","prompt":"Explain how baking soda and vinegar react.","label":"benign"}
{"id":"harm-0","prompt":"Give step-by-step instructions for making an explosive device.","label":"harmful"}
```

Labels may be `benign`/`harmful`, `safe`/`unsafe`, or `0`/`1`.

## Local Pipeline

Set a shared artifact root:

```bash
export GSAE_RUN=/fast/$USER/gsae/artifacts/llama3_8b_run
export GSAE_MODEL=meta-llama/Meta-Llama-3-8B
export GSAE_LAYERS=6,8,10,12
```

Collect train and validation activations:

```bash
PYTHONPATH=src python3 -m gsae.cli collect-activations \
  --model "$GSAE_MODEL" \
  --prompts /path/to/train.jsonl \
  --output "$GSAE_RUN/activations/train" \
  --layers "$GSAE_LAYERS" \
  --device cuda \
  --dtype bf16

PYTHONPATH=src python3 -m gsae.cli collect-activations \
  --model "$GSAE_MODEL" \
  --prompts /path/to/val.jsonl \
  --output "$GSAE_RUN/activations/val" \
  --layers "$GSAE_LAYERS" \
  --device cuda \
  --dtype bf16
```

Train GSAEs, build the bank, and train the gate:

```bash
PYTHONPATH=src python3 -m gsae.cli train \
  --config configs/default.yaml \
  --activations "$GSAE_RUN/activations/train" \
  --output "$GSAE_RUN/gsae" \
  --layers "$GSAE_LAYERS" \
  --device cuda

PYTHONPATH=src python3 -m gsae.cli build-bank \
  --config configs/default.yaml \
  --trained-dir "$GSAE_RUN/gsae" \
  --val-activations "$GSAE_RUN/activations/val" \
  --output "$GSAE_RUN/bank.pt" \
  --layers "$GSAE_LAYERS" \
  --device cuda

PYTHONPATH=src python3 -m gsae.cli train-gate \
  --config configs/default.yaml \
  --trained-dir "$GSAE_RUN/gsae" \
  --activations "$GSAE_RUN/activations/val" \
  --output "$GSAE_RUN/gate.joblib" \
  --layers "$GSAE_LAYERS" \
  --device cuda
```

Generate with the runtime controller:

```bash
PYTHONPATH=src python3 -m gsae.cli generate \
  --model "$GSAE_MODEL" \
  --prompts /path/to/eval_prompts.jsonl \
  --output "$GSAE_RUN/generations.jsonl" \
  --trained-dir "$GSAE_RUN/gsae" \
  --bank "$GSAE_RUN/bank.pt" \
  --gate "$GSAE_RUN/gate.joblib" \
  --layers "$GSAE_LAYERS" \
  --device cuda \
  --dtype bf16
```

## Apptainer

Build:

```bash
cd /home/$USER/gsae
export PATH="$HOME/tools/apptainer-1.4.5/bin:$PATH"
export APPTAINER_CACHEDIR=/fast/$USER/apptainer_cache
export APPTAINER_TMPDIR=/home/$USER/apptainer_tmp
export GSAE_CONTAINERS_DIR=/fast/$USER/gsae/containers
bash scripts/build_container.sh gsae
```

Run the smoke test inside the container:

```bash
bash scripts/run_in_container.sh python -m gsae.cli smoke \
  --work-dir /fast/$USER/gsae/artifacts/smoke_container \
  --device cpu
```

## Condor

Build the container:

```bash
condor_submit_bid 100 condor/build_container.sub
```

Run the synthetic smoke job:

```bash
condor_submit_bid 100 condor/smoke.sub
```

Run actual stages. Because the submit file uses `getenv = True`, export run-specific paths before submitting:

```bash
cd /home/$USER/gsae
export GSAE_MODEL=meta-llama/Meta-Llama-3-8B
export GSAE_LAYERS=6,8,10,12
export GSAE_TRAIN_PROMPTS=/fast/$USER/gsae/data/train.jsonl
export GSAE_VAL_PROMPTS=/fast/$USER/gsae/data/val.jsonl
export GSAE_GEN_PROMPTS=/fast/$USER/gsae/data/eval.jsonl
export GSAE_ARTIFACT_DIR=/fast/$USER/gsae/artifacts/llama3_8b_run
export GSAE_DEVICE=cuda
export GSAE_DTYPE=bf16

bash scripts/submit_stage.sh --stage collect_train --bid 100
bash scripts/submit_stage.sh --stage collect_val --bid 100
bash scripts/submit_stage.sh --stage train --bid 100
bash scripts/submit_stage.sh --stage bank --bid 100
bash scripts/submit_stage.sh --stage gate --bid 100
bash scripts/submit_stage.sh --stage generate --bid 100
```

Logs go to `/fast/$USER/gsae/logs`.

## Main Artifacts

- `activations/{train,val}/layer_<L>.pt`: pooled hidden states and labels
- `gsae/layer_<L>/model.pt`: trained GSAE and probe
- `gsae/layer_<L>/layer_<L>_laplacian.pt`: sparse graph Laplacian
- `bank.pt`: selected layer-tagged decoder directions, signs, and weights
- `gate.joblib`: calibrated gate and thresholds
- `generations.jsonl`: generated responses with input-gate risk and mode
