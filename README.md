# GSAE

Graph-Regularized Sparse Autoencoders learn decoder directions for safety steering. The package provides activation collection, autoencoder training, response-scored bank construction, and a two-gate generation controller.

## Install

```bash
python -m pip install -e .
```

Use Python 3.10 or later. To use the default OpenAI response judge, install `python -m pip install -e '.[judge]'` and set `OPENAI_API_KEY`.

## Data

Prepare disjoint `graph.jsonl`, `train.jsonl`, and `validation.jsonl` files. Each row needs a unique prompt ID and an explicit safety label:

```jsonl
{"id":"safe-0","prompt":"Explain how rain forms.","label":"benign"}
{"id":"harm-0","prompt":"A harmful request for safety training.","label":"harmful"}
```

Use different IDs across splits. Training and validation must each contain harmful and benign examples. Layers use zero-based decoder-block indices. Prefix collection uses an optional `continuation` field or generates a continuation. Prefix states inherit the source prompt's safety label.

## Train

Set the model directory, output directory, and target layers:

```bash
MODEL=models/base
RUN=artifacts/run
LAYERS=6,8,10,12

for SPLIT in graph train validation; do
  gsae collect-activations --model "$MODEL" --prompts "data/$SPLIT.jsonl" \
    --output "$RUN/activations/$SPLIT" --layers "$LAYERS" --device cuda --dtype bf16
done

gsae train --config configs/default.yaml \
  --activations "$RUN/activations/train" --graph-activations "$RUN/activations/graph" \
  --output "$RUN/gsae" --layers "$LAYERS" --device cuda
```

Set `--steps` to the training duration you need. Defaults are in `configs/default.yaml`.

## Build the steering bank

Screen candidates, generate baseline and both-sign interventions, then judge the responses:

```bash
gsae screen-bank --config configs/default.yaml --trained-dir "$RUN/gsae" \
  --output "$RUN/candidates.pt" --layers "$LAYERS" --device cuda

gsae collect-bank-responses --model "$MODEL" --candidates "$RUN/candidates.pt" \
  --prompts data/validation.jsonl --output "$RUN/responses.jsonl" --device cuda --dtype bf16

gsae judge-bank-responses --responses "$RUN/responses.jsonl" --output "$RUN/judgments.jsonl"

gsae build-bank --config configs/default.yaml --trained-dir "$RUN/gsae" \
  --val-activations "$RUN/activations/validation" --efficacy-file "$RUN/judgments.jsonl" \
  --output "$RUN/bank.pt" --layers "$LAYERS" --device cuda
```

The judge command makes OpenAI API calls. A custom judge can use `gsae.evaluate.judge_bank_responses(input_path, output_path, judge)`, where `judge(prompt, response)` returns one of `HARMFUL_COMPLIANCE`, `REFUSAL`, `BENIGN_COMPLIANCE`, or `OTHER`.

Keep checkpoints and bank settings unchanged between screening and assembly. Bank assembly requires judgments for every validation prompt, baseline, candidate, and sign.

## Train the gate and generate

```bash
gsae collect-prefixes --model "$MODEL" --prompts data/train.jsonl \
  --output "$RUN/activations/prefixes" --layers "$LAYERS" --device cuda --dtype bf16

gsae train-gate --config configs/default.yaml --trained-dir "$RUN/gsae" \
  --activations "$RUN/activations/train" --prefix-activations "$RUN/activations/prefixes" \
  --output "$RUN/gate.joblib" --layers "$LAYERS" --device cpu

gsae generate --model "$MODEL" --prompts data/new_prompts.jsonl --output "$RUN/generations.jsonl" \
  --trained-dir "$RUN/gsae" --bank "$RUN/bank.pt" --gate "$RUN/gate.joblib" \
  --layers "$LAYERS" --device cuda --dtype bf16
```

The gate uses pooled prompt states and up to 16 prefix states per training prompt. Generation returns continuation text. Monitored generation uses two forward passes per token with caching disabled. It supports one greedy or sampled sequence.

## Container

```bash
apptainer build gsae.sif containers/gsae.def
apptainer exec --nv --bind "$PWD:/workspace" --pwd /workspace \
  --env PYTHONPATH=/workspace/src gsae.sif python -m gsae.cli --help
```

Replace `--help` with a framework command and its arguments.
Add `openai` to the container recipe to use the default response judge.
