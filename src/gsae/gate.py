from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from scipy import sparse

from .data import load_layer_activations
from .train import checkpoint_path, load_gsae_checkpoint
from .utils import choose_device, ensure_dir


DEFAULT_GATE: dict[str, Any] = {
    "classifier": "calibrated_random_forest",
    "n_estimators": 300,
    "input_t_low": 0.30,
    "input_t_high": 0.65,
    "cont_d_low": 0.70,
    "cont_d_high": 0.90,
    "n_up": 2,
    "n_down": 3,
}


def encode_activation_dir(
    trained_dir: str | Path,
    activation_dir: str | Path,
    layers: list[int],
    device: torch.device | str = "auto",
    require_training_provenance: bool = False,
) -> tuple[sparse.csr_matrix, np.ndarray]:
    device = choose_device(str(device))
    zs = []
    reference = None
    if not layers or len(set(layers)) != len(layers):
        raise ValueError("layers must be nonempty and unique")
    for layer in layers:
        model, checkpoint_metadata = load_gsae_checkpoint(checkpoint_path(trained_dir, layer), device=device)
        x, y, metadata = load_layer_activations(activation_dir, layer)
        if require_training_provenance:
            checkpoint_ids = checkpoint_metadata.get("train_prompt_ids")
            if checkpoint_ids is None:
                raise ValueError(f"Layer {layer} checkpoint is missing train_prompt_ids provenance")
            if metadata.get("ids") != checkpoint_ids:
                raise ValueError(f"Gate pooled prompt IDs must match layer {layer} checkpoint train_prompt_ids in order")
        reference = _check_gate_alignment(reference, y, metadata)
        batches = []
        with torch.no_grad():
            for start in range(0, len(x), 256):
                z = model.encode(x[start:start + 256].to(device)).cpu().numpy()
                batches.append(sparse.csr_matrix(z))
        zs.append(sparse.vstack(batches, format="csr"))
    return sparse.hstack(zs, format="csr"), reference[0].cpu().numpy()


def _check_gate_alignment(reference, labels, metadata):
    ids = metadata.get("ids")
    if ids is None:
        raise ValueError("Activation row IDs are required for cross-layer alignment")
    current = (labels, ids, metadata.get("source_prompt_ids"))
    if reference is not None:
        if not torch.equal(reference[0], labels) or reference[1:] != current[1:]:
            raise ValueError("Activation row IDs, source prompt IDs and labels must align across layers")
    return current


def train_gate(
    trained_dir: str | Path,
    activation_dir: str | Path,
    prefix_activation_dir: str | Path,
    output_path: str | Path,
    layers: list[int],
    cfg: dict[str, Any],
    device: torch.device | str = "auto",
) -> dict[str, Any]:
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    if Path(activation_dir).resolve() == Path(prefix_activation_dir).resolve():
        raise ValueError("Prefix activations must be separate from pooled prompt activations")
    thresholds = _gate_thresholds(cfg)
    classifier = cfg.get("classifier", "calibrated_random_forest")
    if classifier not in {"calibrated_random_forest", "logistic_regression"}:
        raise ValueError(f"Unsupported gate classifier {classifier!r}")
    x_prompt, y_prompt = encode_activation_dir(
        trained_dir, activation_dir, layers, device=device, require_training_provenance=True,
    )
    x_prefix, y_prefix = encode_activation_dir(trained_dir, prefix_activation_dir, layers, device=device)
    _, _, prompt_metadata = load_layer_activations(activation_dir, layers[0])
    _, _, prefix_metadata = load_layer_activations(prefix_activation_dir, layers[0])
    _check_prefix_sources(prompt_metadata, prefix_metadata, y_prompt, y_prefix)
    x = sparse.vstack([x_prompt, x_prefix], format="csr")
    y = np.concatenate([y_prompt, y_prefix])
    class_counts = np.bincount(y.astype(int), minlength=2)
    if classifier == "calibrated_random_forest":
        if min(class_counts) < 3:
            raise ValueError("Calibrated random forest requires at least 3 examples per class; no fallback is used")
        n_estimators, n_jobs = int(cfg.get("n_estimators", 300)), int(cfg.get("n_jobs", 1))
        if n_estimators < 1 or n_jobs < 1:
            raise ValueError("n_estimators and n_jobs must be positive")
        base = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=int(cfg.get("seed", 7)),
            n_jobs=n_jobs,
            class_weight="balanced_subsample",
        )
        clf = CalibratedClassifierCV(base, method="sigmoid", cv=3, n_jobs=1)
    else:
        clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=int(cfg.get("seed", 7)))
    clf.fit(x, y)
    payload = {
        "classifier": clf,
        "layers": [int(l) for l in layers],
        "thresholds": thresholds,
        "config": dict(cfg),
        "num_examples": int(x.shape[0]),
        "num_prompt_examples": int(x_prompt.shape[0]),
        "num_prefix_examples": int(x_prefix.shape[0]),
        "latent_dim": int(x.shape[1]),
    }
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    joblib.dump(payload, output_path)
    return payload


def _check_prefix_sources(prompt_metadata, prefix_metadata, y_prompt, y_prefix):
    sources = prefix_metadata.get("source_prompt_ids")
    if sources is None or len(sources) != len(y_prefix):
        raise ValueError("Prefix shards require one source_prompt_ids entry per row")
    labels_by_id = dict(zip(prompt_metadata["ids"], y_prompt.tolist()))
    counts = {}
    for source, label in zip(sources, y_prefix):
        if source not in labels_by_id or labels_by_id[source] != label:
            raise ValueError("Prefix source prompt IDs and labels must match pooled training prompts")
        counts[source] = counts.get(source, 0) + 1
    if max(counts.values()) > 16:
        raise ValueError("Gate training accepts up to 16 prefix states per source prompt")
    if set(y_prompt.tolist()) != {0, 1} or set(y_prefix.tolist()) != {0, 1}:
        raise ValueError("Gate training requires harmful and benign examples in prompt and prefix states")


def _gate_thresholds(cfg):
    thresholds = {key: float(cfg.get(key, DEFAULT_GATE[key])) for key in
                  ("input_t_low", "input_t_high", "cont_d_low", "cont_d_high")}
    for low, high in [("input_t_low", "input_t_high"), ("cont_d_low", "cont_d_high")]:
        if not 0 <= thresholds[low] < thresholds[high] <= 1:
            raise ValueError(f"Gate thresholds must satisfy 0 <= {low} < {high} <= 1")
    for key in ("n_up", "n_down"):
        value = cfg.get(key, DEFAULT_GATE[key])
        if int(value) != value or int(value) < 1:
            raise ValueError(f"{key} must be a positive integer")
        thresholds[key] = int(value)
    return thresholds


def load_gate(path: str | Path) -> dict[str, Any]:
    return joblib.load(path)
