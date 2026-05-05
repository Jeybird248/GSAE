from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch

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
) -> tuple[np.ndarray, np.ndarray]:
    device = choose_device(str(device))
    zs = []
    labels_ref = None
    for layer in layers:
        model, _ = load_gsae_checkpoint(checkpoint_path(trained_dir, layer), device=device)
        x, y, _ = load_layer_activations(activation_dir, layer)
        with torch.no_grad():
            z = model.encode(x.to(device)).cpu().numpy()
        zs.append(z)
        labels = y.cpu().numpy()
        if labels_ref is None:
            labels_ref = labels
        elif not np.array_equal(labels_ref, labels):
            raise ValueError("Activation shards must have labels in the same order across layers")
    if labels_ref is None:
        raise ValueError("No layers were provided")
    return np.concatenate(zs, axis=1), labels_ref


def train_gate(
    trained_dir: str | Path,
    activation_dir: str | Path,
    output_path: str | Path,
    layers: list[int],
    cfg: dict[str, Any],
    device: torch.device | str = "auto",
) -> dict[str, Any]:
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    x, y = encode_activation_dir(trained_dir, activation_dir, layers, device=device)
    class_counts = np.bincount(y.astype(int), minlength=2)
    if min(class_counts) >= 3 and cfg.get("classifier", "calibrated_random_forest") == "calibrated_random_forest":
        base = RandomForestClassifier(
            n_estimators=int(cfg.get("n_estimators", 300)),
            random_state=int(cfg.get("seed", 7)),
            n_jobs=-1,
            class_weight="balanced_subsample",
        )
        clf = CalibratedClassifierCV(base, method="sigmoid", cv=3)
    else:
        clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=int(cfg.get("seed", 7)))
    clf.fit(x, y)
    payload = {
        "classifier": clf,
        "layers": [int(l) for l in layers],
        "thresholds": {
            "input_t_low": float(cfg.get("input_t_low", 0.30)),
            "input_t_high": float(cfg.get("input_t_high", 0.65)),
            "cont_d_low": float(cfg.get("cont_d_low", 0.70)),
            "cont_d_high": float(cfg.get("cont_d_high", 0.90)),
            "n_up": int(cfg.get("n_up", 2)),
            "n_down": int(cfg.get("n_down", 3)),
        },
        "config": dict(cfg),
        "num_examples": int(x.shape[0]),
        "latent_dim": int(x.shape[1]),
    }
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    joblib.dump(payload, output_path)
    return payload


def load_gate(path: str | Path) -> dict[str, Any]:
    return joblib.load(path)
