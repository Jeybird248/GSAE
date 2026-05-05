from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .data import load_layer_activations
from .graph import load_layer_laplacian
from .train import checkpoint_path, load_gsae_checkpoint
from .utils import atomic_torch_save, choose_device, ensure_dir, minmax, read_jsonl


DEFAULT_BANK: dict[str, Any] = {
    "eta": 1.0,
    "rho": 0.95,
    "candidate_pool_size": 2000,
    "a": 1.0,
    "b": 1.0,
    "c": 1.0,
    "efficacy_mode": "activation_proxy",
    "efficacy_file": None,
    "eps_srr": 0.05,
}


def _column_energies(weight: torch.Tensor, laplacian: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    laplacian = laplacian.to(weight.device)
    lw = torch.sparse.mm(laplacian, weight) if laplacian.is_sparse else laplacian @ weight
    numerator = (weight * lw).sum(dim=0)
    denominator = weight.square().sum(dim=0).clamp_min(eps)
    return numerator / denominator


def _probe_relevance(model) -> torch.Tensor:
    # Binary logit margin is invariant to adding a constant offset to both class logits.
    return (model.probe.weight[1] - model.probe.weight[0]).abs().detach()


def _activation_proxy_efficacy(
    directions: torch.Tensor,
    activations: torch.Tensor,
    labels: torch.Tensor,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Approximate direction efficacy from class-conditional projections.

    Full paper-faithful efficacy requires generation and a response judge. This
    proxy keeps the repo runnable on activation shards by preferring directions
    whose projections separate harmful from benign validation activations.
    """

    dirs = F.normalize(directions.T, dim=1, eps=eps)  # [m, d]
    proj = activations.float() @ dirs.T  # [n, m]
    harmful = labels == 1
    benign = labels == 0
    if harmful.sum() == 0 or benign.sum() == 0:
        raise ValueError("Bank efficacy needs both harmful and benign validation labels")
    h_mean = proj[harmful].mean(dim=0)
    b_mean = proj[benign].mean(dim=0)
    pooled_std = proj.std(dim=0).clamp_min(eps)
    separation = (h_mean - b_mean) / pooled_std
    deltas = torch.where(separation >= 0, torch.ones_like(separation), -torch.ones_like(separation))
    efficacy = separation.abs()
    return deltas, efficacy


def build_bank(
    trained_dir: str | Path,
    val_activation_dir: str | Path,
    output_path: str | Path,
    layers: list[int],
    cfg: dict[str, Any],
    device: torch.device | str = "auto",
) -> dict[str, Any]:
    device = choose_device(str(device))
    candidates: list[dict[str, Any]] = []
    external_efficacy = None
    if cfg.get("efficacy_file"):
        external_efficacy = {
            (int(row["layer"]), int(row["feature"])): (float(row.get("delta", 1.0)), float(row["efficacy"]))
            for row in read_jsonl(cfg["efficacy_file"])
        }

    for layer in layers:
        model, metadata = load_gsae_checkpoint(checkpoint_path(trained_dir, layer), device=device)
        laplacian = load_layer_laplacian(Path(trained_dir) / f"layer_{int(layer)}", layer).to(device)
        weight = model.decoder.weight.detach()
        energy = _column_energies(weight, laplacian)
        coherence = torch.exp(-float(cfg.get("eta", 1.0)) * energy)
        relevance = _probe_relevance(model).to(device)
        coh_n = minmax(coherence)
        rel_n = minmax(relevance)
        pool_score = coh_n * rel_n
        k_pool = min(int(cfg.get("candidate_pool_size", 2000)), int(pool_score.numel()))
        candidate_idx = torch.topk(pool_score, k=k_pool).indices.sort().values

        val_x, val_y, _ = load_layer_activations(val_activation_dir, layer)
        val_x = val_x.to(device)
        val_y = val_y.to(device)
        directions = weight[:, candidate_idx]
        if external_efficacy is not None:
            proxy_deltas, proxy_efficacy = _activation_proxy_efficacy(directions, val_x, val_y)
            delta_values = []
            efficacy_values = []
            for local_pos, feature_idx in enumerate(candidate_idx.detach().cpu().tolist()):
                maybe = external_efficacy.get((int(layer), int(feature_idx)))
                if maybe is None:
                    delta_values.append(float(proxy_deltas[local_pos].detach().cpu()))
                    efficacy_values.append(float(proxy_efficacy[local_pos].detach().cpu()))
                else:
                    delta_values.append(maybe[0])
                    efficacy_values.append(maybe[1])
            deltas = torch.tensor(delta_values, device=device, dtype=torch.float32)
            efficacy = torch.tensor(efficacy_values, device=device, dtype=torch.float32)
        else:
            deltas, efficacy = _activation_proxy_efficacy(directions, val_x, val_y)
        eff_n = minmax(efficacy.clamp_min(0))
        coh_pool = minmax(coherence[candidate_idx])
        rel_pool = minmax(relevance[candidate_idx])
        a = float(cfg.get("a", 1.0))
        b = float(cfg.get("b", 1.0))
        c = float(cfg.get("c", 1.0))
        utility = (
            coh_pool.clamp_min(1e-12).pow(a)
            * rel_pool.clamp_min(1e-12).pow(b)
            * eff_n.clamp_min(1e-12).pow(c)
        ).pow(1.0 / (a + b + c))

        for local_pos, feature_idx in enumerate(candidate_idx.detach().cpu().tolist()):
            candidates.append(
                {
                    "layer": int(layer),
                    "feature": int(feature_idx),
                    "direction": directions[:, local_pos].detach().cpu().half(),
                    "delta": float(deltas[local_pos].detach().cpu()),
                    "score": float(utility[local_pos].detach().cpu()),
                    "coherence": float(coh_pool[local_pos].detach().cpu()),
                    "relevance": float(rel_pool[local_pos].detach().cpu()),
                    "efficacy": float(eff_n[local_pos].detach().cpu()),
                    "metadata": metadata,
                }
            )

    candidates.sort(key=lambda row: row["score"], reverse=True)
    total = sum(max(0.0, row["score"]) for row in candidates)
    if total <= 0:
        selected = candidates[:1]
    else:
        threshold = float(cfg.get("rho", 0.95)) * total
        running = 0.0
        selected = []
        for row in candidates:
            selected.append(row)
            running += max(0.0, row["score"])
            if running >= threshold:
                break

    denom = sum(max(0.0, row["score"]) for row in selected) or 1.0
    for row in selected:
        row["weight"] = max(0.0, row["score"]) / denom

    by_layer: dict[int, dict[str, Any]] = {}
    for row in selected:
        layer = int(row["layer"])
        entry = by_layer.setdefault(layer, {"features": [], "directions": [], "deltas": [], "weights": [], "scores": []})
        entry["features"].append(int(row["feature"]))
        entry["directions"].append(row["direction"])
        entry["deltas"].append(float(row["delta"]))
        entry["weights"].append(float(row["weight"]))
        entry["scores"].append(float(row["score"]))

    packed_layers = {}
    for layer, entry in by_layer.items():
        packed_layers[layer] = {
            "features": torch.tensor(entry["features"], dtype=torch.long),
            "directions": torch.stack(entry["directions"]).float(),  # [m, d]
            "deltas": torch.tensor(entry["deltas"], dtype=torch.float32),
            "weights": torch.tensor(entry["weights"], dtype=torch.float32),
            "scores": torch.tensor(entry["scores"], dtype=torch.float32),
        }

    payload = {
        "layers": packed_layers,
        "selected_count": len(selected),
        "candidate_count": len(candidates),
        "config": dict(cfg),
    }
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    atomic_torch_save(payload, output_path)
    return payload


def load_bank(path: str | Path, device: torch.device | str = "cpu") -> dict[str, Any]:
    payload = torch.load(path, map_location=device)
    payload["layers"] = {int(k): v for k, v in payload["layers"].items()}
    return payload
