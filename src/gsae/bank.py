from __future__ import annotations

import math
import json
from pathlib import Path
from typing import Any

import torch

from .data import load_layer_activations
from .evaluate import LABELS
from .graph import load_layer_laplacian
from .train import checkpoint_path, load_gsae_checkpoint
from .utils import atomic_torch_save, choose_device, minmax


DEFAULT_BANK: dict[str, Any] = {
    "eta": 1.0, "rho": 0.95, "candidate_pool_size": 2000,
    "a": 1.0, "b": 1.0, "c": 1.0, "efficacy_file": None, "eps_srr": 0.05,
}


def _column_energies(weight, laplacian, eps=1e-8):
    laplacian = laplacian.to(weight.device)
    lw = torch.sparse.mm(laplacian, weight) if laplacian.is_sparse else laplacian @ weight
    return (weight * lw).sum(dim=0) / weight.square().sum(dim=0).clamp_min(eps)


def _probe_relevance(model):
    # The binary softmax margin is the equivalent single-logit probe coefficient.
    return (model.probe.weight[1] - model.probe.weight[0]).abs().detach()


def _validate_config(cfg):
    values = [float(cfg.get(key, DEFAULT_BANK[key])) for key in ("eta", "a", "b", "c")]
    if any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("eta and score exponents must be finite and positive")
    if not 0 < float(cfg.get("rho", 0.95)) <= 1:
        raise ValueError("rho must be in (0, 1]")
    if int(cfg.get("candidate_pool_size", 2000)) < 1:
        raise ValueError("candidate_pool_size must be positive")
    if not 0 <= float(cfg.get("eps_srr", 0.05)) <= 1:
        raise ValueError("eps_srr must be in [0, 1]")
    if cfg.get("efficacy_mode", "response_judge") != "response_judge":
        raise ValueError("Bank construction requires response judgments; activation_proxy is unsupported")


def _screen_layer(trained_dir, layer, cfg, device):
    model, metadata = load_gsae_checkpoint(checkpoint_path(trained_dir, layer), device=device)
    laplacian = load_layer_laplacian(Path(trained_dir) / f"layer_{layer}", layer).to(device)
    weight = model.decoder.weight.detach()
    coherence = torch.exp(-float(cfg.get("eta", 1.0)) * _column_energies(weight, laplacian))
    relevance = _probe_relevance(model).to(device)
    pool_score = minmax(coherence) * minmax(relevance)
    count = min(int(cfg.get("candidate_pool_size", 2000)), pool_score.numel())
    indices = torch.argsort(pool_score, descending=True, stable=True)[:count].cpu().tolist()
    return [
        {"layer": layer, "feature": index, "direction": weight[:, index].cpu().float().clone(),
         "coherence": float(coherence[index]), "relevance": float(relevance[index]), "metadata": metadata}
        for index in indices
    ]


def screen_bank(trained_dir, output_path, layers, cfg, device="auto"):
    """Save per-layer candidate pools for response evaluation."""
    _validate_config(cfg)
    if not layers or len(set(layers)) != len(layers):
        raise ValueError("Provide distinct target layers")
    device = choose_device(str(device))
    candidates = []
    for layer in layers:
        candidates.extend(_screen_layer(trained_dir, int(layer), cfg, device))
    payload = {"candidates": candidates, "config": dict(cfg)}
    atomic_torch_save(payload, output_path)
    return payload


def _judgment_key(row):
    layer, feature, delta = row["layer"], row["feature"], row["delta"]
    if layer is None:
        if feature is not None or delta != 0:
            raise ValueError("Baseline judgments need null layer/feature and delta=0")
    elif feature is None or delta not in (-1, 1):
        raise ValueError("Intervention judgments need layer, feature and delta in {-1, 1}")
    return layer, feature, delta, str(row["id"])


def _read_judgments(path):
    outcomes, prompts, positions = {}, {}, {}
    with Path(path).open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            _add_judgment(json.loads(line), outcomes, prompts, positions)
    if {signature[1] for signature in prompts.values()} != {False, True}:
        raise ValueError("Bank validation requires harmful and benign prompts")
    return outcomes, prompts


def _add_judgment(row, outcomes, prompts, positions):
    prompt_id = str(row["id"])
    if not isinstance(row["is_harmful"], bool):
        raise ValueError("Judgment is_harmful must be a boolean")
    signature = (row["prompt"], row["is_harmful"])
    if prompt_id in prompts and prompts[prompt_id] != signature:
        raise ValueError(f"Inconsistent prompt or safety label for {prompt_id}")
    if prompt_id not in prompts:
        positions[prompt_id] = len(prompts)
        prompts[prompt_id] = signature
    key = _judgment_key(row)
    if row["label"] not in LABELS:
        raise ValueError(f"Unknown response judgment: {row['label']}")
    # Store one byte per response instead of retaining generated text and tuple keys.
    codes = outcomes.setdefault(key[:3], bytearray())
    index = positions[prompt_id]
    if len(codes) <= index:
        codes.extend(b"\xff" * (index + 1 - len(codes)))
    if codes[index] != 255:
        raise ValueError(f"Duplicate response judgment: {key}")
    codes[index] = LABELS.index(row["label"])


def _outcome_rates(outcomes, prompts, layer=None, feature=None, delta=0):
    codes = outcomes.get((layer, feature, delta), bytearray())
    if len(codes) != len(prompts) or 255 in codes:
        raise ValueError(f"Missing response judgment for layer={layer}, feature={feature}, delta={delta}")
    harmful, benign = [], []
    for code, (_, is_harmful) in zip(codes, prompts.values()):
        (harmful if is_harmful else benign).append(code)
    return (sum(code == LABELS.index("HARMFUL_COMPLIANCE") for code in harmful) / len(harmful),
            sum(code == LABELS.index("REFUSAL") for code in benign) / len(benign))


def _orientation(candidate, outcomes, prompts, base_hcr, base_srr, tolerance):
    choices = []
    for delta in (1, -1):
        hcr, srr = _outcome_rates(outcomes, prompts, candidate["layer"], candidate["feature"], delta)
        reduction = base_hcr - hcr
        if srr - base_srr <= tolerance + 1e-12:
            choices.append((reduction, -srr, delta, reduction - srr))
    if not choices:
        return 1, 0.0  # Neither sign is eligible for a positive bank score.
    _, _, delta, efficacy = max(choices)
    return delta, efficacy


def _score_layer(pool, outcomes, prompts, cfg, baseline):
    a, b, c = (float(cfg.get(key, 1.0)) for key in ("a", "b", "c"))
    for row in pool:
        row["delta"], row["raw_efficacy"] = _orientation(
            row, outcomes, prompts, *baseline, float(cfg.get("eps_srr", 0.05)))
    coh = minmax(torch.tensor([row["coherence"] for row in pool], dtype=torch.float64))
    rel = minmax(torch.tensor([row["relevance"] for row in pool], dtype=torch.float64))
    eff = minmax(torch.tensor([max(0, row["raw_efficacy"]) for row in pool], dtype=torch.float64))
    scores = (coh.pow(a) * rel.pow(b) * eff.pow(c)).pow(1 / (a + b + c))
    for row, score in zip(pool, scores.tolist()):
        row["score"] = score if row["direction"].norm() > 0 else 0.0


def _pack_bank(selected):
    denominator = math.fsum(row["score"] for row in selected)
    by_layer = {}
    for layer in dict.fromkeys(row["layer"] for row in selected):
        rows = [row for row in selected if row["layer"] == layer]
        by_layer[layer] = {
            "features": torch.tensor([row["feature"] for row in rows]),
            "directions": torch.stack([row["direction"] for row in rows]).float(),
            "deltas": torch.tensor([row["delta"] for row in rows], dtype=torch.float32),
            "weights": torch.tensor([row["score"] / denominator for row in rows], dtype=torch.float32),
            "scores": torch.tensor([row["score"] for row in rows], dtype=torch.float32),
        }
    return by_layer


def _select_bank(candidates, rho):
    candidates.sort(key=lambda row: row["score"], reverse=True)
    total = math.fsum(row["score"] for row in candidates)
    if total <= 0 or not math.isfinite(total):
        raise ValueError("No positive bank score; inspect response efficacy and constant score pools")
    selected, cumulative = [], 0.0
    for row in candidates:
        if row["score"] <= 0:
            break
        selected.append(row)
        cumulative += row["score"]
        if cumulative >= rho * total:
            break
    return _pack_bank(selected), len(selected)


def _validate_bank_split(val_activation_dir, layer, pool, prompts):
    _, labels, metadata = load_layer_activations(val_activation_dir, layer)
    ids = metadata.get("ids")
    if ids is None or set(ids) != set(prompts):
        raise ValueError("Judgment prompt IDs must match explicit bank-validation activation IDs")
    if any(bool(label) != prompts[prompt_id][1] for prompt_id, label in zip(ids, labels.tolist())):
        raise ValueError("Judgment safety labels must match bank-validation activations")
    source = pool[0]["metadata"]
    used_ids = set(source.get("train_prompt_ids") or []) | set(source.get("graph_prompt_ids") or [])
    if used_ids.intersection(ids):
        raise ValueError("Bank-validation prompt IDs must be disjoint from graph and training prompts")


def build_bank(trained_dir, val_activation_dir, output_path, layers, cfg, device="auto"):
    """Build a bank from complete baseline and both-sign response judgments."""
    _validate_config(cfg)
    if not cfg.get("efficacy_file"):
        raise ValueError("efficacy_file is required: judge baseline and both signs of every screened candidate")
    if not layers or len(set(layers)) != len(layers):
        raise ValueError("Provide distinct target layers")
    device = choose_device(str(device))
    outcomes, prompts = _read_judgments(cfg["efficacy_file"])
    baseline = _outcome_rates(outcomes, prompts)
    candidates = []
    for layer in layers:
        pool = _screen_layer(trained_dir, int(layer), cfg, device)
        _validate_bank_split(val_activation_dir, layer, pool, prompts)
        _score_layer(pool, outcomes, prompts, cfg, baseline)
        candidates.extend(pool)
    packed, count = _select_bank(candidates, float(cfg.get("rho", 0.95)))
    payload = {"layers": packed, "selected_count": count, "candidate_count": len(candidates),
               "config": dict(cfg), "efficacy_source": "response_judge"}
    atomic_torch_save(payload, output_path)
    return payload


def load_bank(path: str | Path, device: torch.device | str = "cpu") -> dict[str, Any]:
    payload = torch.load(path, map_location=device, weights_only=True)
    payload["layers"] = {int(key): value for key, value in payload["layers"].items()}
    return payload
