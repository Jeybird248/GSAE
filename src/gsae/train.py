from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import trange

from .data import ActivationDataset, load_layer_activations
from .graph import build_threshold_cosine_laplacian, graph_density, save_layer_laplacian
from .model import GraphSparseAutoencoder, gsae_loss
from .utils import atomic_torch_save, choose_device, ensure_dir, set_seed


DEFAULT_TRAINING: dict[str, Any] = {
    "seed": 7,
    "layers": [6, 8, 10, 12],
    "latent_multiplier": 16,
    "latent_dim": None,
    "batch_size": 16,
    "steps": 500,
    "lr": 1.0e-3,
    "lambda_sparse": 1.0e-4,
    "lambda_sup": 2.0e-2,
    "lambda_graph": 1.0e-3,
    "graph_threshold": 0.6,
    "graph_chunk_size": 1024,
    "normalize_decoder": False,
}


def checkpoint_path(output_dir: str | Path, layer: int) -> Path:
    return Path(output_dir) / f"layer_{int(layer)}" / "model.pt"


def load_gsae_checkpoint(path: str | Path, device: torch.device | str = "cpu") -> tuple[GraphSparseAutoencoder, dict[str, Any]]:
    payload = torch.load(path, map_location=device)
    metadata = dict(payload["metadata"])
    model = GraphSparseAutoencoder(metadata["hidden_dim"], metadata["latent_dim"])
    model.load_state_dict(payload["model"])
    model.to(device)
    model.eval()
    return model, metadata


def train_layer(
    activations: torch.Tensor,
    labels: torch.Tensor,
    output_dir: str | Path,
    layer: int,
    cfg: dict[str, Any],
    laplacian: torch.Tensor | None = None,
    device: torch.device | str = "auto",
    source_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    dataset = ActivationDataset(activations, labels)
    hidden_dim, latent_dim = _training_dimensions(activations, laplacian, cfg)
    device = choose_device(str(device))
    output_dir = ensure_dir(Path(output_dir) / f"layer_{int(layer)}")
    set_seed(int(cfg.get("seed", 7)) + int(layer))

    laplacian = laplacian.coalesce().to(device)

    loader = DataLoader(
        dataset,
        batch_size=int(cfg.get("batch_size", 16)),
        shuffle=True,
        drop_last=False,
    )
    iterator = iter(loader)
    model = GraphSparseAutoencoder(hidden_dim, latent_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg.get("lr", 1.0e-3)))
    history: list[dict[str, float]] = []
    steps = int(cfg.get("steps", 500))

    pbar = trange(steps, desc=f"train layer {layer}", leave=False)
    for _ in pbar:
        try:
            x, y = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            x, y = next(iterator)
        x = x.to(device)
        y = y.to(device)
        opt.zero_grad(set_to_none=True)
        loss, terms = gsae_loss(
            model,
            x,
            y,
            laplacian,
            lambda_sparse=float(cfg.get("lambda_sparse", 1.0e-4)),
            lambda_sup=float(cfg.get("lambda_sup", 2.0e-2)),
            lambda_graph=float(cfg.get("lambda_graph", 1.0e-3)),
        )
        loss.backward()
        opt.step()
        if bool(cfg.get("normalize_decoder", False)):
            model.normalize_decoder_columns()
        history.append(terms)
        if len(history) % 25 == 0:
            pbar.set_postfix({k: f"{v:.4g}" for k, v in terms.items()})

    metadata = {
        "layer": int(layer),
        "hidden_dim": hidden_dim,
        "latent_dim": latent_dim,
        "num_examples": int(activations.shape[0]),
        "config": dict(cfg),
        "graph_density": graph_density(laplacian.detach().cpu()),
        **(source_metadata or {}),
    }
    atomic_torch_save({"model": model.cpu().state_dict(), "metadata": metadata, "history": history}, output_dir / "model.pt")
    save_layer_laplacian(output_dir, layer, laplacian.detach().cpu(), metadata)
    return metadata


def _training_dimensions(activations, laplacian, cfg):
    if laplacian is None:
        raise ValueError("A graph Laplacian from separate held-out activations is required")
    if int(cfg.get("batch_size", 16)) < 1 or int(cfg.get("steps", 500)) < 1:
        raise ValueError("batch_size and steps must be positive")
    hidden_dim = int(activations.shape[1])
    latent_dim = cfg.get("latent_dim")
    latent_dim = int(hidden_dim * int(cfg.get("latent_multiplier", 16)) if latent_dim is None else latent_dim)
    if latent_dim < 1 or float(cfg.get("lr", 1.0e-3)) <= 0:
        raise ValueError("latent_dim and lr must be positive")
    if laplacian.shape != (hidden_dim, hidden_dim):
        raise ValueError("Graph Laplacian dimensions must match the activation hidden dimension")
    if not torch.isfinite(laplacian.coalesce().values()).all():
        raise ValueError("Graph Laplacian must contain only finite values")
    for key in ("lr", "lambda_sparse", "lambda_sup", "lambda_graph"):
        value = float(cfg.get(key, DEFAULT_TRAINING[key]))
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    return hidden_dim, latent_dim


def train_from_activation_dir(
    activation_dir: str | Path,
    graph_activation_dir: str | Path,
    output_dir: str | Path,
    cfg: dict[str, Any],
    layers: list[int],
    device: torch.device | str = "auto",
) -> list[dict[str, Any]]:
    if Path(activation_dir).resolve() == Path(graph_activation_dir).resolve():
        raise ValueError("Graph activations must be separate from autoencoder training activations")
    if not layers or len(set(layers)) != len(layers):
        raise ValueError("layers must be nonempty and unique")
    threshold = float(cfg.get("graph_threshold", 0.6))
    chunk_size = int(cfg.get("graph_chunk_size", 1024))
    if not 0 <= threshold <= 1 or chunk_size < 1:
        raise ValueError("graph_threshold must be in [0, 1] and graph_chunk_size must be positive")
    device = choose_device(str(device))
    output_dir = ensure_dir(output_dir)
    results = []
    train_reference, graph_reference = None, None
    for layer in layers:
        activations, labels, metadata = load_layer_activations(activation_dir, layer)
        graph_activations, graph_labels, graph_metadata = load_layer_activations(graph_activation_dir, layer)
        train_reference = _check_alignment(train_reference, labels, metadata, "Training")
        graph_reference = _check_alignment(graph_reference, graph_labels, graph_metadata, "Graph")
        train_ids, graph_ids = metadata.get("ids"), graph_metadata.get("ids")
        if train_ids is None or graph_ids is None:
            raise ValueError("Explicit prompt IDs are required for training and graph activations")
        if set(train_ids).intersection(graph_ids):
            raise ValueError("Graph and autoencoder training prompt IDs must be disjoint")
        if graph_activations.shape[1] != activations.shape[1]:
            raise ValueError("Graph and training activation hidden dimensions must match")
        laplacian = build_threshold_cosine_laplacian(
            graph_activations, threshold=threshold, chunk_size=chunk_size, device=device,
        )
        results.append(train_layer(
            activations, labels, output_dir, layer, cfg, laplacian=laplacian, device=device,
            source_metadata={"train_prompt_ids": train_ids, "graph_prompt_ids": graph_ids,
                             "graph_activation_dir": str(graph_activation_dir)},
        ))
    return results


def _check_alignment(reference, labels, metadata, name):
    current = (labels, metadata.get("ids"))
    if reference is not None:
        if not torch.equal(reference[0], labels) or reference[1] != current[1]:
            raise ValueError(f"{name} activation row IDs and labels must align across layers")
    return current
