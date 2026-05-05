from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import trange

from .data import ActivationDataset, load_layer_activations
from .graph import build_threshold_cosine_laplacian, graph_density, load_layer_laplacian, save_layer_laplacian
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
    "normalize_decoder": True,
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
) -> dict[str, Any]:
    device = choose_device(str(device))
    output_dir = ensure_dir(Path(output_dir) / f"layer_{int(layer)}")
    hidden_dim = int(activations.shape[1])
    latent_dim = int(cfg.get("latent_dim") or hidden_dim * int(cfg.get("latent_multiplier", 16)))
    set_seed(int(cfg.get("seed", 7)) + int(layer))

    if laplacian is None:
        laplacian = build_threshold_cosine_laplacian(
            activations,
            threshold=float(cfg.get("graph_threshold", 0.6)),
            chunk_size=int(cfg.get("graph_chunk_size", 1024)),
            device=device,
        )
    laplacian = laplacian.coalesce().to(device)

    dataset = ActivationDataset(activations, labels)
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
        if bool(cfg.get("normalize_decoder", True)):
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
    }
    atomic_torch_save({"model": model.cpu().state_dict(), "metadata": metadata, "history": history}, output_dir / "model.pt")
    save_layer_laplacian(output_dir, layer, laplacian.detach().cpu(), metadata)
    return metadata


def train_from_activation_dir(
    activation_dir: str | Path,
    output_dir: str | Path,
    cfg: dict[str, Any],
    layers: list[int],
    device: torch.device | str = "auto",
) -> list[dict[str, Any]]:
    output_dir = ensure_dir(output_dir)
    results = []
    for layer in layers:
        activations, labels, _ = load_layer_activations(activation_dir, layer)
        lap_path = Path(output_dir) / f"layer_{int(layer)}" / f"layer_{int(layer)}_laplacian.pt"
        laplacian = load_layer_laplacian(lap_path.parent, layer) if lap_path.exists() else None
        results.append(train_layer(activations, labels, output_dir, layer, cfg, laplacian=laplacian, device=device))
    return results
