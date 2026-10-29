from __future__ import annotations

from pathlib import Path
import math

import torch
import torch.nn.functional as F

from .utils import atomic_torch_save, ensure_dir


def build_threshold_cosine_laplacian(
    activations: torch.Tensor,
    threshold: float = 0.6,
    chunk_size: int = 1024,
    center: bool = False,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Build a masked normalized Laplacian from [num_examples, hidden_dim] states.

    Keep positive cosine edges at or above threshold. Centering subtracts each
    neuron's mean activation. Compute chunk_size rows at a time and return a
    CPU sparse tensor.
    """

    if activations.ndim != 2:
        raise ValueError("activations must have shape [num_examples, hidden_dim]")
    if min(activations.shape) == 0 or not torch.isfinite(activations).all():
        raise ValueError("activations must be nonempty and finite")
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be finite and in [0, 1]")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    x = activations.float().to(device).T.contiguous()  # [hidden_dim, num_examples]
    if center:
        x = x - x.mean(dim=1, keepdim=True)
    x = F.normalize(x, p=2, dim=1, eps=1e-12)
    hidden_dim = x.shape[0]
    rows: list[torch.Tensor] = []
    cols: list[torch.Tensor] = []
    vals: list[torch.Tensor] = []

    for start in range(0, hidden_dim, chunk_size):
        stop = min(start + chunk_size, hidden_dim)
        sim = x[start:stop] @ x.T
        local_rows = torch.arange(start, stop, device=device).unsqueeze(1)
        global_cols = torch.arange(hidden_dim, device=device).unsqueeze(0)
        # Compute each undirected edge once so threshold rounding cannot break symmetry.
        mask = (sim >= threshold) & (sim > 0) & (local_rows < global_cols)
        nz = mask.nonzero(as_tuple=False)
        if nz.numel() == 0:
            continue
        rows.append((nz[:, 0] + start).cpu())
        cols.append(nz[:, 1].cpu())
        vals.append(sim[mask].detach().cpu())

    if not rows:
        indices = torch.empty(2, 0, dtype=torch.long)
        values = torch.empty(0, dtype=torch.float32)
        return torch.sparse_coo_tensor(indices, values, (hidden_dim, hidden_dim)).coalesce()

    row = torch.cat(rows).long()
    col = torch.cat(cols).long()
    val = torch.cat(vals).float().clamp_min(0)
    row, col, val = torch.cat([row, col]), torch.cat([col, row]), torch.cat([val, val])
    deg = torch.zeros(hidden_dim, dtype=torch.float32)
    deg.index_add_(0, row, val)
    nonzero = deg > 0
    inv_sqrt = torch.zeros_like(deg)
    inv_sqrt[nonzero] = deg[nonzero].rsqrt()

    off_values = -val * inv_sqrt[row] * inv_sqrt[col]
    diag_idx = nonzero.nonzero(as_tuple=False).flatten()
    all_rows = torch.cat([row, diag_idx])
    all_cols = torch.cat([col, diag_idx])
    all_vals = torch.cat([off_values, torch.ones_like(diag_idx, dtype=torch.float32)])
    lap = torch.sparse_coo_tensor(
        torch.stack([all_rows, all_cols]), all_vals, (hidden_dim, hidden_dim)
    ).coalesce()
    return lap


def graph_density(laplacian: torch.Tensor) -> float:
    laplacian = laplacian.coalesce()
    hidden_dim = laplacian.shape[0]
    if hidden_dim <= 1:
        return 0.0
    row, col = laplacian.indices()
    off_diag = (row != col).sum().item()
    return float(off_diag) / float(hidden_dim * (hidden_dim - 1))


def save_laplacian(path: str | Path, laplacian: torch.Tensor, metadata: dict | None = None) -> None:
    atomic_torch_save({"laplacian": laplacian.cpu().coalesce(), "metadata": metadata or {}}, path)


def load_laplacian(path: str | Path) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu")
    return payload["laplacian"].coalesce()


def save_layer_laplacian(root: str | Path, layer: int, laplacian: torch.Tensor, metadata: dict | None = None) -> Path:
    root = ensure_dir(root)
    path = root / f"layer_{int(layer)}_laplacian.pt"
    save_laplacian(path, laplacian, metadata)
    return path


def load_layer_laplacian(root: str | Path, layer: int) -> torch.Tensor:
    return load_laplacian(Path(root) / f"layer_{int(layer)}_laplacian.pt")
