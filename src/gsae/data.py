from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from .utils import atomic_torch_save, ensure_dir, label_to_int, read_jsonl, set_seed, write_jsonl


class ActivationDataset(Dataset):
    def __init__(self, activations: torch.Tensor, labels: torch.Tensor) -> None:
        if activations.ndim != 2:
            raise ValueError("activations must have shape [num_examples, hidden_dim]")
        if labels.ndim != 1 or labels.shape[0] != activations.shape[0]:
            raise ValueError("labels must have shape [num_examples]")
        self.activations = activations.float()
        self.labels = labels.long()

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.activations[idx], self.labels[idx]


def load_prompt_records(path: str | Path) -> tuple[list[str], torch.Tensor, list[str]]:
    rows = read_jsonl(path)
    prompts: list[str] = []
    labels: list[int] = []
    ids: list[str] = []
    for i, row in enumerate(rows):
        if "prompt" not in row:
            raise ValueError(f"Row {i} in {path} is missing a 'prompt' field")
        label = row.get("label", row.get("safety_label", row.get("harmful", 0)))
        prompts.append(str(row["prompt"]))
        labels.append(label_to_int(label))
        ids.append(str(row.get("id", i)))
    return prompts, torch.tensor(labels, dtype=torch.long), ids


def activation_path(root: str | Path, layer: int) -> Path:
    return Path(root) / f"layer_{int(layer)}.pt"


def save_layer_activations(
    root: str | Path,
    layer: int,
    activations: torch.Tensor,
    labels: torch.Tensor,
    ids: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    root = ensure_dir(root)
    path = activation_path(root, layer)
    atomic_torch_save(
        {
            "layer": int(layer),
            "activations": activations.detach().cpu().float(),
            "labels": labels.detach().cpu().long(),
            "ids": ids or [str(i) for i in range(int(labels.shape[0]))],
            "metadata": metadata or {},
        },
        path,
    )
    return path


def load_layer_activations(root: str | Path, layer: int) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    path = activation_path(root, layer)
    if not path.exists():
        raise FileNotFoundError(f"Missing activation shard: {path}")
    payload = torch.load(path, map_location="cpu")
    return payload["activations"].float(), payload["labels"].long(), dict(payload.get("metadata", {}))


def make_synthetic_dataset(
    output_dir: str | Path,
    layers: list[int],
    num_train: int = 256,
    num_val: int = 128,
    hidden_dim: int = 64,
    seed: int = 7,
) -> None:
    """Create a small activation-only dataset for smoke tests.

    The harmful class is distributed across a few smooth-ish latent factors so
    graph regularization has a real signal without needing a Hugging Face model.
    """

    set_seed(seed)
    output_dir = ensure_dir(output_dir)
    prompts_train = []
    prompts_val = []
    for split, n, rows in [("train", num_train, prompts_train), ("val", num_val, prompts_val)]:
        labels = torch.arange(n) % 2
        labels = labels[torch.randperm(n)]
        for i, label in enumerate(labels.tolist()):
            rows.append(
                {
                    "id": f"{split}-{i}",
                    "prompt": f"{'harmful' if label else 'benign'} synthetic prompt {i}",
                    "label": int(label),
                }
            )
        write_jsonl(output_dir / f"{split}.jsonl", rows)

    for split, n, rows in [("train", num_train, prompts_train), ("val", num_val, prompts_val)]:
        labels = torch.tensor([int(row["label"]) for row in rows], dtype=torch.long)
        ids = [str(row["id"]) for row in rows]
        base = torch.randn(n, hidden_dim) * 0.55
        signal = torch.zeros(hidden_dim)
        signal[: hidden_dim // 4] = torch.linspace(1.2, 0.4, hidden_dim // 4)
        signal[hidden_dim // 2 : hidden_dim // 2 + hidden_dim // 8] = -0.8
        for layer in layers:
            layer_noise = torch.randn(n, hidden_dim) * 0.12
            layer_scale = 1.0 + (int(layer) % 7) * 0.04
            x = base + layer_noise + (labels.float().unsqueeze(1) * 2 - 1) * signal * layer_scale
            save_layer_activations(
                output_dir / split,
                layer,
                x,
                labels,
                ids,
                {"source": "synthetic", "hidden_dim": hidden_dim, "split": split},
            )
