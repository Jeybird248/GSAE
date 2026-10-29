from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from .utils import atomic_torch_save, ensure_dir, label_to_int, read_jsonl


class ActivationDataset(Dataset):
    def __init__(self, activations: torch.Tensor, labels: torch.Tensor) -> None:
        if activations.ndim != 2:
            raise ValueError("activations must have shape [num_examples, hidden_dim]")
        if labels.ndim != 1 or labels.shape[0] != activations.shape[0]:
            raise ValueError("labels must have shape [num_examples]")
        if not activations.shape[0] or not activations.shape[1]:
            raise ValueError("activations must contain examples and hidden dimensions")
        if not torch.isfinite(activations).all():
            raise ValueError("activations must contain only finite values")
        if not ((labels == 0) | (labels == 1)).all():
            raise ValueError("labels must be binary 0/1 values")
        self.activations = activations.float()
        self.labels = labels.long()

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.activations[idx], self.labels[idx]


def load_prompt_records(path: str | Path) -> tuple[list[str], torch.Tensor, list[str]]:
    rows = read_jsonl(path)
    if not rows:
        raise ValueError(f"No prompt records in {path}")
    prompts: list[str] = []
    labels: list[int] = []
    ids: list[str] = []
    for i, row in enumerate(rows):
        if "prompt" not in row:
            raise ValueError(f"Row {i} in {path} is missing a 'prompt' field")
        label = row.get("label", row.get("safety_label", row.get("harmful")))
        prompts.append(str(row["prompt"]))
        labels.append(label_to_int(label))
        ids.append(str(row.get("id", i)))
    if len(set(ids)) != len(ids):
        raise ValueError(f"Prompt IDs must be unique in {path}")
    if any(label not in (0, 1) for label in labels):
        raise ValueError("Prompt labels must be binary 0/1 values")
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
    ActivationDataset(activations, labels)
    if ids is not None and (len(ids) != len(labels) or len(set(ids)) != len(ids)):
        raise ValueError("Activation IDs must be unique and match the number of rows")
    root = ensure_dir(root)
    path = activation_path(root, layer)
    atomic_torch_save(
        {
            "layer": int(layer),
            "activations": activations.detach().cpu().float(),
            "labels": labels.detach().cpu().long(),
            "ids": ids,
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
    activations, labels = payload["activations"], payload["labels"]
    ActivationDataset(activations, labels)
    metadata = dict(payload.get("metadata", {}))
    ids = payload.get("ids")
    if ids is not None:
        if len(ids) != len(labels) or len(set(ids)) != len(ids):
            raise ValueError(f"Invalid activation IDs in {path}")
        metadata["ids"] = list(ids)
    return activations.float(), labels.long(), metadata
