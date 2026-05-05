from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str | None = None) -> torch.device:
    if requested and requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def load_yaml(path: str | Path) -> dict[str, Any]:
    import yaml

    with Path(path).open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return loaded


def save_yaml(path: str | Path, data: dict[str, Any]) -> None:
    import yaml

    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


def atomic_torch_save(obj: Any, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def minmax(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if x.numel() == 0:
        return x
    return (x - x.min()) / (x.max() - x.min() + eps)


def parse_layers(value: str | Iterable[int]) -> list[int]:
    if isinstance(value, str):
        return [int(v.strip()) for v in value.split(",") if v.strip()]
    return [int(v) for v in value]


def label_to_int(label: Any) -> int:
    if isinstance(label, bool):
        return int(label)
    if isinstance(label, (int, np.integer)):
        return int(label)
    normalized = str(label).strip().lower()
    harmful = {"1", "harmful", "unsafe", "bad", "jailbreak", "malicious"}
    benign = {"0", "benign", "safe", "good", "harmless"}
    if normalized in harmful:
        return 1
    if normalized in benign:
        return 0
    raise ValueError(f"Unknown label {label!r}; use 0/1 or benign/harmful")
