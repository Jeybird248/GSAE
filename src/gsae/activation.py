from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from .data import load_prompt_records, save_layer_activations
from .utils import choose_device, ensure_dir


def _dtype_from_name(name: str | None) -> torch.dtype | None:
    if name in {None, "auto"}:
        return None
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype {name!r}")
    return mapping[name]


def mean_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.to(hidden.device).unsqueeze(-1).float()
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


def collect_hf_activations(
    model_name_or_path: str,
    prompts_path: str | Path,
    output_dir: str | Path,
    layers: list[int],
    batch_size: int = 4,
    max_length: int = 1024,
    device: torch.device | str = "auto",
    dtype: str | None = "auto",
    trust_remote_code: bool = False,
) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = choose_device(str(device))
    output_dir = ensure_dir(output_dir)
    prompts, labels, ids = load_prompt_records(prompts_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    torch_dtype = _dtype_from_name(dtype)
    model_kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
    if torch_dtype is not None:
        model_kwargs["torch_dtype"] = torch_dtype
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
    model.to(device)
    model.eval()

    buffers = {int(layer): [] for layer in layers}
    for start in tqdm(range(0, len(prompts), batch_size), desc="collect activations"):
        batch_prompts = prompts[start : start + batch_size]
        encoded = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)
        with torch.no_grad():
            out = model(**encoded, output_hidden_states=True, use_cache=False)
        hidden_states = out.hidden_states
        for layer in layers:
            hs_index = int(layer) + 1
            if hs_index >= len(hidden_states):
                raise ValueError(
                    f"Layer {layer} is out of range for model with {len(hidden_states) - 1} decoder layers"
                )
            pooled = mean_pool(hidden_states[hs_index], encoded["attention_mask"])
            buffers[int(layer)].append(pooled.detach().cpu().float())

    for layer, chunks in buffers.items():
        save_layer_activations(
            output_dir,
            layer,
            torch.cat(chunks, dim=0),
            labels,
            ids,
            {"model": model_name_or_path, "prompts": str(prompts_path), "max_length": max_length},
        )
