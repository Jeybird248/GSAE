from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from .data import load_prompt_records, save_layer_activations
from .utils import choose_device, ensure_dir, read_jsonl


def get_decoder_layers(model) -> list:
    for path in [("model", "layers"), ("model", "decoder", "layers"),
                 ("transformer", "h"), ("gpt_neox", "layers")]:
        obj = model
        for part in path:
            obj = getattr(obj, part, None)
        if obj is not None:
            return list(obj)
    raise AttributeError("Could not locate decoder layers on this HF model")


@contextmanager
def capture_layer_hiddens(model, layers: list[int]):
    """Capture decoder block outputs, including the final block before normalization."""
    blocks = get_decoder_layers(model)
    _validate_layers(layers, len(blocks))
    captured: dict[int, torch.Tensor] = {}
    handles = []

    def make_hook(layer):
        def hook(_module, _inputs, output):
            captured[layer] = output[0] if isinstance(output, (tuple, list)) else output
        return hook

    try:
        for layer in layers:
            handles.append(blocks[layer].register_forward_hook(make_hook(layer)))
        yield captured
    finally:
        for handle in handles:
            handle.remove()


def _validate_layers(layers: list[int], count: int | None = None) -> None:
    if not layers or len(set(layers)) != len(layers):
        raise ValueError("layers must be nonempty and unique")
    if any(not isinstance(layer, int) or layer < 0 for layer in layers):
        raise ValueError("layers must contain nonnegative integers")
    if count is not None and max(layers) >= count:
        raise ValueError(f"Layer {max(layers)} is out of range for model with {count} decoder layers")


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


def _load_hf(model_name_or_path, device, dtype, trust_remote_code):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
    torch_dtype = _dtype_from_name(dtype)
    if torch_dtype is not None:
        model_kwargs["torch_dtype"] = torch_dtype
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
    model.to(device)
    model.eval()
    return model, tokenizer


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
    _validate_layers(layers)
    if batch_size < 1 or max_length < 1:
        raise ValueError("batch_size and max_length must be positive")
    device = choose_device(str(device))
    output_dir = ensure_dir(output_dir)
    prompts, labels, ids = load_prompt_records(prompts_path)
    model, tokenizer = _load_hf(model_name_or_path, device, dtype, trust_remote_code)

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
        with torch.no_grad(), capture_layer_hiddens(model, layers) as hidden_states:
            model(**encoded, use_cache=False)
        for layer in layers:
            pooled = mean_pool(hidden_states[layer], encoded["attention_mask"])
            buffers[int(layer)].append(pooled.detach().cpu().float())

    for layer, chunks in buffers.items():
        save_layer_activations(
            output_dir,
            layer,
            torch.cat(chunks, dim=0),
            labels,
            ids,
            {"model": model_name_or_path, "prompts": str(prompts_path), "max_length": max_length,
             "representation": "pooled_decoder_block_output"},
        )


def collect_hf_prefix_activations(
    model_name_or_path: str,
    prompts_path: str | Path,
    output_dir: str | Path,
    layers: list[int],
    max_prefix_tokens: int = 16,
    max_length: int = 1024,
    device: torch.device | str = "auto",
    dtype: str | None = "auto",
    trust_remote_code: bool = False,
) -> None:
    """Collect unsteered token states from generated or supplied continuations."""
    _validate_layers(layers)
    if not 1 <= max_prefix_tokens <= 16 or max_length < 1:
        raise ValueError("max_prefix_tokens must be in [1, 16] and max_length must be positive")
    prompts, prompt_labels, prompt_ids = load_prompt_records(prompts_path)
    rows = read_jsonl(prompts_path)
    device = choose_device(str(device))
    model, tokenizer = _load_hf(model_name_or_path, device, dtype, trust_remote_code)
    buffers = {layer: [] for layer in layers}
    ids, source_ids, labels, positions = [], [], [], []
    for prompt, label, prompt_id, row in tqdm(
        list(zip(prompts, prompt_labels.tolist(), prompt_ids, rows)), desc="collect prefix activations"
    ):
        encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length).to(device)
        prompt_length = encoded["input_ids"].shape[1]
        with torch.no_grad():
            sequence = _continuation_sequence(model, tokenizer, encoded, row, max_prefix_tokens)
        prefix_count = min(max_prefix_tokens, sequence.shape[1] - prompt_length)
        if prefix_count < 1:
            raise ValueError(f"Prompt {prompt_id!r} has no continuation tokens")
        sequence = sequence[:, :prompt_length + prefix_count]
        with torch.no_grad(), capture_layer_hiddens(model, layers) as hiddens:
            model(input_ids=sequence, attention_mask=torch.ones_like(sequence), use_cache=False)
        for layer in layers:
            buffers[layer].append(hiddens[layer][0, prompt_length:].detach().cpu().float())
        ids.extend(f"{prompt_id}:prefix:{position}" for position in range(1, prefix_count + 1))
        source_ids.extend([prompt_id] * prefix_count)
        labels.extend([label] * prefix_count)
        positions.extend(range(1, prefix_count + 1))
    metadata = {"model": model_name_or_path, "prompts": str(prompts_path),
                "source_prompt_ids": source_ids, "prefix_positions": positions,
                "max_prefix_tokens": max_prefix_tokens, "representation": "token_decoder_block_output"}
    for layer, chunks in buffers.items():
        save_layer_activations(output_dir, layer, torch.cat(chunks), torch.tensor(labels), ids, metadata)


def _continuation_sequence(model, tokenizer, encoded, row, max_prefix_tokens):
    if "continuation" not in row:
        return model.generate(**encoded, max_new_tokens=max_prefix_tokens, do_sample=False,
                              use_cache=False, pad_token_id=tokenizer.pad_token_id)
    if not isinstance(row["continuation"], str):
        raise ValueError("continuation must be text")
    tokens = tokenizer(row["continuation"], return_tensors="pt", add_special_tokens=False)["input_ids"]
    tokens = tokens[:, :max_prefix_tokens].to(encoded["input_ids"].device)
    return torch.cat([encoded["input_ids"], tokens], dim=1)
