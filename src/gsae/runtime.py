from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .activation import capture_layer_hiddens, get_decoder_layers, mean_pool
from .bank import load_bank
from .gate import load_gate
from .train import checkpoint_path, load_gsae_checkpoint
from .utils import choose_device


DEFAULT_REFUSAL = "Sorry, but I can't help with that request."


def apply_layer_shift(
    hidden: torch.Tensor,
    directions: torch.Tensor,
    deltas: torch.Tensor,
    weights: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """Apply a normalized cosine-weighted intervention."""
    directions = directions.to(device=hidden.device, dtype=hidden.dtype)
    deltas = deltas.to(device=hidden.device, dtype=hidden.dtype)
    weights = weights.to(device=hidden.device, dtype=hidden.dtype)
    directions = F.normalize(directions, dim=-1, eps=1e-8)
    cosine = (hidden @ directions.T) / hidden.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    shift = (cosine * deltas * weights) @ directions
    return hidden - float(alpha) * shift


@dataclass
class RuntimeConfig:
    alpha: float = 2.5
    refusal_template: str = DEFAULT_REFUSAL


class GSAEArtifacts:
    def __init__(
        self,
        trained_dir: str | Path,
        bank_path: str | Path,
        gate_path: str | Path,
        layers: list[int],
        device: torch.device | str = "auto",
    ) -> None:
        self.device = choose_device(str(device))
        self.layers = [int(l) for l in layers]
        if not self.layers or len(set(self.layers)) != len(self.layers):
            raise ValueError("Target layers must be nonempty and unique")
        self.models = {}
        for layer in self.layers:
            model, _ = load_gsae_checkpoint(checkpoint_path(trained_dir, layer), device=self.device)
            model.eval()
            self.models[layer] = model
        self.bank = load_bank(bank_path, device=self.device)
        self.gate = load_gate(gate_path)
        if self.gate["layers"] != self.layers:
            raise ValueError("Runtime layer order must match the trained gate's feature order")
        if not set(self.bank["layers"]).issubset(self.layers):
            raise ValueError("Every bank source layer must be included in the runtime layers")
        self.thresholds = self.gate["thresholds"]

    @torch.no_grad()
    def encode_layer_hidden(self, layer: int, hidden: torch.Tensor) -> torch.Tensor:
        return self.models[int(layer)].encode(hidden.float().to(self.device))

    @torch.no_grad()
    def risk_from_layer_hiddens(self, layer_hiddens: dict[int, torch.Tensor]) -> float:
        zs = []
        for layer in self.layers:
            if layer not in layer_hiddens:
                raise KeyError(f"Missing hidden state for layer {layer}")
            zs.append(self.encode_layer_hidden(layer, layer_hiddens[layer]))
        z = torch.cat(zs, dim=-1).detach().cpu().numpy()
        clf = self.gate["classifier"]
        if hasattr(clf, "predict_proba"):
            return float(clf.predict_proba(z)[0, 1])
        return float(clf.predict(z)[0])

    @torch.no_grad()
    def prompt_risk(self, layer_hiddens: dict[int, torch.Tensor], attention_mask: torch.Tensor) -> float:
        hiddens = {layer: mean_pool(layer_hiddens[layer], attention_mask) for layer in self.layers}
        return self.risk_from_layer_hiddens(hiddens)

    def layer_shift(self, layer: int, hidden_last: torch.Tensor, alpha: float) -> torch.Tensor:
        layer_bank = self.bank["layers"].get(int(layer))
        if not layer_bank:
            return hidden_last
        return apply_layer_shift(
            hidden_last, layer_bank["directions"], layer_bank["deltas"], layer_bank["weights"], alpha
        )


class _ControllerState:
    def __init__(self, artifacts: GSAEArtifacts, initial_risk: float) -> None:
        self.artifacts = artifacts
        t = artifacts.thresholds
        self.d_low = float(t["cont_d_low"])
        self.d_high = float(t["cont_d_high"])
        self.n_up = int(t["n_up"])
        self.n_down = int(t["n_down"])
        self.initial_risk = float(initial_risk)
        self.active = False
        self.up_count = 0
        self.down_count = 0

    def observe(self, layer_hiddens: dict[int, torch.Tensor]) -> None:
        self.update(self.artifacts.risk_from_layer_hiddens(layer_hiddens))

    def update(self, risk: float) -> None:
        if risk > self.d_high:
            self.up_count += 1
            self.down_count = 0
            if self.up_count >= self.n_up:
                self.active = True
        elif risk < self.d_low:
            self.down_count += 1
            self.up_count = 0
            if self.down_count >= self.n_down:
                self.active = False
        else:
            self.up_count = 0
            self.down_count = 0


def _monitored_generation_kwargs(model, generation_kwargs: dict[str, Any]) -> dict[str, Any]:
    kwargs = dict(generation_kwargs)
    config = kwargs.get("generation_config", getattr(model, "generation_config", None))

    def option(name, default=None):
        return kwargs.get(name, getattr(config, name, default))

    if kwargs.get("use_cache") is True:
        raise ValueError("Monitored generation requires use_cache=False to observe the complete current prefix")
    if option("num_beams", 1) != 1 or option("num_return_sequences", 1) != 1:
        raise ValueError("Monitored generation supports one greedy or sampled sequence, with num_beams=1")
    for name in (
        "assistant_model", "prompt_lookup_num_tokens", "penalty_alpha", "dola_layers",
        "custom_generate", "past_key_values", "inputs_embeds",
    ):
        if option(name) is not None:
            raise ValueError(f"Monitored generation does not support {name}")
    if option("token_healing", False):
        raise ValueError("Monitored generation does not support token_healing")
    kwargs["use_cache"] = False
    return kwargs


@contextmanager
def _monitor_generation(model, artifacts: GSAEArtifacts, state: _ControllerState, alpha: float):
    """Observe each complete prefix before steering its next-token forward pass."""
    observing = False
    handles = []

    def observe_prefix(_module, args, kwargs):
        nonlocal observing
        if observing:
            return
        observing = True
        try:
            with capture_layer_hiddens(model, artifacts.layers) as hiddens:
                model(*args, **kwargs)
            state.observe({layer: hidden[:, -1, :] for layer, hidden in hiddens.items()})
        finally:
            observing = False

    def make_hook(layer):
        def steer(_module, _inputs, output):
            if observing or not state.active:
                return None
            hidden = output[0] if isinstance(output, (tuple, list)) else output
            modified = hidden.clone()
            modified[:, -1, :] = artifacts.layer_shift(layer, hidden[:, -1, :], alpha)
            if isinstance(output, tuple):
                return (modified, *output[1:])
            if isinstance(output, list):
                return [modified, *output[1:]]
            return modified

        return steer

    try:
        handles.append(model.register_forward_pre_hook(observe_prefix, with_kwargs=True))
        decoder_layers = get_decoder_layers(model)
        for layer in artifacts.layers:
            handles.append(decoder_layers[layer].register_forward_hook(make_hook(layer)))
        yield
    finally:
        for handle in handles:
            handle.remove()


def generate_with_gsae(
    model,
    tokenizer,
    artifacts: GSAEArtifacts,
    prompt: str,
    max_new_tokens: int = 128,
    runtime: RuntimeConfig | None = None,
    generation_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate one continuation with a shared controller across source layers.

    Recompute each complete prefix without a KV cache for observation and decoding.
    """
    runtime = runtime or RuntimeConfig()
    generation_kwargs = dict(generation_kwargs or {})
    generation_kwargs.setdefault("do_sample", False)
    device = artifacts.device
    encoded = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad(), capture_layer_hiddens(model, artifacts.layers) as hiddens:
        model(**encoded, use_cache=False)
    risk = artifacts.prompt_risk(hiddens, encoded["attention_mask"])
    thresholds = artifacts.thresholds
    if risk >= float(thresholds["input_t_high"]):
        return {"prompt": prompt, "risk": risk, "mode": "immediate_refusal", "text": runtime.refusal_template}
    if risk < float(thresholds["input_t_low"]):
        mode = "standard"
        with torch.no_grad():
            out = model.generate(**encoded, max_new_tokens=max_new_tokens, **generation_kwargs)
    else:
        mode = "monitored"
        generation_kwargs = _monitored_generation_kwargs(model, generation_kwargs)
        state = _ControllerState(artifacts, risk)
        with torch.no_grad(), _monitor_generation(model, artifacts, state, runtime.alpha):
            out = model.generate(**encoded, max_new_tokens=max_new_tokens, **generation_kwargs)
    sequences = out.sequences if hasattr(out, "sequences") else out
    new_tokens = sequences[0, encoded["input_ids"].shape[1]:]
    return {"prompt": prompt, "risk": risk, "mode": mode, "text": tokenizer.decode(new_tokens, skip_special_tokens=True)}
