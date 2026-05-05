from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .activation import mean_pool
from .bank import load_bank
from .gate import load_gate
from .train import checkpoint_path, load_gsae_checkpoint
from .utils import choose_device


DEFAULT_REFUSAL = "Sorry, but I can't help with that request."


def get_decoder_layers(model) -> list:
    candidates = [
        ("model", "layers"),
        ("model", "decoder", "layers"),
        ("transformer", "h"),
        ("gpt_neox", "layers"),
    ]
    for path in candidates:
        obj = model
        ok = True
        for part in path:
            if not hasattr(obj, part):
                ok = False
                break
            obj = getattr(obj, part)
        if ok:
            return list(obj)
    raise AttributeError("Could not locate decoder layers on this HF model")


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
        self.models = {}
        for layer in self.layers:
            model, _ = load_gsae_checkpoint(checkpoint_path(trained_dir, layer), device=self.device)
            model.eval()
            self.models[layer] = model
        self.bank = load_bank(bank_path, device=self.device)
        self.gate = load_gate(gate_path)
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
    def prompt_risk(self, hf_outputs, attention_mask: torch.Tensor) -> float:
        hiddens = {}
        for layer in self.layers:
            hiddens[layer] = mean_pool(hf_outputs.hidden_states[layer + 1], attention_mask)
        return self.risk_from_layer_hiddens(hiddens)

    def layer_shift(self, layer: int, hidden_last: torch.Tensor, alpha: float) -> torch.Tensor:
        layer_bank = self.bank["layers"].get(int(layer))
        if not layer_bank:
            return hidden_last
        directions = layer_bank["directions"].to(hidden_last.device, dtype=hidden_last.dtype)
        deltas = layer_bank["deltas"].to(hidden_last.device, dtype=hidden_last.dtype)
        weights = layer_bank["weights"].to(hidden_last.device, dtype=hidden_last.dtype)
        dirs = F.normalize(directions, dim=1, eps=1e-8)
        h = hidden_last
        h_norm = h.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        cos = (h @ dirs.T) / h_norm
        coeff = cos * deltas.unsqueeze(0) * weights.unsqueeze(0)
        shift = coeff @ dirs
        return h - float(alpha) * shift


class _ControllerState:
    def __init__(self, artifacts: GSAEArtifacts, initial_risk: float) -> None:
        # initial_risk is recorded for telemetry only; per Algorithm 3 the
        # continuation gate starts from mu=0 and only activates after n_up
        # consecutive token-level threshold crossings above d_high.
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
        self.current: dict[int, torch.Tensor] = {}
        self.max_layer = max(artifacts.layers)

    def observe(self, layer: int, hidden_last: torch.Tensor) -> None:
        self.current[int(layer)] = hidden_last.detach().float()
        if int(layer) != self.max_layer:
            return
        if not all(layer_id in self.current for layer_id in self.artifacts.layers):
            return
        risk = self.artifacts.risk_from_layer_hiddens(self.current)
        self.current = {}
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


def generate_with_gsae(
    model,
    tokenizer,
    artifacts: GSAEArtifacts,
    prompt: str,
    max_new_tokens: int = 128,
    runtime: RuntimeConfig | None = None,
    generation_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    runtime = runtime or RuntimeConfig()
    generation_kwargs = generation_kwargs or {}
    device = artifacts.device
    encoded = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**encoded, output_hidden_states=True, use_cache=False)
    risk = artifacts.prompt_risk(outputs, encoded["attention_mask"])
    thresholds = artifacts.thresholds
    if risk >= float(thresholds["input_t_high"]):
        return {"prompt": prompt, "risk": risk, "mode": "immediate_refusal", "text": runtime.refusal_template}
    if risk < float(thresholds["input_t_low"]):
        out = model.generate(**encoded, max_new_tokens=max_new_tokens, **generation_kwargs)
        return {"prompt": prompt, "risk": risk, "mode": "standard", "text": tokenizer.decode(out[0], skip_special_tokens=True)}

    decoder_layers = get_decoder_layers(model)
    state = _ControllerState(artifacts, risk)
    handles = []

    def make_hook(layer_id: int):
        def hook(_module, _inputs, output):
            if isinstance(output, tuple):
                hidden = output[0]
                rest = output[1:]
            else:
                hidden = output
                rest = None
            hidden_last = hidden[:, -1, :]
            state.observe(layer_id, hidden_last)
            if state.active:
                modified = hidden.clone()
                modified[:, -1, :] = artifacts.layer_shift(layer_id, hidden_last, runtime.alpha).to(modified.dtype)
                hidden = modified
            if rest is None:
                return hidden
            return (hidden, *rest)

        return hook

    try:
        for layer in artifacts.layers:
            handles.append(decoder_layers[layer].register_forward_hook(make_hook(layer)))
        out = model.generate(**encoded, max_new_tokens=max_new_tokens, **generation_kwargs)
    finally:
        for handle in handles:
            handle.remove()
    return {"prompt": prompt, "risk": risk, "mode": "monitored", "text": tokenizer.decode(out[0], skip_special_tokens=True)}
