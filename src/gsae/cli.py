from __future__ import annotations

import argparse
import json
from pathlib import Path

from .activation import collect_hf_activations
from .bank import DEFAULT_BANK, build_bank
from .data import make_synthetic_dataset
from .gate import DEFAULT_GATE, train_gate
from .runtime import GSAEArtifacts, RuntimeConfig, generate_with_gsae
from .train import DEFAULT_TRAINING, train_from_activation_dir
from .utils import deep_update, ensure_dir, load_yaml, parse_layers, read_jsonl, save_yaml, set_seed, write_jsonl


def _load_cfg(path: str | None) -> dict:
    if not path:
        return {}
    return load_yaml(path)


def _section(cfg: dict, name: str, defaults: dict) -> dict:
    return deep_update(defaults, cfg.get(name, {}))


def cmd_make_synthetic(args: argparse.Namespace) -> None:
    layers = parse_layers(args.layers)
    make_synthetic_dataset(
        args.output,
        layers,
        num_train=args.num_train,
        num_val=args.num_val,
        hidden_dim=args.hidden_dim,
        seed=args.seed,
    )


def cmd_collect(args: argparse.Namespace) -> None:
    collect_hf_activations(
        args.model,
        args.prompts,
        args.output,
        parse_layers(args.layers),
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=args.device,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
    )


def cmd_train(args: argparse.Namespace) -> None:
    cfg = _section(_load_cfg(args.config), "training", DEFAULT_TRAINING)
    if args.steps is not None:
        cfg["steps"] = args.steps
    if args.latent_dim is not None:
        cfg["latent_dim"] = args.latent_dim
    layers = parse_layers(args.layers or cfg["layers"])
    results = train_from_activation_dir(args.activations, args.output, cfg, layers, device=args.device)
    save_yaml(Path(args.output) / "training_summary.yaml", {"layers": results, "config": cfg})


def cmd_build_bank(args: argparse.Namespace) -> None:
    cfg = _section(_load_cfg(args.config), "bank", DEFAULT_BANK)
    if args.candidate_pool_size is not None:
        cfg["candidate_pool_size"] = args.candidate_pool_size
    layers = parse_layers(args.layers)
    payload = build_bank(args.trained_dir, args.val_activations, args.output, layers, cfg, device=args.device)
    print(json.dumps({"selected_count": payload["selected_count"], "candidate_count": payload["candidate_count"]}))


def cmd_train_gate(args: argparse.Namespace) -> None:
    cfg = _section(_load_cfg(args.config), "gate", DEFAULT_GATE)
    layers = parse_layers(args.layers)
    payload = train_gate(args.trained_dir, args.activations, args.output, layers, cfg, device=args.device)
    print(json.dumps({"num_examples": payload["num_examples"], "latent_dim": payload["latent_dim"]}))


def cmd_generate(args: argparse.Namespace) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    layers = parse_layers(args.layers)
    artifacts = GSAEArtifacts(args.trained_dir, args.bank, args.gate, layers, device=args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.float16 if args.dtype in {"fp16", "float16"} else torch.bfloat16 if args.dtype in {"bf16", "bfloat16"} else None
    kwargs = {"trust_remote_code": args.trust_remote_code}
    if dtype is not None:
        kwargs["torch_dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs).to(artifacts.device)
    model.eval()
    rows = read_jsonl(args.prompts)
    outputs = []
    for row in rows:
        result = generate_with_gsae(
            model,
            tokenizer,
            artifacts,
            row["prompt"],
            max_new_tokens=args.max_new_tokens,
            runtime=RuntimeConfig(alpha=args.alpha, refusal_template=args.refusal_template),
            generation_kwargs={"do_sample": args.do_sample, "temperature": args.temperature} if args.do_sample else {"do_sample": False},
        )
        result["id"] = row.get("id")
        outputs.append(result)
    write_jsonl(args.output, outputs)


def cmd_smoke(args: argparse.Namespace) -> None:
    work = ensure_dir(args.work_dir)
    layers = parse_layers(args.layers)
    set_seed(args.seed)
    make_synthetic_dataset(work / "data", layers, args.num_train, args.num_val, args.hidden_dim, args.seed)
    train_cfg = deep_update(DEFAULT_TRAINING, {
        "layers": layers,
        "latent_dim": args.latent_dim,
        "steps": args.steps,
        "batch_size": 16,
        "graph_threshold": args.graph_threshold,
        "graph_chunk_size": 256,
    })
    train_from_activation_dir(work / "data" / "train", work / "artifacts" / "gsae", train_cfg, layers, device=args.device)
    bank_cfg = deep_update(DEFAULT_BANK, {"candidate_pool_size": args.candidate_pool_size, "rho": 0.80})
    bank = build_bank(
        work / "artifacts" / "gsae",
        work / "data" / "val",
        work / "artifacts" / "bank.pt",
        layers,
        bank_cfg,
        device=args.device,
    )
    gate_cfg = deep_update(DEFAULT_GATE, {"n_estimators": 20})
    gate = train_gate(
        work / "artifacts" / "gsae",
        work / "data" / "val",
        work / "artifacts" / "gate.joblib",
        layers,
        gate_cfg,
        device=args.device,
    )
    summary = {"selected_count": bank["selected_count"], "gate_examples": gate["num_examples"], "work_dir": str(work)}
    print(json.dumps(summary, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GSAE safety steering pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("make-synthetic", help="create synthetic activation shards for smoke tests")
    p.add_argument("--output", required=True)
    p.add_argument("--layers", default="0,1")
    p.add_argument("--num-train", type=int, default=256)
    p.add_argument("--num-val", type=int, default=128)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--seed", type=int, default=7)
    p.set_defaults(func=cmd_make_synthetic)

    p = sub.add_parser("collect-activations", help="collect mean-pooled HF hidden states")
    p.add_argument("--model", required=True)
    p.add_argument("--prompts", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--layers", required=True)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="auto")
    p.add_argument("--trust-remote-code", action="store_true")
    p.set_defaults(func=cmd_collect)

    p = sub.add_parser("train", help="train graph-regularized SAEs from activation shards")
    p.add_argument("--config")
    p.add_argument("--activations", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--layers")
    p.add_argument("--steps", type=int)
    p.add_argument("--latent-dim", type=int)
    p.add_argument("--device", default="auto")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("build-bank", help="construct steering bank")
    p.add_argument("--config")
    p.add_argument("--trained-dir", required=True)
    p.add_argument("--val-activations", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--layers", required=True)
    p.add_argument("--candidate-pool-size", type=int)
    p.add_argument("--device", default="auto")
    p.set_defaults(func=cmd_build_bank)

    p = sub.add_parser("train-gate", help="train calibrated runtime gate")
    p.add_argument("--config")
    p.add_argument("--trained-dir", required=True)
    p.add_argument("--activations", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--layers", required=True)
    p.add_argument("--device", default="auto")
    p.set_defaults(func=cmd_train_gate)

    p = sub.add_parser("generate", help="generate with the two-gate GSAE controller")
    p.add_argument("--model", required=True)
    p.add_argument("--prompts", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--trained-dir", required=True)
    p.add_argument("--bank", required=True)
    p.add_argument("--gate", required=True)
    p.add_argument("--layers", required=True)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--alpha", type=float, default=2.5)
    p.add_argument("--refusal-template", default="Sorry, but I can't help with that request.")
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="auto")
    p.add_argument("--do-sample", action="store_true")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--trust-remote-code", action="store_true")
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser("smoke", help="run an end-to-end synthetic pipeline")
    p.add_argument("--work-dir", default="/tmp/gsae_smoke")
    p.add_argument("--layers", default="0,1")
    p.add_argument("--hidden-dim", type=int, default=32)
    p.add_argument("--latent-dim", type=int, default=64)
    p.add_argument("--num-train", type=int, default=96)
    p.add_argument("--num-val", type=int, default=48)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--candidate-pool-size", type=int, default=16)
    p.add_argument("--graph-threshold", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", default="auto")
    p.set_defaults(func=cmd_smoke)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
