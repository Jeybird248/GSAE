from __future__ import annotations

import argparse
import json
from pathlib import Path

from .activation import collect_hf_activations, collect_hf_prefix_activations
from .bank import DEFAULT_BANK, build_bank, screen_bank
from .gate import DEFAULT_GATE, train_gate
from .evaluate import collect_bank_responses, judge_bank_responses
from .runtime import GSAEArtifacts, RuntimeConfig, generate_with_gsae
from .train import DEFAULT_TRAINING, train_from_activation_dir
from .utils import deep_update, load_yaml, parse_layers, read_jsonl, save_yaml, write_jsonl


def _load_cfg(path: str | None) -> dict:
    if not path:
        return {}
    return load_yaml(path)


def _section(cfg: dict, name: str, defaults: dict) -> dict:
    return deep_update(defaults, cfg.get(name, {}))


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
    results = train_from_activation_dir(args.activations, args.graph_activations, args.output, cfg, layers, device=args.device)
    save_yaml(Path(args.output) / "training_summary.yaml", {"layers": results, "config": cfg})


def cmd_build_bank(args: argparse.Namespace) -> None:
    cfg = _section(_load_cfg(args.config), "bank", DEFAULT_BANK)
    if args.candidate_pool_size is not None:
        cfg["candidate_pool_size"] = args.candidate_pool_size
    cfg["efficacy_file"] = args.efficacy_file
    layers = parse_layers(args.layers)
    payload = build_bank(args.trained_dir, args.val_activations, args.output, layers, cfg, device=args.device)
    print(json.dumps({"selected_count": payload["selected_count"], "candidate_count": payload["candidate_count"]}))


def cmd_train_gate(args: argparse.Namespace) -> None:
    cfg = _section(_load_cfg(args.config), "gate", DEFAULT_GATE)
    layers = parse_layers(args.layers)
    payload = train_gate(args.trained_dir, args.activations, args.prefix_activations, args.output, layers, cfg, device=args.device)
    print(json.dumps({"num_examples": payload["num_examples"], "latent_dim": payload["latent_dim"]}))


def cmd_collect_prefixes(args: argparse.Namespace) -> None:
    collect_hf_prefix_activations(
        args.model, args.prompts, args.output, parse_layers(args.layers),
        max_prefix_tokens=args.max_prefixes, max_length=args.max_length,
        device=args.device, dtype=args.dtype, trust_remote_code=args.trust_remote_code,
    )


def cmd_screen_bank(args: argparse.Namespace) -> None:
    cfg = _section(_load_cfg(args.config), "bank", DEFAULT_BANK)
    screen_bank(args.trained_dir, args.output, parse_layers(args.layers), cfg, device=args.device)


def cmd_collect_bank_responses(args: argparse.Namespace) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from .activation import _dtype_from_name
    from .utils import choose_device

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=_dtype_from_name(args.dtype), trust_remote_code=args.trust_remote_code,
    ).to(choose_device(args.device)).eval()
    collect_bank_responses(model, tokenizer, args.candidates, args.prompts, args.output,
                           alpha=args.alpha, max_new_tokens=args.max_new_tokens)


def cmd_judge_bank_responses(args: argparse.Namespace) -> None:
    judge_bank_responses(args.responses, args.output)


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GSAE safety steering pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

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

    p = sub.add_parser("collect-prefixes", help="collect continuation-token states for the shared gate")
    p.add_argument("--model", required=True)
    p.add_argument("--prompts", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--layers", required=True)
    p.add_argument("--max-prefixes", type=int, default=16)
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="auto")
    p.add_argument("--trust-remote-code", action="store_true")
    p.set_defaults(func=cmd_collect_prefixes)

    p = sub.add_parser("train", help="train graph-regularized SAEs from activation shards")
    p.add_argument("--config")
    p.add_argument("--activations", required=True)
    p.add_argument("--graph-activations", required=True)
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
    p.add_argument("--efficacy-file", required=True, help="judged baseline and both-sign response JSONL")
    p.add_argument("--device", default="auto")
    p.set_defaults(func=cmd_build_bank)

    p = sub.add_parser("screen-bank", help="save per-layer candidate pools for intervention evaluation")
    p.add_argument("--config")
    p.add_argument("--trained-dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--layers", required=True)
    p.add_argument("--device", default="auto")
    p.set_defaults(func=cmd_screen_bank)

    p = sub.add_parser("collect-bank-responses", help="generate baseline and isolated candidate interventions")
    p.add_argument("--model", required=True)
    p.add_argument("--candidates", required=True)
    p.add_argument("--prompts", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--alpha", type=float, default=2.5)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="auto")
    p.add_argument("--trust-remote-code", action="store_true")
    p.set_defaults(func=cmd_collect_bank_responses)

    p = sub.add_parser("judge-bank-responses", help="label responses with the optional OpenAI bank judge")
    p.add_argument("--responses", required=True)
    p.add_argument("--output", required=True)
    p.set_defaults(func=cmd_judge_bank_responses)

    p = sub.add_parser("train-gate", help="train calibrated runtime gate")
    p.add_argument("--config")
    p.add_argument("--trained-dir", required=True)
    p.add_argument("--activations", required=True)
    p.add_argument("--prefix-activations", required=True)
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

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
