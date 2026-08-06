#!/usr/bin/env python3
"""Validate a full, offline-loadable Hugging Face causal-LM repository."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


def fail(message: str) -> None:
    raise ValueError(message)


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - context is useful on remote hosts
        fail(f"cannot parse {path.name}: {exc}")


def discover_weight_files(repo: Path) -> tuple[list[Path], int | None]:
    index_files = sorted(repo.glob("*.safetensors.index.json"))
    if index_files:
        if len(index_files) != 1:
            fail(f"expected one safetensors index, found {len(index_files)}")
        index = load_json(index_files[0])
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            fail("safetensors index has no non-empty weight_map")
        names = sorted(set(weight_map.values()))
        files = [repo / name for name in names]
        missing = [path.name for path in files if not path.is_file()]
        if missing:
            fail(f"weight index references missing files: {missing[:5]}")
        total_size = index.get("metadata", {}).get("total_size")
        return files, total_size if isinstance(total_size, int) else None

    files = sorted(repo.glob("*.safetensors"))
    if not files:
        fail("no full safetensors weights found")
    return files, None


def count_safetensor_parameters(files: list[Path]) -> tuple[int, dict[str, int]]:
    from safetensors import safe_open

    total = 0
    dtype_counts: dict[str, int] = {}
    tensor_names: set[str] = set()
    for path in files:
        with safe_open(path, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                if name in tensor_names:
                    fail(f"duplicate tensor name across shards: {name}")
                tensor_names.add(name)
                tensor = handle.get_slice(name)
                shape = tensor.get_shape()
                count = math.prod(shape)
                dtype = str(tensor.get_dtype())
                total += count
                dtype_counts[dtype] = dtype_counts.get(dtype, 0) + count
    return total, dtype_counts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("repo", type=Path)
    parser.add_argument("--expected-model-type", default="qwen3")
    parser.add_argument("--min-parameters", type=int, default=13_000_000_000)
    parser.add_argument("--max-parameters", type=int, default=20_000_000_000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    repo = args.repo.resolve()
    if not repo.is_dir():
        fail(f"repository directory does not exist: {repo}")

    required = ["config.json", "tokenizer_config.json"]
    missing = [name for name in required if not (repo / name).is_file()]
    if missing:
        fail(f"missing required repository files: {missing}")
    if not ((repo / "tokenizer.json").is_file() or (repo / "vocab.json").is_file()):
        fail("missing tokenizer.json or vocab.json")

    config_data = load_json(repo / "config.json")
    if config_data.get("model_type") != args.expected_model_type:
        fail(
            f"unexpected model_type={config_data.get('model_type')!r}; "
            f"expected {args.expected_model_type!r}"
        )

    files, declared_bytes = discover_weight_files(repo)
    parameter_count, dtype_counts = count_safetensor_parameters(files)
    if not args.min_parameters <= parameter_count <= args.max_parameters:
        fail(
            f"tensor parameter count {parameter_count:,} is outside "
            f"[{args.min_parameters:,}, {args.max_parameters:,}]"
        )

    from transformers import AutoConfig, AutoTokenizer

    config = AutoConfig.from_pretrained(repo, local_files_only=True, trust_remote_code=False)
    tokenizer = AutoTokenizer.from_pretrained(
        repo, local_files_only=True, trust_remote_code=False, use_fast=True
    )
    rendered = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "你是一个严谨的社会认知推理助手。"},
            {"role": "user", "content": "测试题。最终仅给出\\boxed{A}。"},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    if not isinstance(rendered, str) or not rendered.strip():
        fail("tokenizer chat template produced an empty prompt")

    tokenizer_ids = {
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    }
    config_ids = {
        "bos_token_id": getattr(config, "bos_token_id", None),
        "eos_token_id": getattr(config, "eos_token_id", None),
        "pad_token_id": getattr(config, "pad_token_id", None),
    }
    report = {
        "repo": str(repo),
        "model_type": config.model_type,
        "architectures": getattr(config, "architectures", None),
        "weight_shards": len(files),
        "weight_bytes_on_disk": sum(path.stat().st_size for path in files),
        "declared_weight_bytes": declared_bytes,
        "tensor_parameters": parameter_count,
        "tensor_parameter_dtypes": dtype_counts,
        "tokenizer_class": tokenizer.__class__.__name__,
        "tokenizer_ids": tokenizer_ids,
        "config_ids": config_ids,
        "chat_template_chars": len(rendered),
        "adapter_only": (repo / "adapter_config.json").is_file() and len(files) == 0,
        "status": "ok",
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    print(rendered, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"VALIDATION_FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
