"""Validate that a directory is a complete BF16 Hugging Face/ModelScope repo."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RepositoryReport:
    path: str
    ok: bool
    errors: list[str]
    warnings: list[str]
    weight_files: list[str]
    total_weight_bytes: int
    model_type: str | None
    dtype: str | None


def _read_object(path: Path, errors: list[str]) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"cannot read {path.name}: {exc}")
        return {}
    if not isinstance(value, dict):
        errors.append(f"{path.name} must contain a JSON object")
        return {}
    return value


def _weight_files(root: Path, config: dict[str, Any], errors: list[str]) -> list[Path]:
    index_path = root / "model.safetensors.index.json"
    single_path = root / "model.safetensors"
    if index_path.exists():
        index = _read_object(index_path, errors)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            errors.append("model.safetensors.index.json has no non-empty weight_map")
            return []
        names = sorted({str(value) for value in weight_map.values()})
        files = [root / name for name in names]
        for file in files:
            if file.parent.resolve() != root.resolve():
                errors.append(f"weight index references path outside repository: {file}")
            elif not file.is_file():
                errors.append(f"weight shard referenced by index is missing: {file.name}")
        metadata = index.get("metadata")
        if isinstance(metadata, dict) and isinstance(metadata.get("total_size"), int):
            actual = sum(file.stat().st_size for file in files if file.is_file())
            if actual < int(metadata["total_size"]):
                errors.append(
                    f"weight shards contain {actual} bytes, below index metadata total_size={metadata['total_size']}"
                )
        return files
    if single_path.is_file():
        return [single_path]
    legacy = sorted(root.glob("pytorch_model*.bin"))
    if legacy:
        errors.append("legacy .bin weights found; submission must use safe_serialization (.safetensors)")
    else:
        errors.append("no model.safetensors or model.safetensors.index.json found")
    return []


def inspect_repository(
    path: str | Path,
    *,
    expected_model_type: str | None = "qwen3",
    min_weight_bytes: int = 20_000_000_000,
) -> RepositoryReport:
    """Perform dependency-free structural checks on a merged model directory."""

    root = Path(path).expanduser().resolve()
    errors: list[str] = []
    warnings: list[str] = []
    if not root.is_dir():
        return RepositoryReport(str(root), False, ["repository directory does not exist"], [], [], 0, None, None)

    required = ["config.json", "tokenizer_config.json"]
    for name in required:
        if not (root / name).is_file():
            errors.append(f"missing required file: {name}")
    if not (root / "tokenizer.json").is_file():
        errors.append("missing tokenizer.json (fast tokenizer)")
    if not (root / "generation_config.json").is_file():
        warnings.append("generation_config.json is absent; vLLM can load but generation defaults are not documented")
    if not (root / "configuration.json").is_file():
        warnings.append("configuration.json is absent; add ModelScope metadata before upload")

    forbidden = sorted(
        file.name
        for pattern in ("adapter_config.json", "adapter_model*", "*4bit*", "*gptq*", "*awq*")
        for file in root.glob(pattern)
    )
    if forbidden:
        errors.append("adapter/quantized artifacts present in full-model repo: " + ", ".join(sorted(set(forbidden))))

    config = _read_object(root / "config.json", errors) if (root / "config.json").is_file() else {}
    model_type = config.get("model_type") if isinstance(config.get("model_type"), str) else None
    dtype_value = config.get("torch_dtype", config.get("dtype"))
    dtype = str(dtype_value).lower() if dtype_value is not None else None
    if expected_model_type and model_type != expected_model_type:
        errors.append(f"config.model_type={model_type!r}, expected {expected_model_type!r}")
    if dtype not in {"bfloat16", "bf16", "torch.bfloat16"}:
        errors.append(f"config dtype must be bfloat16, got {dtype_value!r}")
    if config.get("quantization_config") is not None or config.get("load_in_4bit"):
        errors.append("config.json still declares quantization; merge must produce unquantized BF16 weights")
    architectures = config.get("architectures")
    if not isinstance(architectures, list) or not architectures:
        errors.append("config.architectures is missing or empty")

    tokenizer_cfg = (
        _read_object(root / "tokenizer_config.json", errors) if (root / "tokenizer_config.json").is_file() else {}
    )
    if not tokenizer_cfg.get("chat_template"):
        errors.append("tokenizer_config.json has no chat_template")

    weight_files = _weight_files(root, config, errors)
    total_weight_bytes = sum(file.stat().st_size for file in weight_files if file.is_file())
    empty = [file.name for file in weight_files if file.is_file() and file.stat().st_size == 0]
    if empty:
        errors.append("zero-byte weight shards: " + ", ".join(empty))
    if total_weight_bytes < min_weight_bytes:
        errors.append(
            f"total safetensors size is {total_weight_bytes:,} bytes, below required {min_weight_bytes:,}; "
            "this is probably an adapter or incomplete upload"
        )

    return RepositoryReport(
        path=str(root),
        ok=not errors,
        errors=errors,
        warnings=warnings,
        weight_files=[file.name for file in weight_files],
        total_weight_bytes=total_weight_bytes,
        model_type=model_type,
        dtype=dtype,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo")
    parser.add_argument("--expected-model-type", default="qwen3")
    parser.add_argument("--min-weight-gb", type=float, default=20.0)
    parser.add_argument("--output", help="optional JSON report path")
    args = parser.parse_args()
    report = inspect_repository(
        args.repo,
        expected_model_type=args.expected_model_type or None,
        min_weight_bytes=int(args.min_weight_gb * 1_000_000_000),
    )
    rendered = json.dumps(asdict(report), ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    raise SystemExit(0 if report.ok else 1)


if __name__ == "__main__":
    main()
