"""Merge a QLoRA adapter into an unquantized BF16 ModelScope-ready model."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Mapping

from .config import apply_overrides, deep_merge, load_config, redact_config, require_keys, resolve_path
from .repo_check import inspect_repository

LOGGER = logging.getLogger("socialmind.merge")

DEFAULTS: dict[str, Any] = {
    "trust_remote_code": False,
    "device_map": "auto",
    "max_shard_size": "5GB",
    "safe_merge": True,
    "overwrite_output_dir": False,
    "expected_model_type": "qwen3",
    "min_weight_gb": 20.0,
    "require_source_verification": False,
    "expected_base_repo": None,
    "expected_base_revision": None,
}


def _write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _assert_output_is_safe(output_dir: Path, overwrite: bool) -> None:
    if not output_dir.exists():
        return
    existing = list(output_dir.iterdir())
    if existing and not overwrite:
        raise FileExistsError(
            f"merge output is not empty: {output_dir}; choose a new directory or explicitly set overwrite_output_dir=true"
        )
    if existing and overwrite:
        # save_pretrained overwrites matching files but never silently removes
        # unrelated user files.  Reject a different existing model instead.
        config_path = output_dir / "config.json"
        if config_path.exists():
            raise FileExistsError(
                f"refusing to overwrite an existing model in {output_dir}; use a new versioned output directory"
            )


def verify_base_source(config: Mapping[str, Any]) -> dict[str, Any] | None:
    """Revalidate the recorded base snapshot immediately before merging.

    The prior verification report proves what was downloaded at that earlier
    point in time.  Re-running the file verifier here also proves that the
    local snapshot has not changed between training and merge.
    """

    revision_ref = config.get("source_revision_manifest")
    verification_ref = config.get("source_verification_manifest")
    required = bool(config.get("require_source_verification", False))
    if bool(revision_ref) != bool(verification_ref):
        raise ValueError(
            "source_revision_manifest and source_verification_manifest must be configured together"
        )
    if not revision_ref:
        if required:
            raise ValueError("formal merge requires source revision and verification manifests")
        return None

    base_model = resolve_path(config["base_model"])
    revision_path = resolve_path(revision_ref)
    verification_path = resolve_path(verification_ref)
    for label, path in (("source revision", revision_path), ("source verification", verification_path)):
        if not path.is_file():
            raise FileNotFoundError(f"base {label} manifest not found: {path}")
    try:
        source = json.loads(revision_path.read_text(encoding="utf-8"))
        recorded = json.loads(verification_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid base provenance JSON: {exc}") from exc
    if not isinstance(source, dict) or not isinstance(recorded, dict):
        raise ValueError("base source and verification manifests must contain JSON objects")

    expected_repo = config.get("expected_base_repo")
    expected_revision = config.get("expected_base_revision")
    failures: list[str] = []
    if expected_repo is not None and source.get("repo_id") != expected_repo:
        failures.append(f"source repo_id differs: {source.get('repo_id')!r} != {expected_repo!r}")
    if expected_revision is not None and source.get("requested_revision") != expected_revision:
        failures.append(
            "source requested_revision differs: "
            f"{source.get('requested_revision')!r} != {expected_revision!r}"
        )

    remote_files = source.get("remote_files")
    if not isinstance(remote_files, list) or not remote_files:
        failures.append("source manifest has no non-empty remote_files list")
        remote_files = []
    declared_count = source.get("remote_file_count")
    declared_bytes = source.get("remote_total_bytes")
    listed_bytes = sum(
        item.get("size", 0)
        for item in remote_files
        if isinstance(item, Mapping) and isinstance(item.get("size"), int)
    )
    if declared_count != len(remote_files):
        failures.append("source remote_file_count differs from remote_files")
    if declared_bytes != listed_bytes:
        failures.append("source remote_total_bytes differs from remote_files")

    fingerprint = source.get("content_manifest_sha256")
    checks = {
        "verification status is not ok": recorded.get("status") == "ok",
        "verification reports failures": not recorded.get("failures"),
        "recorded content fingerprint differs": recorded.get("content_manifest_sha256") == fingerprint,
        "recorded file count differs": recorded.get("checked_files") == declared_count,
        "recorded byte count differs": recorded.get("checked_bytes") == declared_bytes,
        "recorded local directory differs": (
            Path(str(recorded.get("local_dir", ""))).expanduser().resolve() == base_model
        ),
    }
    failures.extend(message for message, passed in checks.items() if not passed)
    if failures:
        raise ValueError("base snapshot provenance gate failed: " + "; ".join(failures))

    # Imported lazily so the merge CLI remains importable in lightweight test
    # environments.  This recomputes the manifest fingerprint and every local
    # file SHA-256 instead of trusting the earlier report.
    from infra.verify_source_snapshot import verify_snapshot

    current = verify_snapshot(revision_path, base_model)
    if (
        current.get("content_manifest_sha256") != fingerprint
        or current.get("checked_files") != declared_count
        or current.get("checked_bytes") != declared_bytes
        or current.get("status") != "ok"
    ):
        raise ValueError("merge-time base snapshot verification differs from source manifest")
    expected_paths = {
        str(item["path"])
        for item in remote_files
        if isinstance(item, Mapping) and isinstance(item.get("path"), str)
    }
    unexpected = sorted(
        str(path.relative_to(base_model))
        for path in base_model.rglob("*")
        if path.is_file()
        and str(path.relative_to(base_model)) not in expected_paths
        and not any(part.startswith(".") for part in path.relative_to(base_model).parts)
    )
    if unexpected:
        raise ValueError(
            "base snapshot contains files outside the source manifest: "
            + ", ".join(unexpected[:20])
        )
    return {
        "source_revision": source,
        "recorded_verification": recorded,
        "merge_time_verification": current,
    }


def run(config: Mapping[str, Any]) -> None:
    require_keys(config, ["base_model", "adapter_path", "output_dir"])
    base_model = resolve_path(config["base_model"])
    adapter_path = resolve_path(config["adapter_path"])
    output_dir = resolve_path(config["output_dir"])
    if not base_model.is_dir():
        raise FileNotFoundError(f"base model directory not found: {base_model}")
    if not adapter_path.is_dir():
        raise FileNotFoundError(f"adapter directory not found: {adapter_path}")
    if not (adapter_path / "adapter_config.json").is_file():
        raise FileNotFoundError(f"adapter_config.json not found in {adapter_path}")
    _assert_output_is_safe(output_dir, bool(config.get("overwrite_output_dir", False)))
    base_provenance = verify_base_source(config)
    output_dir.mkdir(parents=True, exist_ok=True)

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    LOGGER.info("loading unquantized base model in BF16: %s", base_model)
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map=config.get("device_map", "auto"),
        low_cpu_mem_usage=True,
        trust_remote_code=bool(config.get("trust_remote_code", False)),
    )
    peft_model = PeftModel.from_pretrained(base, str(adapter_path), is_trainable=False)
    merged = peft_model.merge_and_unload(safe_merge=bool(config.get("safe_merge", True)))
    if any("lora_" in name or "base_model.model" in name for name, _ in merged.named_parameters()):
        raise RuntimeError("PEFT parameter names remain after merge; refusing to write an incomplete repository")
    merged.config.torch_dtype = torch.bfloat16
    if hasattr(merged.config, "quantization_config"):
        delattr(merged.config, "quantization_config")
    merged.config.use_cache = True
    merged.save_pretrained(
        output_dir,
        safe_serialization=True,
        max_shard_size=str(config.get("max_shard_size", "5GB")),
    )

    tokenizer_source = adapter_path if (adapter_path / "tokenizer_config.json").is_file() else base_model
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        trust_remote_code=bool(config.get("trust_remote_code", False)),
        use_fast=True,
    )
    tokenizer.save_pretrained(output_dir)
    adapter_training_manifest = None
    adapter_manifest_path = adapter_path / "training_manifest.json"
    if adapter_manifest_path.is_file():
        with adapter_manifest_path.open("r", encoding="utf-8") as handle:
            adapter_training_manifest = json.load(handle)
    _write_json(
        output_dir / "configuration.json",
        {
            "framework": "pytorch",
            "task": "text-generation",
            "model": {"type": "qwen3"},
        },
    )
    _write_json(
        output_dir / "merge_manifest.json",
        {
            "base_model": str(base_model),
            "base_provenance": base_provenance,
            "adapter_path": str(adapter_path),
            "adapter_training_manifest": adapter_training_manifest,
            "dtype": "bfloat16",
            "safe_serialization": True,
            "config": redact_config(config),
        },
    )
    report = inspect_repository(
        output_dir,
        expected_model_type=str(config.get("expected_model_type", "qwen3")),
        min_weight_bytes=int(float(config.get("min_weight_gb", 20.0)) * 1_000_000_000),
    )
    _write_json(output_dir / "repository_check.json", report.__dict__)
    if not report.ok:
        raise RuntimeError("merged repository failed integrity check:\n- " + "\n- ".join(report.errors))
    LOGGER.info("complete BF16 repository passed integrity checks: %s", output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config = deep_merge(DEFAULTS, load_config(args.config))
    config = apply_overrides(config, args.set)
    run(config)


if __name__ == "__main__":
    main()
