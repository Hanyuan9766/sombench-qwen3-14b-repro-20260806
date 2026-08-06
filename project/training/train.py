"""Run 4-bit QLoRA SFT for Qwen3-14B.

Example:
    python -m training.train --config training/configs/qwen3_14b_qlora.yaml \
        --set data.train_file=/root/autodl-tmp/data/train.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Mapping

from .config import apply_overrides, deep_merge, load_config, redact_config, require_keys, resolve_path
from .data import AssistantOnlyDataCollator, ChatJsonlDataset
from .trainer import build_group_vocab, make_dimension_trainer_class

LOGGER = logging.getLogger("socialmind.train")


DEFAULTS: dict[str, Any] = {
    "model": {
        "trust_remote_code": False,
        "revision": "main",
        "init_adapter": None,
        "attn_implementation": "flash_attention_2",
        "gradient_checkpointing": True,
        "use_reentrant": False,
        "lora": {
            "r": 64,
            "alpha": 128,
            "dropout": 0.05,
            "target_modules": [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        },
    },
    "data": {
        "max_length": 4096,
        "truncation": "left",
        "pad_to_multiple_of": 8,
    },
    "training": {
        "num_train_epochs": 2.0,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": 16,
        "learning_rate": 0.0001,
        "warmup_ratio": 0.05,
        "lr_scheduler_type": "cosine",
        "weight_decay": 0.0,
        "max_grad_norm": 1.0,
        "logging_steps": 5,
        "save_steps": 50,
        "eval_steps": 50,
        "save_total_limit": 3,
        "seed": 3407,
        "data_seed": 3407,
        "bf16": True,
        "tf32": True,
        "optim": "paged_adamw_8bit",
        "report_to": ["tensorboard"],
        "resume_from_checkpoint": "auto",
    },
}


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, default=lambda item: item.__dict__)
        handle.write("\n")


def _checkpoint_to_resume(output_dir: Path, configured: Any) -> str | None:
    if configured in (None, False, "", "none"):
        return None
    if configured not in (True, "auto"):
        checkpoint = Path(str(configured)).expanduser().resolve()
        if not checkpoint.is_dir():
            raise FileNotFoundError(f"resume checkpoint does not exist: {checkpoint}")
        return str(checkpoint)
    from transformers.trainer_utils import get_last_checkpoint

    if not output_dir.is_dir():
        return None
    return get_last_checkpoint(str(output_dir))


def load_verified_source(model_cfg: Mapping[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    revision_ref = model_cfg.get("source_revision_manifest")
    verification_ref = model_cfg.get("source_verification_manifest")
    if bool(revision_ref) != bool(verification_ref):
        raise ValueError("source_revision_manifest and source_verification_manifest must be configured together")
    if not revision_ref:
        return None, None

    revision_path = resolve_path(revision_ref)
    verification_path = resolve_path(verification_ref)
    for label, path in (("source revision", revision_path), ("source verification", verification_path)):
        if not path.is_file():
            raise FileNotFoundError(f"base {label} manifest not found: {path}")
    source = json.loads(revision_path.read_text(encoding="utf-8"))
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    fingerprint = source.get("content_manifest_sha256")
    expected_files = source.get("remote_file_count")
    expected_bytes = source.get("remote_total_bytes")
    failures = []
    if verification.get("status") != "ok":
        failures.append("verification status is not ok")
    if verification.get("content_manifest_sha256") != fingerprint:
        failures.append("content manifest SHA-256 differs")
    if verification.get("checked_files") != expected_files:
        failures.append("verified file count differs")
    if verification.get("checked_bytes") != expected_bytes:
        failures.append("verified byte count differs")
    expected_local_dir = resolve_path(model_cfg["name_or_path"])
    observed_local_dir = Path(str(verification.get("local_dir", ""))).expanduser().resolve()
    if observed_local_dir != expected_local_dir:
        failures.append("verified local directory differs from model.name_or_path")
    if failures:
        raise ValueError("base snapshot provenance gate failed: " + "; ".join(failures))
    return source, verification


def load_verified_initial_adapter(
    model_cfg: Mapping[str, Any],
    source_revision: Mapping[str, Any] | None,
) -> tuple[Path | None, dict[str, Any] | None]:
    """Validate a prior training stage before continuing its LoRA weights."""

    adapter_ref = model_cfg.get("init_adapter")
    if not adapter_ref:
        return None, None
    adapter_dir = resolve_path(adapter_ref)
    if not adapter_dir.is_dir():
        raise FileNotFoundError(f"initial adapter directory not found: {adapter_dir}")
    manifest_path = adapter_dir / "training_manifest.json"
    config_path = adapter_dir / "adapter_config.json"
    for label, path in (("training manifest", manifest_path), ("adapter config", config_path)):
        if not path.is_file():
            raise FileNotFoundError(f"initial adapter {label} not found: {path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    adapter_config = json.loads(config_path.read_text(encoding="utf-8"))
    expected_fingerprint = (source_revision or {}).get("content_manifest_sha256")
    observed_fingerprint = (manifest.get("base_source_revision") or {}).get("content_manifest_sha256")
    if expected_fingerprint and observed_fingerprint != expected_fingerprint:
        raise ValueError("initial adapter base content fingerprint differs from the verified base snapshot")

    expected_lora = model_cfg["lora"]
    comparisons = {
        "r": (adapter_config.get("r"), int(expected_lora["r"])),
        "lora_alpha": (adapter_config.get("lora_alpha"), int(expected_lora["alpha"])),
        "lora_dropout": (adapter_config.get("lora_dropout"), float(expected_lora.get("dropout", 0.0))),
    }
    mismatches = [name for name, (observed, expected) in comparisons.items() if observed != expected]
    observed_targets = set(adapter_config.get("target_modules") or [])
    expected_targets = set(expected_lora["target_modules"])
    if observed_targets != expected_targets:
        mismatches.append("target_modules")
    if mismatches:
        raise ValueError("initial adapter LoRA configuration differs: " + ", ".join(sorted(mismatches)))
    return adapter_dir, {"training_manifest": manifest, "adapter_config": adapter_config}


def _training_arguments(config: Mapping[str, Any], *, has_eval: bool) -> Any:
    from transformers import TrainingArguments

    values = dict(config["training"])
    for internal in ("resume_from_checkpoint",):
        values.pop(internal, None)
    values["output_dir"] = str(resolve_path(config["output_dir"]))
    values.setdefault("remove_unused_columns", False)
    values.setdefault("label_names", ["labels"])
    values.setdefault("gradient_checkpointing", bool(config["model"].get("gradient_checkpointing", True)))
    values.setdefault(
        "gradient_checkpointing_kwargs",
        {"use_reentrant": bool(config["model"].get("use_reentrant", False))},
    )
    values.setdefault("logging_strategy", "steps")
    values.setdefault("save_strategy", "steps")
    values.setdefault("eval_strategy", "steps" if has_eval else "no")
    if not has_eval:
        values.pop("eval_steps", None)
        values["load_best_model_at_end"] = False
    else:
        values.setdefault("load_best_model_at_end", True)
        values.setdefault("metric_for_best_model", "eval_loss")
        values.setdefault("greater_is_better", False)
    return TrainingArguments(**values)


def run(config: Mapping[str, Any]) -> None:
    require_keys(config, ["model.name_or_path", "data.train_file", "output_dir"])
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed

    if not torch.cuda.is_available():
        raise RuntimeError("4-bit QLoRA requires a CUDA GPU")
    if bool(config["training"].get("bf16", True)) and not torch.cuda.is_bf16_supported():
        raise RuntimeError("configured bf16=true, but this GPU/runtime does not report BF16 support")

    output_dir = resolve_path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    _json_dump(output_dir / "resolved_config.json", redact_config(config))
    set_seed(int(config["training"].get("seed", 3407)))

    model_cfg = config["model"]
    source_revision, source_verification = load_verified_source(model_cfg)
    initial_adapter_dir, initial_adapter_provenance = load_verified_initial_adapter(
        model_cfg,
        source_revision,
    )
    if source_revision is not None:
        _json_dump(output_dir / "base_source_revision.json", source_revision)
        _json_dump(output_dir / "base_source_verification.json", source_verification)
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["name_or_path"],
        revision=model_cfg.get("revision", "main"),
        trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError("tokenizer has neither pad_token_id nor eos_token_id")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    data_cfg = config["data"]
    train_dataset = ChatJsonlDataset(
        resolve_path(data_cfg["train_file"]),
        tokenizer,
        max_length=int(data_cfg["max_length"]),
        truncation=str(data_cfg.get("truncation", "left")),
    )
    eval_file = data_cfg.get("eval_file")
    eval_dataset = (
        ChatJsonlDataset(
            resolve_path(eval_file),
            tokenizer,
            max_length=int(data_cfg["max_length"]),
            truncation=str(data_cfg.get("truncation", "left")),
        )
        if eval_file
        else None
    )
    _json_dump(
        output_dir / "dataset_stats.json",
        {"train": train_dataset.stats, "eval": eval_dataset.stats if eval_dataset else None},
    )

    compute_dtype = torch.bfloat16 if bool(config["training"].get("bf16", True)) else torch.float16
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    model_kwargs: dict[str, Any] = {
        "revision": model_cfg.get("revision", "main"),
        "trust_remote_code": bool(model_cfg.get("trust_remote_code", False)),
        "quantization_config": quantization_config,
        "torch_dtype": compute_dtype,
        "low_cpu_mem_usage": True,
        "device_map": {"": local_rank},
    }
    attn_implementation = model_cfg.get("attn_implementation")
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation
    model = AutoModelForCausalLM.from_pretrained(model_cfg["name_or_path"], **model_kwargs)
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=bool(model_cfg.get("gradient_checkpointing", True)),
        gradient_checkpointing_kwargs={"use_reentrant": bool(model_cfg.get("use_reentrant", False))},
    )
    lora_cfg = model_cfg["lora"]
    if initial_adapter_dir is not None:
        model = PeftModel.from_pretrained(model, str(initial_adapter_dir), is_trainable=True)
    else:
        model = get_peft_model(
            model,
            LoraConfig(
                r=int(lora_cfg["r"]),
                lora_alpha=int(lora_cfg["alpha"]),
                lora_dropout=float(lora_cfg.get("dropout", 0.0)),
                target_modules=list(lora_cfg["target_modules"]),
                bias="none",
                task_type="CAUSAL_LM",
            ),
        )
    model.print_trainable_parameters()

    training_args = _training_arguments(config, has_eval=eval_dataset is not None)
    qtypes, dimensions = build_group_vocab(train_dataset, eval_dataset)
    TrainerClass = make_dimension_trainer_class()
    trainer = TrainerClass(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=AssistantOnlyDataCollator(
            tokenizer,
            pad_to_multiple_of=int(data_cfg.get("pad_to_multiple_of", 8)),
        ),
        processing_class=tokenizer,
        group_qtypes=qtypes,
        group_dimensions=dimensions,
    )
    resume = _checkpoint_to_resume(output_dir, config["training"].get("resume_from_checkpoint", "auto"))
    LOGGER.info("starting training; resume_from_checkpoint=%s", resume)
    train_result = trainer.train(resume_from_checkpoint=resume)
    trainer.save_state()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    if eval_dataset is not None:
        eval_metrics = trainer.evaluate()
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)

    adapter_dir = output_dir / "final_adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(adapter_dir)
    _json_dump(
        adapter_dir / "training_manifest.json",
        {
            "base_model": model_cfg["name_or_path"],
            "base_source_revision": source_revision,
            "base_source_verification": source_verification,
            "initial_adapter": (
                {
                    "path": str(initial_adapter_dir),
                    "provenance": initial_adapter_provenance,
                }
                if initial_adapter_dir is not None
                else None
            ),
            "config": redact_config(config),
        },
    )
    LOGGER.info("adapter saved to %s; merge it with `python -m training.merge`", adapter_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="JSON/YAML training configuration")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="dotted config override")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config = deep_merge(DEFAULTS, load_config(args.config))
    config = apply_overrides(config, args.set)
    run(config)


if __name__ == "__main__":
    main()
