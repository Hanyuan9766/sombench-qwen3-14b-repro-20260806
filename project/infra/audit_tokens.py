#!/usr/bin/env python3
"""Audit processed chat JSONL with the exact Qwen3 chat template before training."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def percentile(values: Sequence[int], fraction: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-overlength", action="store_true")
    args = parser.parse_args()

    from transformers import AutoTokenizer
    from training.data import assistant_only_labels, iter_jsonl, record_metadata, validate_messages

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=True,
        trust_remote_code=False,
        use_fast=True,
    )
    files = []
    global_overlength = []
    for input_path in args.input:
        token_lengths: list[int] = []
        trainable_lengths: list[int] = []
        qtypes: Counter[str] = Counter()
        dimensions: Counter[str] = Counter()
        overlength = []
        format_violations = []
        trainable_previews = []
        for row_number, record in enumerate(iter_jsonl(input_path), start=1):
            messages = validate_messages(record)
            input_ids, labels = assistant_only_labels(tokenizer, messages)
            trainable = sum(label != -100 for label in labels)
            sample_id, qtype, dimension = record_metadata(record, row_number)
            token_lengths.append(len(input_ids))
            trainable_lengths.append(trainable)
            qtypes[qtype] += 1
            dimensions[dimension] += 1
            trainable_ids = [token for token, label in zip(input_ids, labels) if label != -100]
            trainable_text = tokenizer.decode(trainable_ids, skip_special_tokens=False)
            missing_markers = [
                marker for marker in ("</think>", "\\boxed{") if marker not in trainable_text
            ]
            if missing_markers:
                format_violations.append(
                    {"sample_id": sample_id, "missing_trainable_markers": missing_markers}
                )
            if len(trainable_previews) < 3:
                trainable_previews.append(
                    {
                        "sample_id": sample_id,
                        "prefix": trainable_text[:160],
                        "suffix": trainable_text[-240:],
                    }
                )
            if len(input_ids) > args.max_length:
                item = {
                    "sample_id": sample_id,
                    "tokens": len(input_ids),
                    "trainable_tokens": trainable,
                    "qtype": qtype,
                    "dimension": dimension,
                }
                overlength.append(item)
                global_overlength.append({"file": str(input_path), **item})
        files.append(
            {
                "path": str(input_path.resolve()),
                "rows": len(token_lengths),
                "tokens": {
                    "min": min(token_lengths) if token_lengths else None,
                    "p50": percentile(token_lengths, 0.50),
                    "p95": percentile(token_lengths, 0.95),
                    "p99": percentile(token_lengths, 0.99),
                    "max": max(token_lengths) if token_lengths else None,
                },
                "trainable_tokens": {
                    "min": min(trainable_lengths) if trainable_lengths else None,
                    "p50": percentile(trainable_lengths, 0.50),
                    "p95": percentile(trainable_lengths, 0.95),
                    "p99": percentile(trainable_lengths, 0.99),
                    "max": max(trainable_lengths) if trainable_lengths else None,
                },
                "qtypes": dict(sorted(qtypes.items())),
                "dimension_count": len(dimensions),
                "overlength_count": len(overlength),
                "overlength": sorted(overlength, key=lambda item: item["tokens"], reverse=True),
                "format_violation_count": len(format_violations),
                "format_violations": format_violations,
                "trainable_previews": trainable_previews,
            }
        )

    report = {
        "model": str(Path(args.model).resolve()),
        "tokenizer_class": tokenizer.__class__.__name__,
        "max_length": args.max_length,
        "files": files,
        "total_overlength": len(global_overlength),
        "total_format_violations": sum(item["format_violation_count"] for item in files),
        "status": (
            "ok"
            if not global_overlength and not any(item["format_violation_count"] for item in files)
            else "failed"
        ),
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    print(rendered, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    if global_overlength and not args.allow_overlength:
        print(
            f"ERROR: {len(global_overlength)} examples exceed max_length={args.max_length}",
            file=sys.stderr,
        )
        return 2
    if any(item["format_violation_count"] for item in files):
        print("ERROR: trainable assistant spans are missing required markers", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
