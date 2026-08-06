"""Command-line entry point for the SoMBench data pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from .core import PipelineConfig, run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit and prepare SoMBench JSONL for grouped/balanced SFT."
    )
    parser.add_argument("--train", required=True, type=Path, help="Reference training JSONL")
    parser.add_argument(
        "--public-test", required=True, type=Path, help="Public labeled test JSONL"
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--fuzzy-story-threshold", type=float, default=0.92)
    parser.add_argument("--shingle-size", type=int, default=5)
    parser.add_argument("--sketch-size", type=int, default=12)
    parser.add_argument("--max-cot-chars", type=int, default=1800)
    parser.add_argument(
        "--cot-style", choices=("compressed", "answer_only"), default="compressed"
    )
    parser.add_argument(
        "--balance-target",
        default="median",
        help="Per-dimension target: median (default), min, max, none, or integer",
    )
    parser.add_argument(
        "--safe-dev-policy",
        choices=("agreement_only", "mapped_answer_text"),
        default="agreement_only",
        help="Whether conflicts are excluded or corrected from exact answer text",
    )
    parser.add_argument(
        "--safe-dev-require-complete-story",
        action="store_true",
        help="Keep a public story only when every question in its bundle is safe",
    )
    parser.add_argument("--allow-pred-mismatch", action="store_true")
    parser.add_argument(
        "--force", action="store_true", help="Atomically replace known output files"
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    config = PipelineConfig(
        seed=args.seed,
        val_fraction=args.val_fraction,
        fuzzy_story_threshold=args.fuzzy_story_threshold,
        shingle_size=args.shingle_size,
        sketch_size=args.sketch_size,
        max_cot_chars=args.max_cot_chars,
        cot_style=args.cot_style,
        balance_target=args.balance_target,
        safe_dev_policy=args.safe_dev_policy,
        safe_dev_require_complete_story=args.safe_dev_require_complete_story,
        allow_pred_mismatch=args.allow_pred_mismatch,
    )
    audit = run_pipeline(
        args.train,
        args.public_test,
        args.output_dir,
        config,
        force=args.force,
    )
    summary = {
        "output_dir": str(args.output_dir.resolve()),
        "training": {
            "accepted": audit["training"]["accepted_rows"],
            "rejected": audit["training"]["rejected_rows"],
            "dimensions": audit["training"]["dimensions_after"]["count"],
        },
        "split": {
            "train": audit["split"]["train_rows"],
            "dev": audit["split"]["dev_rows"],
        },
        "balanced_train": audit["balancing"]["rows_after"],
        "public_safe_dev": audit["public_test"]["safe_dev_rows"],
        "public_manual_review": audit["public_test"]["manual_review_rows"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

