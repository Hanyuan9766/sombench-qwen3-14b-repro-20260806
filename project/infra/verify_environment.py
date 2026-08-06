#!/usr/bin/env python3
"""Fail fast if the training or official-compatible evaluation environment drifted."""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import sys
from importlib import metadata


EXPECTED = {
    "transformers": "4.51.3",
    "torch": "2.6.0",
}


def version_of(distribution: str) -> str:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return "MISSING"


def version_matches(observed: str, expected: str) -> bool:
    """Accept PEP 440 local build tags such as torch 2.6.0+cu124."""

    return observed == expected or observed.startswith(expected + "+")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("train", "eval"), required=True)
    args = parser.parse_args()

    expected = dict(EXPECTED)
    if args.mode == "train":
        expected.update(
            {
                "accelerate": "1.6.0",
                "bitsandbytes": "0.45.5",
                "peft": "0.15.2",
            }
        )
    else:
        expected["vllm"] = "0.8.5.post1"

    observed = {name: version_of(name) for name in expected}
    failures = {
        name: {"expected": wanted, "observed": observed[name]}
        for name, wanted in expected.items()
        if not version_matches(observed[name], wanted)
    }

    torch = importlib.import_module("torch")
    report = {
        "mode": args.mode,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": observed,
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda": torch.version.cuda,
        "gpu_count": torch.cuda.device_count(),
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "bf16_supported": bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))

    if failures:
        print(json.dumps({"version_failures": failures}, indent=2), file=sys.stderr)
        return 2
    if not report["cuda_available"] or report["gpu_count"] < 1:
        print("CUDA GPU is not available.", file=sys.stderr)
        return 3
    if not report["bf16_supported"]:
        print("GPU does not report bfloat16 support.", file=sys.stderr)
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
