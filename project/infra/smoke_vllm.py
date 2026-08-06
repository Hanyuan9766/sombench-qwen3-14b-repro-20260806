#!/usr/bin/env python3
"""Probe the OpenAI-compatible vLLM server with the official decode settings."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from urllib import error, request


BOX_RE = re.compile(r"\\boxed\{\s*([A-Z](?:\s*,\s*[A-Z])*)\s*\}")


def positive_int(value: str) -> int:
    """Parse a strictly positive integer for generation limits."""
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="sombench-qwen3-14b")
    parser.add_argument("--wait-seconds", type=int, default=600)
    parser.add_argument(
        "--max-tokens",
        type=positive_int,
        default=256,
        help="maximum completion tokens for this smoke probe (default: 256)",
    )
    return parser


def build_payload(model: str, max_tokens: int) -> dict:
    return {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "请分析社会情境题。先给出简洁推理，最后一行仅用 "
                    "\\boxed{X} 输出选项字母。"
                ),
            },
            {
                "role": "user",
                "content": "小明看到盒中有苹果。问：盒中有什么？ A. 香蕉 B. 苹果 C. 梨",
            },
        ],
        "temperature": 0.6,
        "top_p": 0.95,
        "max_tokens": max_tokens,
        "n": 1,
        "seed": 20260805,
    }


def post_json(url: str, payload: dict, timeout: float) -> dict:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> int:
    args = build_parser().parse_args()

    models_url = args.base_url.rstrip("/") + "/models"
    deadline = time.monotonic() + args.wait_seconds
    while True:
        try:
            with request.urlopen(models_url, timeout=10) as response:
                if response.status == 200:
                    break
        except (error.URLError, TimeoutError):
            if time.monotonic() >= deadline:
                print("vLLM readiness timed out", file=sys.stderr)
                return 2
            time.sleep(5)

    payload = build_payload(args.model, args.max_tokens)
    result = post_json(args.base_url.rstrip("/") + "/chat/completions", payload, 180)
    text = result["choices"][0]["message"]["content"]
    print(json.dumps({"response": text, "raw": result}, ensure_ascii=False, indent=2))
    matches = BOX_RE.findall(text)
    if not matches or matches[-1].replace(" ", "") != "B":
        print("smoke response does not end in the expected boxed answer", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
