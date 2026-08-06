#!/usr/bin/env python3
"""Download a ModelScope revision to a clean folder and verify offline loading."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default=os.environ.get("MODELSCOPE_MODEL_ID"))
    parser.add_argument("--revision", required=True)
    parser.add_argument("--validator", type=Path, required=True)
    parser.add_argument("--keep-dir", type=Path)
    parser.add_argument(
        "--temp-root",
        type=Path,
        default=Path(os.environ.get("SOMBENCH_TMP_DIR", "/root/autodl-tmp/sombench/tmp")),
    )
    args = parser.parse_args()
    if not args.repo_id:
        parser.error("--repo-id or MODELSCOPE_MODEL_ID is required")
    if args.revision in {"main", "latest"}:
        parser.error("use master, a tag, or an immutable commit hash")

    token = os.environ.get("MODELSCOPE_API_TOKEN")
    from modelscope_hub import HubApi

    api = HubApi(token=token) if token else HubApi()
    if args.keep_dir:
        target = args.keep_dir.resolve()
        target.mkdir(parents=True, exist_ok=True)
        api.download_repo(
            args.repo_id,
            "model",
            revision=args.revision,
            local_dir=str(target),
        )
        return subprocess.call([sys.executable, str(args.validator), str(target)])

    args.temp_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="sombench-ms-verify-", dir=args.temp_root) as temp_dir:
        api.download_repo(
            args.repo_id,
            "model",
            revision=args.revision,
            local_dir=temp_dir,
        )
        return subprocess.call([sys.executable, str(args.validator), temp_dir])


if __name__ == "__main__":
    raise SystemExit(main())
