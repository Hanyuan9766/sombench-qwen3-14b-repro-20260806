#!/usr/bin/env python3
"""Build a clean, uploadable ModelScope/Hugging Face repository via hardlinks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ALLOWED_EXACT = {
    "LICENSE",
    "NOTICE",
    "added_tokens.json",
    "chat_template.jinja",
    "config.json",
    "configuration.json",
    "generation_config.json",
    "merges.txt",
    "model.safetensors.index.json",
    "preprocessor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "vocab.json",
    "merge_manifest.json",
    "repository_check.json",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def link_or_copy(source: Path, target: Path) -> None:
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def include_file(path: Path) -> bool:
    name = path.name
    return (
        name in ALLOWED_EXACT
        or name.endswith(".safetensors")
        or name.endswith(".safetensors.index.json")
        or name.startswith("tokenizer")
    )


def make_model_card(
    base_repo: str,
    base_revision: str,
    base_content_fingerprint: str,
    base_source_manifest_sha256: str,
    training_summary: dict,
    repo_id: str,
    release_revision: str,
) -> str:
    summary_json = json.dumps(training_summary, ensure_ascii=False, indent=2)
    return f"""---
license: apache-2.0
frameworks:
- PyTorch
tasks:
- chat
model-type:
- qwen3
---

# SoMBench Qwen3-14B

这是基于 `{base_repo}` 的完整合并权重仓库，用于 SoMBench / SocialMind 社会认知选择题评测。

## 基座与兼容性

- Base model: `{base_repo}`
- Base requested revision: `{base_revision}`
- Base content-manifest SHA-256: `{base_content_fingerprint}`
- Base source-manifest SHA-256: `{base_source_manifest_sha256}`
- Model format: Hugging Face safetensors（完整权重，不是独立 Adapter）
- Verified Transformers: `4.51.3`
- Verified vLLM: `0.8.5.post1`
- Recommended dtype: `bfloat16`

## 评测解码参数

```text
temperature = 0.6
top_p = 0.95
max_tokens = 8192
n = 1
```

输出约定：先完成推理，最后一行以 `\\boxed{{A}}` 或 `\\boxed{{A,C}}` 给出答案。

## 训练摘要

```json
{summary_json}
```

## 加载

```python
from modelscope_hub import HubApi
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = "{repo_id}"
local_repo = HubApi().download_repo(repo, "model", revision="{release_revision}")
tokenizer = AutoTokenizer.from_pretrained(local_repo)
model = AutoModelForCausalLM.from_pretrained(
    local_repo,
    torch_dtype="auto",
    device_map="auto",
)
```
"""


def validate_base_source_manifest(
    value: object,
    *,
    expected_repo_id: str,
    expected_revision: str,
) -> tuple[dict, str]:
    """Bind the immutable content fingerprint to the requested ModelScope source."""

    if not isinstance(value, dict):
        raise ValueError("source revision manifest must contain a JSON object")
    mismatches = []
    if value.get("repo_id") != expected_repo_id:
        mismatches.append(
            f"repo_id={value.get('repo_id')!r}, expected {expected_repo_id!r}"
        )
    if value.get("requested_revision") != expected_revision:
        mismatches.append(
            "requested_revision="
            f"{value.get('requested_revision')!r}, expected {expected_revision!r}"
        )
    if mismatches:
        raise ValueError("source revision manifest identity mismatch: " + "; ".join(mismatches))

    fingerprint = value.get("content_manifest_sha256")
    if (
        not isinstance(fingerprint, str)
        or len(fingerprint) != 64
        or any(character not in "0123456789abcdef" for character in fingerprint)
    ):
        raise ValueError("source revision manifest has no valid content_manifest_sha256")
    return value, fingerprint


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="merged Hugging Face model directory")
    parser.add_argument("destination", type=Path, help="new clean release directory")
    parser.add_argument("--validator", type=Path, required=True)
    parser.add_argument("--base-repo", default="Qwen/Qwen3-14B")
    parser.add_argument("--base-revision", default="master")
    parser.add_argument(
        "--source-revision-manifest",
        type=Path,
        default=Path("/root/autodl-tmp/sombench/models/Qwen3-14B.source_revision.json"),
    )
    parser.add_argument("--repo-id", default="Bithyhy/sombench-qwen3-14b-a-v1")
    parser.add_argument("--upload-revision", default="master")
    parser.add_argument("--release-revision", default="sombench-a-20260805-v1")
    parser.add_argument("--training-summary", type=Path)
    args = parser.parse_args()

    source = args.source.resolve()
    destination = args.destination.resolve()
    source_revision_manifest = args.source_revision_manifest.resolve()
    if not source.is_dir():
        parser.error(f"source directory does not exist: {source}")
    if source == destination:
        parser.error("source and destination must differ")
    if destination.exists() and any(destination.iterdir()):
        parser.error(f"destination must be absent or empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)

    if not source_revision_manifest.is_file():
        parser.error(f"source revision manifest does not exist: {source_revision_manifest}")
    try:
        base_source, base_content_fingerprint = validate_base_source_manifest(
            json.loads(source_revision_manifest.read_text(encoding="utf-8")),
            expected_repo_id=args.base_repo,
            expected_revision=args.base_revision,
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        parser.error(str(exc))
    base_source_manifest_sha256 = sha256(source_revision_manifest)

    selected = [path for path in sorted(source.iterdir()) if path.is_file() and include_file(path)]
    if not selected:
        parser.error("no model repository files selected")
    for path in selected:
        link_or_copy(path, destination / path.name)

    training_summary: dict = {}
    if args.training_summary:
        training_summary = json.loads(args.training_summary.read_text(encoding="utf-8"))
    model_card = make_model_card(
        args.base_repo,
        args.base_revision,
        base_content_fingerprint,
        base_source_manifest_sha256,
        training_summary,
        args.repo_id,
        args.release_revision,
    )
    (destination / "README.md").write_text(model_card, encoding="utf-8")

    validator_cmd = [sys.executable, str(args.validator.resolve()), str(destination)]
    subprocess.run(validator_cmd, check=True)

    manifest_files = []
    for path in sorted(destination.iterdir()):
        if path.is_file() and path.name != "release_manifest.json":
            manifest_files.append(
                {
                    "path": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    manifest = {
        "schema_version": 2,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repo_id": args.repo_id,
        "upload_revision": args.upload_revision,
        "release_revision": args.release_revision,
        "base_repo": args.base_repo,
        "base_revision": args.base_revision,
        "base_content_manifest_sha256": base_content_fingerprint,
        "base_source_manifest_sha256": base_source_manifest_sha256,
        "base_source_revision": base_source,
        "source": str(source),
        "files": manifest_files,
        "total_bytes": sum(item["bytes"] for item in manifest_files),
    }
    (destination / "release_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
