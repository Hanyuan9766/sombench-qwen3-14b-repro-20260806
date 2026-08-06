#!/usr/bin/env python3
"""Verify a downloaded snapshot against the pre-download ModelScope manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_manifest_sha256(files: list[dict[str, Any]]) -> str:
    payload = json.dumps(
        files,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def verify_snapshot(manifest_path: Path, local_dir: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest.get("remote_files")
    if not isinstance(files, list) or not files:
        raise ValueError("source manifest has no remote_files")
    expected_fingerprint = manifest.get("content_manifest_sha256")
    observed_fingerprint = canonical_manifest_sha256(files)
    if expected_fingerprint != observed_fingerprint:
        raise ValueError("source manifest content fingerprint is invalid")

    checked: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for item in files:
        relative = item.get("path")
        expected_size = item.get("size")
        expected_sha256 = item.get("sha256")
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or not isinstance(expected_size, int)
            or expected_size < 0
            or not isinstance(expected_sha256, str)
            or not SHA256_RE.fullmatch(expected_sha256)
        ):
            failures.append({"path": relative, "reason": "invalid_manifest_entry"})
            continue

        path = local_dir / relative
        if not path.is_file():
            failures.append({"path": relative, "reason": "missing"})
            continue
        observed_size = path.stat().st_size
        if observed_size != expected_size:
            failures.append(
                {
                    "path": relative,
                    "reason": "size_mismatch",
                    "expected_size": expected_size,
                    "observed_size": observed_size,
                }
            )
            continue
        observed_sha256 = sha256(path)
        if observed_sha256 != expected_sha256:
            failures.append(
                {
                    "path": relative,
                    "reason": "sha256_mismatch",
                    "expected_sha256": expected_sha256,
                    "observed_sha256": observed_sha256,
                }
            )
            continue
        checked.append({"path": relative, "bytes": observed_size, "sha256": observed_sha256})

    report = {
        "manifest": str(manifest_path.resolve()),
        "local_dir": str(local_dir.resolve()),
        "content_manifest_sha256": observed_fingerprint,
        "expected_files": len(files),
        "checked_files": len(checked),
        "checked_bytes": sum(item["bytes"] for item in checked),
        "failures": failures,
        "status": "ok" if not failures and len(checked) == len(files) else "failed",
    }
    if report["status"] != "ok":
        raise ValueError(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("local_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = verify_snapshot(args.manifest.resolve(), args.local_dir.resolve())
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    print(rendered, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
