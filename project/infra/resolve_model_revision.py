#!/usr/bin/env python3
"""Record the ModelScope branch/tag metadata resolved before a model download."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def find_named(value: Any, revision: str) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    if isinstance(value, dict):
        names = [value.get(key) for key in ("name", "Name", "revision", "Revision")]
        if revision in names:
            matches.append(value)
        for nested in value.values():
            matches.extend(find_named(nested, revision))
    elif isinstance(value, list):
        for nested in value:
            matches.extend(find_named(nested, revision))
    return matches


def serialize_file_info(value: Any) -> dict[str, Any]:
    modified = getattr(value, "last_modified", None)
    if hasattr(modified, "isoformat"):
        modified = modified.isoformat()
    return {
        "path": getattr(value, "path", None),
        "size": getattr(value, "size", None),
        "sha256": getattr(value, "sha256", None),
        "last_modified": modified,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--revision", default="master")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from modelscope_hub import HubApi

    api = HubApi()
    revisions = api.list_repo_revisions(args.repo_id, "model")
    remote_files = sorted(
        (serialize_file_info(value) for value in api.list_repo_files(
            args.repo_id,
            "model",
            revision=args.revision,
        )),
        key=lambda value: str(value["path"]),
    )
    canonical_files = json.dumps(
        remote_files,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    serializable = json.loads(json.dumps(revisions, default=str))
    selected = find_named(serializable, args.revision)
    report = {
        "repo_id": args.repo_id,
        "requested_revision": args.revision,
        "resolved_at": datetime.now(timezone.utc).isoformat(),
        "matching_revision_metadata": selected,
        "all_revision_metadata": serializable,
        "remote_files": remote_files,
        "remote_file_count": len(remote_files),
        "remote_total_bytes": sum(
            value["size"] for value in remote_files if isinstance(value["size"], int)
        ),
        "content_manifest_sha256": hashlib.sha256(canonical_files).hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
