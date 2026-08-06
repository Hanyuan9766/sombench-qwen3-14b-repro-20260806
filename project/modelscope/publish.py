#!/usr/bin/env python3
"""Create/update a private ModelScope model repository without persisting a token."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable


def validate_release_target(
    manifest: dict,
    *,
    repo_id: str,
    upload_revision: str,
    release_revision: str | None,
) -> None:
    expected = {
        "repo_id": repo_id,
        "upload_revision": upload_revision,
        "release_revision": release_revision,
    }
    mismatches = [
        f"{key}: manifest={manifest.get(key)!r}, publish={value!r}"
        for key, value in expected.items()
        if manifest.get(key) != value
    ]
    if mismatches:
        raise ValueError("release target mismatch: " + "; ".join(mismatches))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_remote_upload(
    manifest: dict[str, Any],
    manifest_path: Path,
    remote_files: Iterable[Any],
) -> dict[str, Any]:
    """Verify every release byte on a concrete remote revision."""

    expected = {
        str(item["path"]): {"bytes": int(item["bytes"]), "sha256": str(item["sha256"])}
        for item in manifest.get("files", [])
    }
    expected[manifest_path.name] = {
        "bytes": manifest_path.stat().st_size,
        "sha256": _sha256(manifest_path),
    }
    observed = {
        str(getattr(item, "path", "")): {
            "bytes": getattr(item, "size", None),
            "sha256": getattr(item, "sha256", None),
        }
        for item in remote_files
    }
    errors: list[str] = []
    for path, expected_info in expected.items():
        remote_info = observed.get(path)
        if remote_info is None:
            errors.append(f"missing:{path}")
            continue
        if remote_info["bytes"] != expected_info["bytes"]:
            errors.append(f"size:{path}")
        if remote_info["sha256"] != expected_info["sha256"]:
            errors.append(f"sha256:{path}")
    unexpected = sorted(
        path
        for path in observed
        if path not in expected and path and not any(part.startswith(".") for part in Path(path).parts)
    )
    if unexpected:
        errors.extend(f"unexpected:{path}" for path in unexpected)
    if errors:
        raise ValueError("remote upload verification failed: " + ", ".join(errors))
    return {
        "verified_files": len(expected),
        "verified_bytes": sum(item["bytes"] for item in expected.values()),
        "unexpected_hidden_files": sorted(path for path in observed if path not in expected),
    }


def verify_uploaded_revision(
    api: Any,
    *,
    repo_id: str,
    revision: str,
    manifest: dict[str, Any],
    manifest_path: Path,
) -> dict[str, Any]:
    """Fetch and byte-verify one concrete remote branch or tag."""

    return validate_remote_upload(
        manifest,
        manifest_path,
        api.list_repo_files(repo_id, "model", revision=revision),
    )


def create_and_verify_release_tag(
    api: Any,
    *,
    repo_id: str,
    source_revision: str,
    tag: str,
    manifest: dict[str, Any],
    manifest_path: Path,
) -> tuple[Any, dict[str, Any]]:
    """Create the immutable tag, then verify the tag itself byte for byte."""

    result = api.create_repo_tag(repo_id, "model", tag, revision=source_revision)
    verification = verify_uploaded_revision(
        api,
        repo_id=repo_id,
        revision=tag,
        manifest=manifest,
        manifest_path=manifest_path,
    )
    return result, verification


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("release_dir", type=Path)
    parser.add_argument("--repo-id", default=os.environ.get("MODELSCOPE_MODEL_ID"))
    parser.add_argument("--revision", default="master")
    parser.add_argument(
        "--tag",
        help="optional immutable submission tag created after the master upload",
    )
    parser.add_argument("--visibility", choices=("private", "public"), default="private")
    parser.add_argument("--commit-message", default="SoMBench Qwen3-14B merged release")
    parser.add_argument("--max-workers", type=int, default=8)
    args = parser.parse_args()

    token = os.environ.get("MODELSCOPE_API_TOKEN")
    if not token:
        parser.error("MODELSCOPE_API_TOKEN must be set in the environment")
    if not args.repo_id or args.repo_id.count("/") != 1:
        parser.error("--repo-id or MODELSCOPE_MODEL_ID must be owner/name")
    if args.revision in {"main", "latest"}:
        parser.error("competition submission must not use revision main or latest")
    if args.tag in {"main", "master", "latest"}:
        parser.error("--tag must be a new immutable release name")

    release_dir = args.release_dir.resolve()
    if not (release_dir / "config.json").is_file():
        parser.error(f"not a complete release directory: {release_dir}")
    manifest_path = release_dir / "release_manifest.json"
    if not manifest_path.is_file():
        parser.error(f"release manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_release_target(
        manifest,
        repo_id=args.repo_id,
        upload_revision=args.revision,
        release_revision=args.tag,
    )

    from modelscope_hub import HubApi

    api = HubApi(token=token)
    identity = api.whoami()
    if not api.repo_exists(args.repo_id, "model"):
        api.create_repo(
            args.repo_id,
            "model",
            visibility=args.visibility,
            license="apache-2.0",
        )

    result = api.upload_folder(
        args.repo_id,
        "model",
        str(release_dir),
        path_in_repo="",
        revision=args.revision,
        commit_message=args.commit_message,
        max_workers=args.max_workers,
    )
    remote_verification = verify_uploaded_revision(
        api,
        repo_id=args.repo_id,
        revision=args.revision,
        manifest=manifest,
        manifest_path=manifest_path,
    )
    tag_result = None
    tag_verification = None
    if args.tag:
        tag_result, tag_verification = create_and_verify_release_tag(
            api,
            repo_id=args.repo_id,
            source_revision=args.revision,
            tag=args.tag,
            manifest=manifest,
            manifest_path=manifest_path,
        )
    output = {
        "repo_id": args.repo_id,
        "revision": args.revision,
        "url": f"https://modelscope.cn/models/{args.repo_id}",
        "submission_revision": args.tag or args.revision,
        "release_manifest_schema": manifest.get("schema_version"),
        "base_content_manifest_sha256": manifest.get("base_content_manifest_sha256"),
        "authenticated_user": getattr(identity, "username", str(identity)),
        "upload_result": result,
        "remote_verification": remote_verification,
        "tag_result": tag_result,
        "tag_verification": tag_verification,
        "token_persisted": False,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"PUBLISH_FAILED: {exc}", file=sys.stderr)
        raise
