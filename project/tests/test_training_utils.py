from __future__ import annotations

import json
import importlib
import sys
import types
from pathlib import Path

import pytest

from infra.verify_source_snapshot import canonical_manifest_sha256, verify_snapshot
from modelscope.prepare_release import make_model_card, validate_base_source_manifest
from modelscope.publish import (
    create_and_verify_release_tag,
    validate_release_target,
    validate_remote_upload,
)
from training.config import ConfigError, apply_overrides, deep_merge, redact_config
from training.data import IGNORE_INDEX, assistant_only_labels, truncate_example
from training.evaluate import (
    DEFAULTS as EVAL_DEFAULTS,
    audit_gold,
    audit_dataset,
    build_sample_stability,
    build_messages,
    dimension_coverage,
    extract_last_boxed,
    load_prediction_log,
    normalize_letters,
    run as run_evaluation,
    summarize_multi_seed,
    validate_dataset_expectations,
)
from training.merge import verify_base_source
from training.repo_check import inspect_repository
from training.summarize_evaluation import (
    SummaryError,
    build_summary as build_evaluation_summary,
    compact_evaluation,
    render_markdown as render_evaluation_markdown,
)
from training.train import load_verified_initial_adapter, load_verified_source
from training.trainer import GroupLossAccumulator, masked_cross_entropy_by_example, metric_key


def test_model_card_contains_fixed_modelscope_location() -> None:
    card = make_model_card(
        "Qwen/Qwen3-14B",
        "master",
        "c" * 64,
        "d" * 64,
        {"train_samples": 2982},
        "Bithyhy/sombench-qwen3-14b-a-v1",
        "sombench-a-20260805-v1",
    )
    assert 'repo = "Bithyhy/sombench-qwen3-14b-a-v1"' in card
    assert 'revision="sombench-a-20260805-v1"' in card
    assert "c" * 64 in card
    assert "d" * 64 in card
    assert "OWNER/REPO" not in card
    assert "RELEASE_TAG" not in card


def test_release_source_manifest_is_bound_to_repo_and_requested_revision() -> None:
    source = {
        "repo_id": "Qwen/Qwen3-14B",
        "requested_revision": "master",
        "content_manifest_sha256": "c" * 64,
    }
    loaded, fingerprint = validate_base_source_manifest(
        source,
        expected_repo_id="Qwen/Qwen3-14B",
        expected_revision="master",
    )
    assert loaded is source
    assert fingerprint == "c" * 64
    with pytest.raises(ValueError, match="identity mismatch"):
        validate_base_source_manifest(
            source,
            expected_repo_id="Qwen/Qwen3-8B",
            expected_revision="master",
        )


def test_publish_target_must_match_release_manifest() -> None:
    manifest = {
        "repo_id": "Bithyhy/sombench-qwen3-14b-a-v1",
        "upload_revision": "master",
        "release_revision": "sombench-a-20260805-v1",
    }
    validate_release_target(
        manifest,
        repo_id=manifest["repo_id"],
        upload_revision="master",
        release_revision=manifest["release_revision"],
    )
    with pytest.raises(ValueError, match="release target mismatch"):
        validate_release_target(
            manifest,
            repo_id="Bithyhy/wrong-repo",
            upload_revision="master",
            release_revision=manifest["release_revision"],
        )


def test_publish_verifies_remote_bytes_before_tagging(tmp_path: Path) -> None:
    payload = tmp_path / "config.json"
    payload.write_bytes(b"model-config")
    release_manifest = tmp_path / "release_manifest.json"
    release_manifest.write_text("{}\n", encoding="utf-8")
    digest = __import__("hashlib").sha256(payload.read_bytes()).hexdigest()
    manifest_digest = __import__("hashlib").sha256(release_manifest.read_bytes()).hexdigest()

    class RemoteFile:
        def __init__(self, path: str, size: int, sha256: str) -> None:
            self.path = path
            self.size = size
            self.sha256 = sha256

    report = validate_remote_upload(
        {"files": [{"path": payload.name, "bytes": payload.stat().st_size, "sha256": digest}]},
        release_manifest,
        [
            RemoteFile(payload.name, payload.stat().st_size, digest),
            RemoteFile(release_manifest.name, release_manifest.stat().st_size, manifest_digest),
            RemoteFile(".gitattributes", 1, "ignored"),
        ],
    )
    assert report["verified_files"] == 2
    with pytest.raises(ValueError, match="sha256:config.json"):
        validate_remote_upload(
            {"files": [{"path": payload.name, "bytes": payload.stat().st_size, "sha256": digest}]},
            release_manifest,
            [
                RemoteFile(payload.name, payload.stat().st_size, "0" * 64),
                RemoteFile(release_manifest.name, release_manifest.stat().st_size, manifest_digest),
            ],
        )


def test_publish_reverifies_the_created_tag_itself(tmp_path: Path) -> None:
    payload = tmp_path / "config.json"
    payload.write_bytes(b"model-config")
    release_manifest = tmp_path / "release_manifest.json"
    release_manifest.write_text("{}\n", encoding="utf-8")
    digest = __import__("hashlib").sha256(payload.read_bytes()).hexdigest()
    manifest_digest = __import__("hashlib").sha256(release_manifest.read_bytes()).hexdigest()

    class RemoteFile:
        def __init__(self, path: str, size: int, sha256: str) -> None:
            self.path = path
            self.size = size
            self.sha256 = sha256

    class FakeApi:
        def __init__(self, corrupt: bool = False) -> None:
            self.corrupt = corrupt
            self.calls = []

        def create_repo_tag(self, repo_id, repo_type, tag, *, revision):
            self.calls.append(("create", revision, tag))
            return {"tag": tag}

        def list_repo_files(self, repo_id, repo_type, *, revision):
            self.calls.append(("list", revision))
            return [
                RemoteFile(payload.name, payload.stat().st_size, "0" * 64 if self.corrupt else digest),
                RemoteFile(release_manifest.name, release_manifest.stat().st_size, manifest_digest),
            ]

    manifest = {"files": [{"path": payload.name, "bytes": payload.stat().st_size, "sha256": digest}]}
    api = FakeApi()
    result, verification = create_and_verify_release_tag(
        api,
        repo_id="owner/model",
        source_revision="master",
        tag="release-v1",
        manifest=manifest,
        manifest_path=release_manifest,
    )
    assert result == {"tag": "release-v1"}
    assert verification["verified_files"] == 2
    assert api.calls == [("create", "master", "release-v1"), ("list", "release-v1")]

    with pytest.raises(ValueError, match="sha256:config.json"):
        create_and_verify_release_tag(
            FakeApi(corrupt=True),
            repo_id="owner/model",
            source_revision="master",
            tag="release-v1",
            manifest=manifest,
            manifest_path=release_manifest,
        )


def test_source_snapshot_verifier_checks_recorded_sha256(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    payload = b"fixed snapshot"
    (model_dir / "config.json").write_bytes(payload)
    files = [
        {
            "path": "config.json",
            "size": len(payload),
            "sha256": __import__("hashlib").sha256(payload).hexdigest(),
            "last_modified": "2026-08-05T00:00:00+00:00",
        }
    ]
    manifest = tmp_path / "source.json"
    manifest.write_text(
        json.dumps(
            {
                "remote_files": files,
                "content_manifest_sha256": canonical_manifest_sha256(files),
            }
        ),
        encoding="utf-8",
    )
    report = verify_snapshot(manifest, model_dir)
    assert report["status"] == "ok"
    (model_dir / "config.json").write_bytes(b"tampered snap!")
    with pytest.raises(ValueError, match="sha256_mismatch"):
        verify_snapshot(manifest, model_dir)


def test_merge_provenance_gate_rehashes_current_base_snapshot(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    payload = b"fixed snapshot"
    (model_dir / "config.json").write_bytes(payload)
    files = [
        {
            "path": "config.json",
            "size": len(payload),
            "sha256": __import__("hashlib").sha256(payload).hexdigest(),
            "last_modified": "2026-08-05T00:00:00+00:00",
        }
    ]
    fingerprint = canonical_manifest_sha256(files)
    source_path = tmp_path / "source.json"
    source_path.write_text(
        json.dumps(
            {
                "repo_id": "Qwen/Qwen3-14B",
                "requested_revision": "master",
                "remote_files": files,
                "remote_file_count": 1,
                "remote_total_bytes": len(payload),
                "content_manifest_sha256": fingerprint,
            }
        ),
        encoding="utf-8",
    )
    verification_path = tmp_path / "verification.json"
    verification_path.write_text(
        json.dumps(
            {
                "status": "ok",
                "failures": [],
                "content_manifest_sha256": fingerprint,
                "checked_files": 1,
                "checked_bytes": len(payload),
                "local_dir": str(model_dir),
            }
        ),
        encoding="utf-8",
    )
    config = {
        "base_model": str(model_dir),
        "source_revision_manifest": str(source_path),
        "source_verification_manifest": str(verification_path),
        "require_source_verification": True,
        "expected_base_repo": "Qwen/Qwen3-14B",
        "expected_base_revision": "master",
    }
    provenance = verify_base_source(config)
    assert provenance is not None
    assert provenance["merge_time_verification"]["status"] == "ok"

    (model_dir / "unexpected.bin").write_bytes(b"not in manifest")
    with pytest.raises(ValueError, match="outside the source manifest"):
        verify_base_source(config)
    (model_dir / "unexpected.bin").unlink()

    (model_dir / "config.json").write_bytes(b"tampered snap!")
    with pytest.raises(ValueError, match="sha256_mismatch"):
        verify_base_source(config)

    with pytest.raises(ValueError, match="formal merge requires"):
        verify_base_source({"base_model": str(model_dir), "require_source_verification": True})


def test_training_provenance_gate_requires_matching_verified_snapshot(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    source_path = tmp_path / "source.json"
    verification_path = tmp_path / "verification.json"
    source = {
        "content_manifest_sha256": "a" * 64,
        "remote_file_count": 2,
        "remote_total_bytes": 17,
    }
    verification = {
        "status": "ok",
        "content_manifest_sha256": "a" * 64,
        "checked_files": 2,
        "checked_bytes": 17,
        "local_dir": str(model_dir),
    }
    source_path.write_text(json.dumps(source), encoding="utf-8")
    verification_path.write_text(json.dumps(verification), encoding="utf-8")
    loaded_source, loaded_verification = load_verified_source(
        {
            "name_or_path": str(model_dir),
            "source_revision_manifest": str(source_path),
            "source_verification_manifest": str(verification_path),
        }
    )
    assert loaded_source == source
    assert loaded_verification == verification
    verification["checked_bytes"] = 16
    verification_path.write_text(json.dumps(verification), encoding="utf-8")
    with pytest.raises(ValueError, match="verified byte count differs"):
        load_verified_source(
            {
                "name_or_path": str(model_dir),
                "source_revision_manifest": str(source_path),
                "source_verification_manifest": str(verification_path),
            }
        )


def test_initial_adapter_gate_checks_base_fingerprint_and_lora_shape(tmp_path: Path) -> None:
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    fingerprint = "b" * 64
    (adapter_dir / "training_manifest.json").write_text(
        json.dumps({"base_source_revision": {"content_manifest_sha256": fingerprint}}),
        encoding="utf-8",
    )
    adapter_config = {
        "r": 64,
        "lora_alpha": 128,
        "lora_dropout": 0.05,
        "target_modules": ["q_proj", "v_proj"],
    }
    (adapter_dir / "adapter_config.json").write_text(json.dumps(adapter_config), encoding="utf-8")
    model_cfg = {
        "init_adapter": str(adapter_dir),
        "lora": {
            "r": 64,
            "alpha": 128,
            "dropout": 0.05,
            "target_modules": ["q_proj", "v_proj"],
        },
    }
    path, provenance = load_verified_initial_adapter(
        model_cfg,
        {"content_manifest_sha256": fingerprint},
    )
    assert path == adapter_dir.resolve()
    assert provenance["adapter_config"] == adapter_config

    model_cfg["lora"]["r"] = 32
    with pytest.raises(ValueError, match="LoRA configuration differs"):
        load_verified_initial_adapter(model_cfg, {"content_manifest_sha256": fingerprint})


class PrefixStableTokenizer:
    """Tiny Qwen-like chat renderer for exact masking tests."""

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        return_tensors=None,
    ):
        rendered = "".join(f"<{item['role']}>\n{item['content']}</{item['role']}>\n" for item in messages)
        if add_generation_prompt:
            rendered += "<assistant>\n"
        assert tokenize and return_tensors is None
        return list(rendered.encode("utf-8"))


def test_deep_merge_and_dotted_overrides_are_non_mutating() -> None:
    base = {"model": {"name": "qwen", "targets": ["q"]}, "epochs": 1}
    merged = deep_merge(base, {"model": {"targets": ["q", "v"]}})
    result = apply_overrides(merged, ["epochs=2", "model.enabled=true", "model.note=run-a"])
    assert base == {"model": {"name": "qwen", "targets": ["q"]}, "epochs": 1}
    assert result == {
        "model": {"name": "qwen", "targets": ["q", "v"], "enabled": True, "note": "run-a"},
        "epochs": 2,
    }
    with pytest.raises(ConfigError):
        apply_overrides({"model": "qwen"}, ["model.name=bad"])


@pytest.mark.parametrize(
    ("module_name", "default_path"),
    [
        ("training.train", ("training", "learning_rate")),
        ("training.merge", ("max_shard_size",)),
        ("training.evaluate", ("generation", "temperature")),
    ],
)
def test_cli_entrypoints_deep_merge_file_then_apply_dotted_overrides(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
    default_path: tuple[str, ...],
) -> None:
    module = importlib.import_module(module_name)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"sentinel": {"from_file": True}}), encoding="utf-8")
    captured = {}
    monkeypatch.setattr(module, "run", lambda config: captured.update(config))
    monkeypatch.setattr(
        sys,
        "argv",
        [module_name, "--config", str(config_path), "--set", "sentinel.from_cli=7"],
    )
    module.main()
    assert captured["sentinel"] == {"from_file": True, "from_cli": 7}
    value = captured
    for part in default_path:
        assert part in value
        value = value[part]


def test_redact_config_recurses() -> None:
    redacted = redact_config({"api_key": "x", "nested": {"password_file": "p"}, "safe": "ok"})
    assert redacted == {"api_key": "***REDACTED***", "nested": {"password_file": "***REDACTED***"}, "safe": "ok"}


def test_assistant_only_labels_mask_all_non_assistant_spans() -> None:
    tokenizer = PrefixStableTokenizer()
    messages = [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "answer"},
    ]
    input_ids, labels = assistant_only_labels(tokenizer, messages)
    prompt_ids = tokenizer.apply_chat_template(messages[:-1], tokenize=True, add_generation_prompt=True)
    assert labels[: len(prompt_ids)] == [IGNORE_INDEX] * len(prompt_ids)
    assert labels[len(prompt_ids) :] == input_ids[len(prompt_ids) :]
    decoded_trainable = bytes(label for label in labels if label != IGNORE_INDEX).decode("utf-8")
    assert decoded_trainable == "answer</assistant>\n"


def test_assistant_only_labels_support_multiple_turns() -> None:
    tokenizer = PrefixStableTokenizer()
    messages = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "a2"},
    ]
    _, labels = assistant_only_labels(tokenizer, messages)
    trainable = bytes(label for label in labels if label != IGNORE_INDEX).decode("utf-8")
    assert trainable == "a1</assistant>\na2</assistant>\n"


def test_truncation_is_explicit_and_preserves_labels() -> None:
    ids = list(range(10))
    labels = [IGNORE_INDEX] * 7 + [7, 8, 9]
    assert truncate_example(ids, labels, max_length=5, strategy="left") == ([5, 6, 7, 8, 9], [IGNORE_INDEX] * 2 + [7, 8, 9])
    assert truncate_example(ids, labels, max_length=5, strategy="drop") is None
    with pytest.raises(Exception, match="removed the entire assistant"):
        truncate_example(ids, labels, max_length=5, strategy="right")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (r"analysis \boxed{A} more \boxed{\text{B, D}}", ("B", "D")),
        (r"$\boxed{C,A,C}$", ("A", "C")),
        ("no answer", None),
        (r"\boxed{A} then malformed \boxed{", ("A",)),
    ],
)
def test_extract_last_boxed(text: str, expected) -> None:
    assert extract_last_boxed(text) == expected


def test_normalize_letters_does_not_read_latex_command_letters() -> None:
    assert normalize_letters(r"\text{A,C}") == ("A", "C")
    assert normalize_letters("选项 B 和 E") == ("B", "E")
    assert normalize_letters("BE") == ("B", "E")


def test_gold_audit_distinguishes_safe_conflict_and_unmapped() -> None:
    base = {"options": {"A": "无法确定", "B": "是", "C": "否"}}
    safe = audit_gold({**base, "correct_letters": ["B"], "correct_answers": ["是"]})
    conflict = audit_gold({**base, "correct_letters": ["A"], "correct_answers": ["是"]})
    unmapped = audit_gold({**base, "correct_letters": ["A"], "correct_answers": ["证据不足"]})
    invalid_single = audit_gold(
        {
            "options": {"A": "甲", "B": "乙"},
            "correct_letters": ["A", "B"],
            "correct_answers": ["甲", "乙"],
            "task_format": "mcq_single",
        }
    )
    assert safe.safe and safe.mapped == ("B",)
    assert not conflict.safe and conflict.reason == "letter_text_conflict"
    assert not unmapped.safe and unmapped.reason == "answer_text_unmapped"
    assert not invalid_single.safe and invalid_single.reason == "single_format_with_multiple_answers"


def test_build_messages_never_includes_gold() -> None:
    record = {
        "sample_id": "x",
        "story": "故事",
        "question": "问题？",
        "options": {"B": "第二项", "A": "第一项"},
        "correct_letters": ["B"],
    }
    messages = build_messages(record, "system")
    assert messages[0] == {"role": "system", "content": "system"}
    assert "A. 第一项\nB. 第二项" in messages[1]["content"]
    assert "correct" not in messages[1]["content"]


def test_build_messages_selects_single_or_multi_reference_prompt() -> None:
    prompts = {"single": "SINGLE", "multi": "MULTI"}
    base = {
        "sample_id": "x",
        "story": "故事",
        "question": "问题？",
        "options": {"A": "甲", "B": "乙"},
    }
    assert build_messages({**base, "task_format": "mcq_single"}, prompts)[0]["content"] == "SINGLE"
    assert build_messages({**base, "task_format": "mcq_multi"}, prompts)[0]["content"] == "MULTI"


def _prediction(sample_id: str, seed: int, predicted, gold, qtype: str, safe: bool = True):
    return {
        "sample_id": sample_id,
        "seed": seed,
        "predicted_letters": predicted,
        "gold_letters": gold,
        "mapped_letters": gold,
        "qtype": qtype,
        "dimension": "1.1" if qtype == "Q1" else "2.1",
        "safe_gold": safe,
        "gold_audit_reason": "letter_text_agree" if safe else "letter_text_conflict",
    }


def test_multi_seed_metrics_include_groups_stability_and_majority() -> None:
    rows = [
        _prediction("a", 1, ["A"], ["A"], "Q1"),
        _prediction("b", 1, ["B"], ["A"], "Q2", safe=False),
        _prediction("a", 2, ["A"], ["A"], "Q1"),
        _prediction("b", 2, ["A"], ["A"], "Q2", safe=False),
        _prediction("a", 3, ["A"], ["A"], "Q1"),
        _prediction("b", 3, ["A"], ["A"], "Q2", safe=False),
    ]
    metrics = summarize_multi_seed(rows, [1, 2, 3])
    assert metrics["per_seed"]["1"]["reported_letter_overall"]["accuracy"] == 0.5
    assert metrics["per_seed"]["1"]["safe_agreement_subset"]["accuracy"] == 1.0
    assert metrics["aggregate"]["by_qtype"]["Q1"]["reported_letters"]["mean"] == 1.0
    assert metrics["aggregate"]["by_dimension"]["1.1"]["reported_letters"]["mean"] == 1.0
    assert metrics["aggregate"]["by_primary_dimension"]["1:mentalization"]["reported_letters"]["mean"] == 1.0
    assert metrics["stability"]["all_seed_consistency_rate"] == 0.5
    assert metrics["stability"]["majority_vote"]["reported_letter_overall"]["accuracy"] == 1.0


def test_public_rows_keep_unknown_dimension_without_inference() -> None:
    base = {
        "sample_id": "public-a",
        "predicted_letters": ["A"],
        "gold_letters": ["A"],
        "mapped_letters": ["A"],
        "qtype": "Q1",
        "dimension": "unknown",
        "safe_gold": True,
        "gold_audit_reason": "letter_text_agree",
    }
    rows = [{**base, "seed": seed} for seed in (1, 2, 3)]
    coverage = dimension_coverage(rows)
    assert coverage == {
        "total": 3,
        "known": 0,
        "unknown": 3,
        "coverage": 0.0,
        "fine_dimension_count": 0,
        "fine_dimensions": [],
        "primary_dimension_count": 0,
        "primary_dimensions": [],
        "note": "No dimension field was supplied; rows are retained and reported as unknown, not inferred.",
    }
    metrics = summarize_multi_seed(rows, [1, 2, 3], sample_order=["public-a"])
    assert metrics["aggregate"]["dimension_coverage"]["unknown"] == 1
    assert metrics["aggregate"]["by_dimension"] == {}
    assert metrics["aggregate"]["by_primary_dimension"] == {}


def test_sample_stability_has_one_row_per_sample_and_majority() -> None:
    rows = [
        _prediction("a", 11, ["A"], ["A"], "Q1"),
        _prediction("a", 22, ["A"], ["A"], "Q1"),
        _prediction("a", 33, ["B"], ["A"], "Q1"),
        _prediction("b", 11, ["C"], ["C"], "Q2"),
        _prediction("b", 22, ["C"], ["C"], "Q2"),
        _prediction("b", 33, ["C"], ["C"], "Q2"),
    ]
    stability = build_sample_stability(rows, [11, 22, 33], sample_order=["a", "b"])
    assert [row["sample_id"] for row in stability] == ["a", "b"]
    assert stability[0]["majority_prediction"] == ["A"]
    assert stability[0]["all_seed_consistent"] is False
    assert stability[1]["all_seed_consistent"] is True


def test_prediction_log_deduplicates_successes_but_audits_attempts(tmp_path: Path) -> None:
    path = tmp_path / "predictions.jsonl"
    attempts = [
        {"seed": 1, "sample_id": "a", "predicted_letters": ["A"], "error": None},
        {"seed": 2, "sample_id": "a", "predicted_letters": None, "error": "timeout"},
        {"seed": 1, "sample_id": "a", "predicted_letters": ["B"], "error": None},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in attempts), encoding="utf-8")
    state = load_prediction_log(path)
    assert len(state.completed) == 1
    assert state.completed[(1, "a")]["predicted_letters"] == ["B"]
    assert state.total_log_rows == 3
    assert state.failed_log_rows == 1
    assert state.duplicate_success_keys == {"1:a": 2}


def test_dataset_audit_and_expectations_separate_safe_labels_and_dimensions() -> None:
    records = [
        {
            "sample_id": "safe",
            "question_type": "Q1",
            "task_format": "mcq_single",
            "options": {"A": "甲", "B": "乙"},
            "correct_letters": ["A"],
            "correct_answers": ["甲"],
        },
        {
            "sample_id": "conflict",
            "question_type": "Q2",
            "task_format": "mcq_multi",
            "options": {"A": "甲", "B": "乙"},
            "correct_letters": ["A"],
            "correct_answers": ["乙"],
        },
    ]
    audit = audit_dataset(records)
    assert audit["records"] == 2
    assert audit["safe_records"] == 1
    assert audit["dimension_coverage"]["unknown"] == 2
    validate_dataset_expectations(
        audit,
        {
            "expected_records": 2,
            "expected_safe_records": 1,
            "expected_qtype_counts": {"Q1": 1, "Q2": 1},
            "expected_dimension_coverage": 0.0,
        },
    )
    with pytest.raises(ValueError, match="safe_records"):
        validate_dataset_expectations(audit, {"expected_safe_records": 2})


def test_evaluation_resume_writes_exact_canonical_grid_without_recalling_api(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = [
        {
            "sample_id": sample_id,
            "question_type": "Q1",
            "task_format": "mcq_single",
            "story": "故事",
            "question": "选择？",
            "options": {"A": "甲", "B": "乙"},
            "correct_letters": ["A"],
            "correct_answers": ["甲"],
        }
        for sample_id in ("a", "b")
    ]
    dataset = tmp_path / "public.jsonl"
    dataset.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8")
    calls: list[dict] = []

    class FakeCompletions:
        def create(self, **kwargs):
            calls.append(kwargs)
            message = types.SimpleNamespace(content=r"分析\n\boxed{A}", reasoning_content=None)
            choice = types.SimpleNamespace(message=message, finish_reason="stop")
            return types.SimpleNamespace(choices=[choice], usage=None)

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = types.SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeOpenAI))
    output_dir = tmp_path / "eval"
    config = deep_merge(
        EVAL_DEFAULTS,
        {
            "model": "fake-model",
            "dataset": str(dataset),
            "output_dir": str(output_dir),
            "api": {"workers": 2, "max_retries": 0},
            "generation": {"seeds": [11, 22, 33]},
            "validation": {
                "expected_records": 2,
                "expected_safe_records": 2,
                "expected_qtype_counts": {"Q1": 2},
            },
        },
    )
    run_evaluation(config)
    assert len(calls) == 6
    run_evaluation(config)
    assert len(calls) == 6
    canonical = [json.loads(line) for line in (output_dir / "predictions_canonical.jsonl").read_text().splitlines()]
    stability = [json.loads(line) for line in (output_dir / "sample_stability.jsonl").read_text().splitlines()]
    metrics = json.loads((output_dir / "metrics.json").read_text())
    assert len(canonical) == 6
    assert len({(row["seed"], row["sample_id"]) for row in canonical}) == 6
    assert len(stability) == 2
    assert metrics["run"]["all_records_retained"] is True
    assert metrics["run"]["missing_or_failed"] == 0


def _write_evaluation_summary_fixture(root: Path, *, public: bool) -> None:
    seeds = [11, 22, 33]
    sample_ids = ["q1", "q2", "q3"]
    qtypes = {"q1": "Q1", "q2": "Q2", "q3": "Q3"}
    dimensions = {"q1": "1.1.1", "q2": "2.1.1", "q3": "3.1.1"}
    predictions = {
        "q1": {11: ["A"], 22: ["A"], 33: ["A"]},
        "q2": {11: ["A"], 22: None, 33: ["A"]},
        "q3": {11: ["A"], 22: ["B"], 33: ["C"]},
    }
    rows = []
    for sample_id in sample_ids:
        for seed in seeds:
            safe = public and sample_id != "q2"
            rows.append(
                {
                    "sample_id": sample_id,
                    "seed": seed,
                    "predicted_letters": predictions[sample_id][seed],
                    "gold_letters": ["A"],
                    "mapped_letters": ["A"] if public else None,
                    "qtype": qtypes[sample_id],
                    "dimension": "unknown" if public else dimensions[sample_id],
                    "safe_gold": safe,
                    "gold_audit_reason": (
                        "letter_text_agree"
                        if safe
                        else ("letter_text_conflict" if public else "answer_text_not_provided")
                    ),
                }
            )
    metrics = summarize_multi_seed(rows, seeds, sample_order=sample_ids)
    metrics["run"] = {
        "expected_predictions": 9,
        "completed_predictions": 9,
        "missing_or_failed": 0,
        "canonical_prediction_rows": 9,
        "all_records_retained": True,
    }
    root.mkdir()
    (root / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    (root / "evaluation_manifest.json").write_text(
        json.dumps({"model": "fixture", "dataset": f"{root.name}.jsonl", "record_count": 3}),
        encoding="utf-8",
    )
    stability = build_sample_stability(rows, seeds, sample_order=sample_ids)
    (root / "sample_stability.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in stability),
        encoding="utf-8",
    )
    canonical = []
    for row in rows:
        copied = dict(row)
        copied["finish_reason"] = (
            "length"
            if copied["sample_id"] == "q3" and copied["seed"] == 33
            else "stop"
        )
        canonical.append(copied)
    (root / "predictions_canonical.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in canonical),
        encoding="utf-8",
    )


def test_read_only_evaluation_summary_extracts_current_schema(tmp_path: Path) -> None:
    public_dir = tmp_path / "public"
    dev_dir = tmp_path / "dev"
    _write_evaluation_summary_fixture(public_dir, public=True)
    _write_evaluation_summary_fixture(dev_dir, public=False)
    summary = build_evaluation_summary(public_dir, dev_dir)
    assert summary["schema_version"] == 3
    public = summary["evaluations"]["public"]
    dev = summary["evaluations"]["dev"]
    assert public["seeds"] == ["11", "22", "33"]
    assert public["labels"]["safe_agreement"]["per_seed"]["11"]["total"] == 2
    assert public["labels"]["mapped_answer_text"]["per_seed"]["11"]["total"] == 3
    assert public["qtypes"]["Q1"]["reported_letters"]["mean"] == 1.0
    assert public["qtypes"]["Q2"]["per_seed"]["22"]["reported_letters"]["accuracy"] == 0.0
    assert public["parse_rate"]["per_seed"]["22"] == pytest.approx(2 / 3)
    assert public["stability"]["consistency_rate"] == pytest.approx(1 / 3)
    assert public["stability"]["majority_tie_count"] == 1
    assert public["stability"]["majority_determinable_samples"] == 2
    assert public["stability"]["majority_determinable_rate"] == pytest.approx(2 / 3)
    prediction_audit = public["prediction_artifact_audit"]
    assert prediction_audit["grid_complete"] is True
    assert prediction_audit["rows"] == 9
    assert prediction_audit["finish_reason_counts"] == {"length": 1, "stop": 8}
    assert prediction_audit["length_truncation_count"] == 1
    assert prediction_audit["length_truncation_rate"] == pytest.approx(1 / 9)
    assert prediction_audit["empty_predicted_letters_count"] == 1
    assert prediction_audit["empty_predicted_letters_rate"] == pytest.approx(1 / 9)
    assert public["dimension_coverage"]["coverage"] == 0.0
    assert dev["dimension_coverage"]["fine_dimension_count"] == 3
    assert dev["primary_dimensions"]["1:mentalization"]["per_seed"]["11"][
        "reported_letters"
    ]["accuracy"] == 1.0
    assert dev["primary_dimensions"]["2:strategic_social_interaction"]["per_seed"]["22"][
        "reported_letters"
    ]["accuracy"] == 0.0
    leaderboard = dev["leaderboard_a_inferred_weighted_score"]
    assert leaderboard["status"] == "complete"
    assert leaderboard["per_seed"]["11"]["score"] == 1.0
    assert leaderboard["per_seed"]["22"]["score"] == pytest.approx(750 / 1775)
    assert leaderboard["per_seed"]["33"]["score"] == pytest.approx((750 + 675) / 1775)
    assert public["leaderboard_a_inferred_weighted_score"]["status"] == "not_applicable"
    assert public["warnings"] == []

    markdown = render_evaluation_markdown(summary)
    assert "Reported letters" in markdown
    assert "Safe agreement" in markdown
    assert "Mapped answer text" in markdown
    assert "Seed 11 acc/parse" in markdown
    assert "Q1/Q2/Q3 accuracy" in markdown
    assert "Primary dimensions (reported-letter accuracy)" in markdown
    assert "A-leaderboard inferred weighted score" in markdown
    assert "Majority ties" in markdown
    assert "Finish reasons" in markdown
    assert "Length truncations" in markdown
    assert "Empty predicted letters" in markdown
    assert "length=1, stop=8" in markdown


def test_evaluation_summary_rejects_non_three_seed_schema(tmp_path: Path) -> None:
    public_dir = tmp_path / "public"
    _write_evaluation_summary_fixture(public_dir, public=True)
    metrics_path = public_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    metrics["per_seed"].pop("33")
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    with pytest.raises(SummaryError, match="exactly three seeds"):
        compact_evaluation(public_dir, name="public")


def test_evaluation_summary_rejects_duplicate_canonical_prediction_key(tmp_path: Path) -> None:
    public_dir = tmp_path / "public"
    _write_evaluation_summary_fixture(public_dir, public=True)
    canonical_path = public_dir / "predictions_canonical.jsonl"
    rows = canonical_path.read_text(encoding="utf-8").splitlines()
    canonical_path.write_text("\n".join([*rows, rows[0]]) + "\n", encoding="utf-8")
    with pytest.raises(SummaryError, match=r"duplicate \(seed, sample_id\)"):
        compact_evaluation(public_dir, name="public")


def test_evaluation_summary_rejects_incomplete_canonical_prediction_grid(tmp_path: Path) -> None:
    public_dir = tmp_path / "public"
    _write_evaluation_summary_fixture(public_dir, public=True)
    canonical_path = public_dir / "predictions_canonical.jsonl"
    rows = canonical_path.read_text(encoding="utf-8").splitlines()
    canonical_path.write_text("\n".join(rows[:-1]) + "\n", encoding="utf-8")
    with pytest.raises(SummaryError, match=r"missing \(seed, sample_id\) grid rows"):
        compact_evaluation(public_dir, name="public")


def test_group_loss_accumulator_is_token_weighted() -> None:
    accumulator = GroupLossAccumulator(["Q1", "Q2"], ["1.1"])
    accumulator.add("train", ["Q1", "Q1"], ["1.1", "1.1"], [4.0, 2.0], [2.0, 1.0])
    metrics = accumulator.flush("train")
    assert metrics["loss_by_qtype/Q1"] == 2.0
    assert metrics["loss_by_dimension/1.1"] == 2.0
    assert metrics["loss_macro_qtype"] == 2.0
    assert metrics["loss_macro_dimension"] == 2.0
    assert metric_key("a/b c") == "a_b_c"


def test_group_loss_accumulator_macro_dimension_is_not_frequency_weighted() -> None:
    accumulator = GroupLossAccumulator(["Q1"], ["1.1", "1.2"])
    accumulator.add(
        "eval",
        ["Q1", "Q1"],
        ["1.1", "1.2"],
        [90.0, 10.0],
        [90.0, 1.0],
    )
    metrics = accumulator.flush("eval")
    assert metrics["eval_loss_by_dimension/1.1"] == 1.0
    assert metrics["eval_loss_by_dimension/1.2"] == 10.0
    assert metrics["eval_loss_macro_dimension"] == 5.5


def test_masked_cross_entropy_by_example() -> None:
    torch = pytest.importorskip("torch")
    labels = torch.tensor([[IGNORE_INDEX, 1, 0], [IGNORE_INDEX, 0, IGNORE_INDEX]])
    logits = torch.tensor(
        [
            [[-10.0, 10.0], [10.0, -10.0], [0.0, 0.0]],
            [[10.0, -10.0], [0.0, 0.0], [0.0, 0.0]],
        ]
    )
    sums, counts = masked_cross_entropy_by_example(logits, labels)
    assert counts.tolist() == [2, 1]
    assert sums.tolist() == pytest.approx([0.0, 0.0], abs=1e-6)


def test_repository_check_accepts_complete_bf16_layout(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "qwen3", "torch_dtype": "bfloat16", "architectures": ["Qwen3ForCausalLM"]}),
        encoding="utf-8",
    )
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"chat_template": "{{ messages }}"}), encoding="utf-8")
    (tmp_path / "tokenizer.json").write_text("{}", encoding="utf-8")
    (tmp_path / "generation_config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "configuration.json").write_text("{}", encoding="utf-8")
    (tmp_path / "model.safetensors").write_bytes(b"safetensor")
    report = inspect_repository(tmp_path, min_weight_bytes=1)
    assert report.ok, report.errors
    assert report.total_weight_bytes == len(b"safetensor")


def test_repository_check_rejects_adapter_or_quantized_repo(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3",
                "torch_dtype": "float16",
                "architectures": ["Qwen3ForCausalLM"],
                "quantization_config": {"load_in_4bit": True},
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"chat_template": "x"}), encoding="utf-8")
    (tmp_path / "tokenizer.json").write_text("{}", encoding="utf-8")
    (tmp_path / "model.safetensors").write_bytes(b"x")
    (tmp_path / "adapter_config.json").write_text("{}", encoding="utf-8")
    report = inspect_repository(tmp_path, min_weight_bytes=1)
    assert not report.ok
    assert any("adapter/quantized" in error for error in report.errors)
    assert any("bfloat16" in error for error in report.errors)
