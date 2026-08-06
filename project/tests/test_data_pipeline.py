import json
from pathlib import Path

import pytest

from data_pipeline.cli import main as cli_main
from data_pipeline.core import (
    BOXED_RE,
    PipelineConfig,
    assign_story_clusters,
    audit_public_records,
    balance_by_dimension,
    clean_assistant_answer,
    clean_training_records,
    grouped_train_dev_split,
    read_jsonl,
)


def _train_record(
    sample_id,
    story,
    *,
    dimension="1.1.1",
    qtype="Q1",
    options=("A", "B", "C", "D"),
    gold=("B",),
    answer=None,
):
    option_text = "\n".join(f"{letter}. 选项{letter}" for letter in options)
    prompt = (
        f"请根据下面的故事回答问题。\n\n故事：\n{story}\n\n"
        f"问题：\n谁知道消息？\n\n选项：\n{option_text}"
    )
    answer = answer or (
        "<think>\n嗯，这个问题看起来需要分析。角色只听到了旧消息。"
        "角色没有收到更正，所以仍相信原来的说法。\n</think>\n"
        "综上，答案是 B。\n\\boxed{A}\n最终答案为：\n\\boxed{B}"
    )
    return {
        "sample_id": sample_id,
        "sample_index": sample_id,
        "prompt_type": "mcq_multi" if len(gold) > 1 else "mcq_single",
        "gold_letters": list(gold),
        "pred_letters": list(gold),
        "meta": {"dim": dimension, "qtype": qtype},
        "messages": [
            {"role": "system", "content": "回答问题"},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ],
    }


def _public_record(
    sample_id,
    *,
    story="甲告诉乙会议延期，之后没有人更正。",
    declared=("B",),
    answers=("正确选项",),
    task_format="mcq_single",
):
    return {
        "sample_id": sample_id,
        "story": story,
        "question": "乙相信什么？",
        "question_type": "Q1",
        "task_format": task_format,
        "options": {"A": "错误选项", "B": "正确选项", "C": "无法确定"},
        "correct_letters": list(declared),
        "correct_answers": list(answers),
    }


def _write_jsonl(path: Path, rows):
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_clean_assistant_answer_has_one_final_canonical_box():
    raw = (
        "<think>嗯，这个问题看起来复杂。甲知道消息。甲知道消息。"
        "乙没有听到更新，因此保留旧信念。</think>\n"
        "最终答案为：\\boxed{A}\n补充说明。\\boxed{B,D}"
    )
    cleaned = clean_assistant_answer(raw, ["D", "B"], max_cot_chars=100)

    assert BOXED_RE.findall(cleaned) == ["B,D"]
    assert cleaned.endswith(r"\boxed{B,D}")
    assert "这个问题看起来" not in cleaned
    assert len(cleaned) < len(raw) + 50


def test_clean_training_filters_legacy_six_option_q2_and_clusters_story():
    same_story_a = "①甲在办公室告诉乙会议延期。②乙没有收到更正。"
    same_story_b = "1甲在办公室告诉乙会议延期；2乙没有收到更正！"
    rows = [
        _train_record("one", same_story_a),
        _train_record("two", same_story_b, dimension="1.1.2"),
        _train_record(
            "legacy",
            "一段旧故事。",
            qtype="Q2",
            options=("A", "B", "C", "D", "E", "F"),
        ),
    ]
    config = PipelineConfig(fuzzy_story_threshold=0.80, max_cot_chars=120)
    clean, rejected, audit = clean_training_records(rows, config)

    assert len(clean) == 2
    assert rejected[0]["reasons"] == ["legacy_q2_six_options"]
    assert audit["rejection_reasons"]["legacy_q2_six_options"] == 1
    assert clean[0]["story_cluster_id"] == clean[1]["story_cluster_id"]
    assert all(len(BOXED_RE.findall(row["messages"][-1]["content"])) == 1 for row in clean)


def test_fuzzy_story_clustering_merges_close_rewrites_not_unrelated_text():
    records = [
        {"story": "张老师在会议室宣布活动推迟，学生小王听见后通知同桌小李。"},
        {"story": "张老师在会议室宣布活动延期，学生小王听见后通知同桌小李。"},
        {"story": "厨师在餐厅准备晚餐，顾客查看菜单后点了一碗面。"},
    ]
    audit = assign_story_clusters(records, threshold=0.70, shingle_size=3, sketch_size=10)

    assert records[0]["story_cluster_id"] == records[1]["story_cluster_id"]
    assert records[0]["story_cluster_id"] != records[2]["story_cluster_id"]
    assert audit["heuristic_story_clusters"] == 2


def test_grouped_split_has_no_story_leakage():
    rows = []
    for cluster in range(12):
        for qtype in ("Q1", "Q2", "Q3"):
            rows.append(
                {
                    "sample_id": f"{cluster}-{qtype}",
                    "story_cluster_id": f"story-{cluster}",
                    "dimension": f"1.1.{cluster % 3}",
                }
            )
    train, dev, audit = grouped_train_dev_split(rows, val_fraction=0.25, seed=7)

    assert train and dev
    assert {row["story_cluster_id"] for row in train}.isdisjoint(
        {row["story_cluster_id"] for row in dev}
    )
    assert audit["cluster_overlap"] == []


def test_dimension_balancing_produces_equal_counts_and_instance_ids():
    rows = [
        {"sample_id": "a1", "dimension": "a"},
        {"sample_id": "b1", "dimension": "b"},
        {"sample_id": "b2", "dimension": "b"},
        {"sample_id": "b3", "dimension": "b"},
    ]
    balanced, audit = balance_by_dimension(rows, target="3", seed=3)

    assert audit["counts_after"] == {"a": 3, "b": 3}
    assert len(balanced) == 6
    assert len({row["_pipeline"]["sample_instance_id"] for row in balanced}) == 6


def test_public_label_audit_is_conservative_and_keeps_review_fields():
    rows = [
        _public_record("agree"),
        _public_record("conflict", declared=("A",)),
        _public_record("unmapped", answers=("答案文本不在选项中",)),
        _public_record(
            "format",
            declared=("A", "B"),
            answers=("错误选项", "正确选项"),
            task_format="mcq_single",
        ),
    ]
    safe, conflicts, audit = audit_public_records(
        rows, PipelineConfig(fuzzy_story_threshold=1.0)
    )

    assert [row["sample_id"] for row in safe] == ["agree"]
    assert {row["sample_id"] for row in conflicts} == {"conflict", "unmapped", "format"}
    conflict = next(row for row in conflicts if row["sample_id"] == "conflict")
    assert conflict["_label_audit"]["declared_letters"] == ["A"]
    assert conflict["_label_audit"]["answer_text_mapped_letters"] == ["B"]
    assert conflict["_label_audit"]["manual_letters"] is None
    assert audit["label_status"]["conflict"] == 1
    assert audit["label_status"]["agree"] == 2  # one is still unsafe due to format
    assert audit["issues"]["single_format_with_multiple_answers"] == 1


def test_mapped_answer_text_policy_can_emit_corrected_conflict():
    rows = [_public_record("conflict", declared=("A",))]
    safe, conflicts, _ = audit_public_records(
        rows,
        PipelineConfig(
            fuzzy_story_threshold=1.0,
            safe_dev_policy="mapped_answer_text",
        ),
    )

    assert not conflicts
    assert safe[0]["original_correct_letters"] == ["A"]
    assert safe[0]["correct_letters"] == ["B"]
    assert safe[0]["evaluation_label_source"] == "exact_answer_text_mapping"


def test_cli_end_to_end_writes_audited_outputs(tmp_path):
    train_path = tmp_path / "train.jsonl"
    public_path = tmp_path / "public.jsonl"
    output_dir = tmp_path / "out"
    _write_jsonl(
        train_path,
        [
            _train_record("a", "甲告诉乙消息。", dimension="1.1.1"),
            _train_record("b", "丙告诉丁消息。", dimension="1.1.2"),
            _train_record("c", "戊告诉己消息。", dimension="1.1.1"),
            _train_record("d", "庚告诉辛消息。", dimension="1.1.2"),
        ],
    )
    _write_jsonl(public_path, [_public_record("public")])

    result = cli_main(
        [
            "--train",
            str(train_path),
            "--public-test",
            str(public_path),
            "--output-dir",
            str(output_dir),
            "--fuzzy-story-threshold",
            "1.0",
            "--balance-target",
            "2",
        ]
    )

    assert result == 0
    expected = {
        "train_clean_all.jsonl",
        "train.jsonl",
        "dev.jsonl",
        "train_balanced.jsonl",
        "train_rejected.jsonl",
        "public_safe_dev.jsonl",
        "public_label_conflicts.jsonl",
        "audit.json",
    }
    assert expected == {path.name for path in output_dir.iterdir()}
    audit = json.loads((output_dir / "audit.json").read_text(encoding="utf-8"))
    assert audit["invariants"]["train_dev_cluster_overlap"] == 0
    assert audit["invariants"]["clean_answers_with_noncanonical_box_count"] == 0
    assert len(read_jsonl(output_dir / "public_safe_dev.jsonl")) == 1

    with pytest.raises(FileExistsError):
        cli_main(
            [
                "--train",
                str(train_path),
                "--public-test",
                str(public_path),
                "--output-dir",
                str(output_dir),
            ]
        )
