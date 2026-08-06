"""Evaluate an OpenAI-compatible vLLM endpoint on SoMBench JSONL.

The request defaults intentionally match the published unified protocol:
``protocol=cot, temperature=0.6, top_p=0.95, max_tokens=8192, n=1``.
Predictions are checkpointed after every response and can be resumed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import statistics
import time
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .config import apply_overrides, deep_merge, load_config, redact_config, require_keys, resolve_path
from .data import iter_jsonl

LOGGER = logging.getLogger("socialmind.evaluate")

DEFAULT_MULTI_SYSTEM_PROMPT = (
    "你是一个认真阅读的人,正在回答一道关于心理状态(theory-of-mind)的多选题。"
    "请一步步推理角色的心理状态,然后输出最终答案:把所有正确选项的字母放进同一个 "
    "\\boxed{} 中,用英文逗号分隔,例如 \\boxed{A,C}。把最终的 \\boxed{} 放在最后一行。"
)
DEFAULT_SINGLE_SYSTEM_PROMPT = (
    "你是一个认真阅读的人,正在回答一道关于心理状态(theory-of-mind)的单选题。"
    "请一步步推理角色的心理状态,然后输出最终答案,格式为 \\boxed{X},"
    "其中 X 是唯一最合适选项的字母。把最终的 \\boxed{X} 放在最后一行。"
)

DEFAULTS: dict[str, Any] = {
    "api": {
        "base_url": "http://127.0.0.1:8000/v1",
        "api_key_env": "OPENAI_API_KEY",
        "api_key_fallback": "EMPTY",
        "timeout_seconds": 900,
        "workers": 4,
        "max_retries": 3,
        "retry_backoff_seconds": 2.0,
    },
    "generation": {
        "protocol": "cot",
        "temperature": 0.6,
        "top_p": 0.95,
        "max_tokens": 8192,
        "n": 1,
        "seeds": [17, 29, 43],
        "extra_body": {"chat_template_kwargs": {"enable_thinking": True}},
    },
    "prompt": {
        "single": DEFAULT_SINGLE_SYSTEM_PROMPT,
        "multi": DEFAULT_MULTI_SYSTEM_PROMPT,
    },
    "resume": True,
    "fail_fast": False,
    "validation": {},
}

UNKNOWN_DIMENSION = "unknown"
PRIMARY_DIMENSION_NAMES = {
    "1": "mentalization",
    "2": "strategic_social_interaction",
    "3": "sociocultural_norms",
}


@dataclass(frozen=True)
class GoldAudit:
    reported: tuple[str, ...]
    mapped: tuple[str, ...] | None
    safe: bool
    reason: str


@dataclass(frozen=True)
class PredictionLogState:
    """Canonical successful rows plus an audit of append-only attempts."""

    completed: dict[tuple[int, str], dict[str, Any]]
    total_log_rows: int
    successful_log_rows: int
    failed_log_rows: int
    duplicate_success_keys: dict[str, int]


def normalize_letters(value: Any, *, allowed: str = "ABCDEF") -> tuple[str, ...]:
    """Canonicalize an answer into an ordered, duplicate-free letter tuple."""

    if value is None:
        return ()
    if isinstance(value, (list, tuple, set)):
        combined: list[str] = []
        for item in value:
            combined.extend(normalize_letters(item, allowed=allowed))
        return tuple(letter for letter in allowed if letter in set(combined))
    text = unicodedata.normalize("NFKC", str(value)).upper().strip()
    # Remove common LaTeX wrappers before letter extraction; otherwise the E in
    # ``\\text`` would look like a selected option.
    previous = None
    while previous != text:
        previous = text
        text = re.sub(r"\\(?:TEXT|MATHRM|MATHBF|OPERATORNAME)\s*\{([^{}]*)\}", r"\1", text)
    compact = re.sub(r"[\s,，、;；/|&和及与+]+", "", text)
    if compact and all(character in allowed for character in compact):
        found = set(compact)
    else:
        found = set(re.findall(rf"(?<![A-Z])[{re.escape(allowed)}](?![A-Z])", text))
    return tuple(letter for letter in allowed if letter in found)


def extract_last_boxed(text: str | None) -> tuple[str, ...] | None:
    """Extract letters from the final balanced ``\\boxed{...}`` occurrence."""

    if not text:
        return None
    boxes: list[str] = []
    for match in re.finditer(r"\\+boxed\s*", text, flags=re.IGNORECASE):
        cursor = match.end()
        if cursor >= len(text) or text[cursor] != "{":
            continue
        depth = 0
        for end in range(cursor, len(text)):
            character = text[end]
            if character == "{":
                depth += 1
            elif character == "}":
                depth -= 1
                if depth == 0:
                    boxes.append(text[cursor + 1 : end])
                    break
    if not boxes:
        return None
    letters = normalize_letters(boxes[-1])
    return letters or None


def _normalize_option_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value))
    text = re.sub(r"\s+", "", text)
    text = text.translate(str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'", "：": ":"}))
    return text.strip("。.;； ")


def map_answer_texts(options: Mapping[str, Any], answers: Any) -> tuple[str, ...] | None:
    """Map exact normalized answer text back to option letters, or return None."""

    if not isinstance(answers, list) or not answers:
        return None
    reverse: dict[str, list[str]] = defaultdict(list)
    for letter, text in options.items():
        reverse[_normalize_option_text(text)].append(str(letter).upper())
    mapped: list[str] = []
    for answer in answers:
        candidates = reverse.get(_normalize_option_text(answer), [])
        if len(candidates) != 1:
            return None
        mapped.append(candidates[0])
    return normalize_letters(mapped)


def audit_gold(record: Mapping[str, Any]) -> GoldAudit:
    reported = normalize_letters(record.get("correct_letters") or record.get("gold_letters"))
    options = record.get("options")
    if not isinstance(options, Mapping) or not isinstance(record.get("correct_answers"), list):
        # Internal held-out training/dev rows carry gold_letters but deliberately
        # do not duplicate labels as answer text. They remain scoreable against
        # reported letters without being mislabeled as public-data corruption.
        return GoldAudit(reported, None, False, "answer_text_not_provided")
    mapped = map_answer_texts(options, record.get("correct_answers")) if isinstance(options, Mapping) else None
    if mapped is None:
        return GoldAudit(reported, None, False, "answer_text_unmapped")
    if mapped != reported:
        return GoldAudit(reported, mapped, False, "letter_text_conflict")
    task_format = str(record.get("task_format") or record.get("prompt_type") or "").lower()
    if task_format == "mcq_single" and len(reported) != 1:
        return GoldAudit(reported, mapped, False, "single_format_with_multiple_answers")
    return GoldAudit(reported, mapped, True, "letter_text_agree")


def format_options(options: Mapping[str, Any]) -> str:
    def key(item: tuple[str, Any]) -> tuple[int, str]:
        letter = str(item[0]).upper()
        return (ord(letter[0]) if letter else 999, letter)

    return "\n".join(f"{str(letter).upper()}. {text}" for letter, text in sorted(options.items(), key=key))


def select_system_prompt(record: Mapping[str, Any], prompt: str | Mapping[str, Any]) -> str:
    """Select the reference-training single/multi prompt from task_format."""

    if isinstance(prompt, str):
        return prompt
    task_format = str(record.get("task_format") or record.get("prompt_type") or "").lower()
    key = "single" if task_format == "mcq_single" else "multi"
    selected = prompt.get(key)
    if not isinstance(selected, str) or not selected.strip():
        raise ValueError(f"prompt.{key} must be a non-empty string")
    return selected


def build_messages(
    record: Mapping[str, Any],
    system_prompt: str | Mapping[str, Any] = DEFAULT_MULTI_SYSTEM_PROMPT,
) -> list[dict[str, str]]:
    """Create the no-label evaluation prompt accepted by chat-completions."""

    selected_prompt = select_system_prompt(record, system_prompt)

    if isinstance(record.get("messages"), list):
        # Always install the selected reference prompt so the manifest describes
        # the prompt that was actually sent. Assistant and stale system messages
        # from teacher-forced JSONL are never sent to the inference endpoint.
        messages = [
            {"role": str(message["role"]), "content": str(message["content"])}
            for message in record["messages"]
            if isinstance(message, Mapping) and message.get("role") not in {"assistant", "system"}
        ]
        return [{"role": "system", "content": selected_prompt}, *messages]
    for field in ("story", "question", "options"):
        if not record.get(field):
            raise ValueError(f"evaluation record {record.get('sample_id', '<unknown>')} has no {field}")
    options = record["options"]
    if not isinstance(options, Mapping):
        raise ValueError("record.options must be an object keyed by answer letter")
    user = (
        f"请根据下面的故事回答问题。\n\n故事：\n{record['story']}\n\n"
        f"问题：\n{record['question']}\n\n选项：\n{format_options(options)}"
    )
    return [{"role": "system", "content": selected_prompt}, {"role": "user", "content": user}]


def record_qtype(record: Mapping[str, Any]) -> str:
    meta = record.get("meta") if isinstance(record.get("meta"), Mapping) else {}
    return str(record.get("question_type") or meta.get("qtype") or UNKNOWN_DIMENSION).upper()


def record_dimension(record: Mapping[str, Any]) -> str:
    """Return an explicit dimension only; never infer one for public raw rows."""

    meta = record.get("meta") if isinstance(record.get("meta"), Mapping) else {}
    value = record.get("dimension") or meta.get("dim")
    return str(value).strip() if value not in (None, "") else UNKNOWN_DIMENSION


def primary_dimension(dimension: Any) -> str:
    value = str(dimension or UNKNOWN_DIMENSION).strip()
    prefix = value.split(".", 1)[0]
    return prefix if prefix in PRIMARY_DIMENSION_NAMES else UNKNOWN_DIMENSION


def _metric_block(rows: Sequence[Mapping[str, Any]], gold_key: str = "gold_letters") -> dict[str, Any]:
    total = len(rows)
    parsed = sum(row.get("predicted_letters") is not None for row in rows)
    correct = sum(
        row.get("predicted_letters") is not None
        and tuple(row["predicted_letters"]) == tuple(row.get(gold_key) or ())
        for row in rows
    )
    return {
        "total": total,
        "correct": correct,
        "parsed": parsed,
        "accuracy": correct / total if total else None,
        "parse_rate": parsed / total if total else None,
    }


def _diagnostic_block(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Keep incompatible public-label interpretations visibly separate."""

    safe = [row for row in rows if row.get("safe_gold")]
    mapped = [row for row in rows if row.get("mapped_letters") is not None]
    return {
        "reported_letters": _metric_block(rows),
        "safe_agreement": _metric_block(safe),
        "mapped_answer_text": _metric_block(mapped, gold_key="mapped_letters"),
    }


def dimension_coverage(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    known = [row for row in rows if str(row.get("dimension") or UNKNOWN_DIMENSION) != UNKNOWN_DIMENSION]
    dimensions = sorted({str(row["dimension"]) for row in known})
    primaries = sorted({primary_dimension(row["dimension"]) for row in known})
    return {
        "total": len(rows),
        "known": len(known),
        "unknown": len(rows) - len(known),
        "coverage": len(known) / len(rows) if rows else None,
        "fine_dimension_count": len(dimensions),
        "fine_dimensions": dimensions,
        "primary_dimension_count": len([item for item in primaries if item != UNKNOWN_DIMENSION]),
        "primary_dimensions": [item for item in primaries if item != UNKNOWN_DIMENSION],
        "note": (
            "No dimension field was supplied; rows are retained and reported as unknown, not inferred."
            if rows and not known
            else None
        ),
    }


def summarize_seed(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    qtypes: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    dimensions: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    primaries: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        qtypes[str(row.get("qtype") or UNKNOWN_DIMENSION)].append(row)
        dimension = str(row.get("dimension") or UNKNOWN_DIMENSION)
        if dimension != UNKNOWN_DIMENSION:
            dimensions[dimension].append(row)
            primary = primary_dimension(dimension)
            if primary != UNKNOWN_DIMENSION:
                primaries[primary].append(row)
    safe = [row for row in rows if row.get("safe_gold")]
    mapped = [row for row in rows if row.get("mapped_letters") is not None]
    recommended = "safe_agreement_subset" if safe else "reported_letter_overall"
    audit_counts = Counter(str(row.get("gold_audit_reason") or "unknown") for row in rows)
    return {
        "metric_policy": {
            "recommended_model_diagnostic": recommended,
            "reported_letters_include_rows_without_safe_text_agreement": bool(rows) and len(safe) != len(rows),
            "known_letter_text_conflict_count": sum(
                row.get("gold_audit_reason") == "letter_text_conflict" for row in rows
            ),
            "safe_agreement_excludes_all_letter_text_conflicts": True,
        },
        "reported_letter_overall": _metric_block(rows),
        "safe_agreement_subset": _metric_block(safe),
        "mapped_answer_text_available": _metric_block(mapped, gold_key="mapped_letters"),
        "by_qtype": {qtype: _diagnostic_block(group) for qtype, group in sorted(qtypes.items())},
        "by_primary_dimension": {
            f"{dimension}:{PRIMARY_DIMENSION_NAMES[dimension]}": _diagnostic_block(group)
            for dimension, group in sorted(primaries.items())
        },
        "by_dimension": {
            dimension: _diagnostic_block(group) for dimension, group in sorted(dimensions.items())
        },
        "dimension_coverage": dimension_coverage(rows),
        "label_audit": {"safe": len(safe), "mapped": len(mapped), **dict(sorted(audit_counts.items()))},
    }


def _summary(values: Sequence[float | None]) -> dict[str, float | None]:
    usable = [float(value) for value in values if value is not None]
    if not usable:
        return {"mean": None, "std": None, "min": None, "max": None}
    return {
        "mean": statistics.fmean(usable),
        "std": statistics.pstdev(usable),
        "min": min(usable),
        "max": max(usable),
    }


def _aggregate_diagnostics(
    per_seed: Mapping[str, Mapping[str, Any]],
    seeds: Sequence[int],
    path: Sequence[str],
) -> dict[str, Any]:
    def value(seed: int, label: str) -> float | None:
        node: Any = per_seed[str(seed)]
        for key in path:
            node = node.get(key, {}) if isinstance(node, Mapping) else {}
        metric = node.get(label, {}) if isinstance(node, Mapping) else {}
        return metric.get("accuracy") if isinstance(metric, Mapping) else None

    return {
        "reported_letters": _summary([value(seed, "reported_letters") for seed in seeds]),
        "safe_agreement": _summary([value(seed, "safe_agreement") for seed in seeds]),
        "mapped_answer_text": _summary([value(seed, "mapped_answer_text") for seed in seeds]),
    }


def build_sample_stability(
    rows: Sequence[Mapping[str, Any]],
    seeds: Sequence[int],
    *,
    sample_order: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Create one deterministic majority/stability record per sample."""

    expected_seeds = [int(seed) for seed in seeds]
    grouped: dict[str, dict[int, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        sample_id = str(row["sample_id"])
        seed = int(row["seed"])
        if seed in grouped[sample_id]:
            raise ValueError(f"duplicate canonical prediction for seed={seed}, sample_id={sample_id}")
        grouped[sample_id][seed] = row
    ordered_ids = list(sample_order) if sample_order is not None else sorted(grouped)
    extras = sorted(set(grouped) - set(ordered_ids))
    if extras:
        raise ValueError(f"predictions contain sample ids outside sample_order: {extras[:10]}")

    output: list[dict[str, Any]] = []
    for sample_id in ordered_ids:
        seed_rows = grouped.get(sample_id, {})
        first = next(iter(seed_rows.values()), {})
        predictions = {
            str(seed): (
                list(seed_rows[seed]["predicted_letters"])
                if seed in seed_rows and seed_rows[seed].get("predicted_letters") is not None
                else None
            )
            for seed in expected_seeds
        }
        missing_seeds = [seed for seed in expected_seeds if seed not in seed_rows]
        counts = Counter(
            tuple(predictions[str(seed)]) if predictions[str(seed)] is not None else None
            for seed in expected_seeds
            if seed in seed_rows
        )
        ordered = counts.most_common()
        tie = bool(ordered and len(ordered) > 1 and ordered[0][1] == ordered[1][1])
        winner = ordered[0][0] if ordered and not tie and not missing_seeds else None
        unique_predictions = len(counts)
        reported = list(first.get("gold_letters") or [])
        mapped = first.get("mapped_letters")
        safe = bool(first.get("safe_gold"))
        output.append(
            {
                "sample_id": sample_id,
                "qtype": first.get("qtype"),
                "dimension": first.get("dimension", UNKNOWN_DIMENSION),
                "primary_dimension": primary_dimension(first.get("dimension")),
                "predictions_by_seed": predictions,
                "missing_seeds": missing_seeds,
                "complete": not missing_seeds,
                "unique_prediction_count": unique_predictions,
                "all_seed_consistent": not missing_seeds and unique_predictions == 1,
                "majority_tie": tie,
                "majority_prediction": list(winner) if winner is not None else None,
                "gold_letters": reported,
                "mapped_letters": list(mapped) if mapped is not None else None,
                "safe_gold": safe,
                "gold_audit_reason": first.get("gold_audit_reason"),
                "majority_correct_reported": list(winner) == reported if winner is not None else False,
                "majority_correct_safe": (list(winner) == reported if winner is not None else False) if safe else None,
                "majority_correct_mapped": (
                    list(winner) == list(mapped) if winner is not None else False
                ) if mapped is not None else None,
            }
        )
    return output


def summarize_multi_seed(
    rows: Sequence[Mapping[str, Any]],
    seeds: Sequence[int],
    *,
    sample_order: Sequence[str] | None = None,
) -> dict[str, Any]:
    by_seed_rows: dict[int, list[Mapping[str, Any]]] = {int(seed): [] for seed in seeds}
    for row in rows:
        by_seed_rows.setdefault(int(row["seed"]), []).append(row)
    per_seed = {str(seed): summarize_seed(by_seed_rows.get(seed, [])) for seed in seeds}
    qtypes = sorted({str(row.get("qtype") or UNKNOWN_DIMENSION) for row in rows})
    dimensions = sorted(
        {str(row.get("dimension")) for row in rows if str(row.get("dimension") or UNKNOWN_DIMENSION) != UNKNOWN_DIMENSION}
    )
    primary_keys = sorted(
        {
            f"{primary_dimension(row.get('dimension'))}:{PRIMARY_DIMENSION_NAMES[primary_dimension(row.get('dimension'))]}"
            for row in rows
            if primary_dimension(row.get("dimension")) != UNKNOWN_DIMENSION
        }
    )
    aggregate = {
        "reported_letter_accuracy": _summary(
            [per_seed[str(seed)]["reported_letter_overall"]["accuracy"] for seed in seeds]
        ),
        "safe_agreement_accuracy": _summary(
            [per_seed[str(seed)]["safe_agreement_subset"]["accuracy"] for seed in seeds]
        ),
        "mapped_answer_text_accuracy": _summary(
            [per_seed[str(seed)]["mapped_answer_text_available"]["accuracy"] for seed in seeds]
        ),
        "parse_rate": _summary(
            [per_seed[str(seed)]["reported_letter_overall"]["parse_rate"] for seed in seeds]
        ),
        "by_qtype": {
            qtype: _aggregate_diagnostics(per_seed, seeds, ("by_qtype", qtype)) for qtype in qtypes
        },
        "by_primary_dimension": {
            primary: _aggregate_diagnostics(per_seed, seeds, ("by_primary_dimension", primary))
            for primary in primary_keys
        },
        "by_dimension": {
            dimension: _aggregate_diagnostics(per_seed, seeds, ("by_dimension", dimension))
            for dimension in dimensions
        },
        "dimension_coverage": (
            per_seed[str(seeds[0])]["dimension_coverage"] if seeds else dimension_coverage([])
        ),
    }
    per_sample = build_sample_stability(rows, seeds, sample_order=sample_order)
    complete = [item for item in per_sample if item["complete"]]
    majority_rows = [
        {
            "sample_id": item["sample_id"],
            "qtype": item["qtype"],
            "dimension": item["dimension"],
            "predicted_letters": item["majority_prediction"],
            "gold_letters": item["gold_letters"],
            "mapped_letters": item["mapped_letters"],
            "safe_gold": item["safe_gold"],
            "gold_audit_reason": item["gold_audit_reason"],
        }
        for item in complete
    ]
    stability = {
        "complete_samples": len(complete),
        "incomplete_samples": len(per_sample) - len(complete),
        "all_seed_consistency_rate": (
            sum(item["all_seed_consistent"] for item in complete) / len(complete) if complete else None
        ),
        "majority_tie_count": sum(item["majority_tie"] for item in complete),
        "majority_vote": summarize_seed(majority_rows),
    }
    return {"per_seed": per_seed, "aggregate": aggregate, "stability": stability}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _write_jsonl_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def load_prediction_log(path: Path) -> PredictionLogState:
    """Load append-only attempts, selecting the last success for each key."""

    completed: dict[tuple[int, str], dict[str, Any]] = {}
    if not path.is_file():
        return PredictionLogState({}, 0, 0, 0, {})
    success_counts: Counter[tuple[int, str]] = Counter()
    total = failed = 0
    for row in iter_jsonl(path):
        total += 1
        if "seed" not in row or "sample_id" not in row:
            raise ValueError(f"prediction row lacks seed/sample_id in {path}")
        key = (int(row["seed"]), str(row["sample_id"]))
        if not row.get("error"):
            success_counts[key] += 1
            completed[key] = row
        else:
            failed += 1
    duplicates = {
        f"{seed}:{sample_id}": count
        for (seed, sample_id), count in sorted(success_counts.items())
        if count > 1
    }
    return PredictionLogState(
        completed=completed,
        total_log_rows=total,
        successful_log_rows=sum(success_counts.values()),
        failed_log_rows=failed,
        duplicate_success_keys=duplicates,
    )


def _load_completed(path: Path) -> dict[tuple[int, str], dict[str, Any]]:
    """Backward-compatible shorthand used by older local scripts."""

    return load_prediction_log(path).completed


def audit_dataset(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    gold_audits = [audit_gold(record) for record in records]
    qtypes = Counter(record_qtype(record) for record in records)
    task_formats = Counter(str(record.get("task_format") or record.get("prompt_type") or "unknown") for record in records)
    dimensions = [record_dimension(record) for record in records]
    known_dimensions = [item for item in dimensions if item != UNKNOWN_DIMENSION]
    reasons = Counter(audit.reason for audit in gold_audits)
    return {
        "records": len(records),
        "safe_records": sum(audit.safe for audit in gold_audits),
        "mapped_answer_text_records": sum(audit.mapped is not None for audit in gold_audits),
        "qtypes": dict(sorted(qtypes.items())),
        "task_formats": dict(sorted(task_formats.items())),
        "label_audit": dict(sorted(reasons.items())),
        "dimension_coverage": {
            "known": len(known_dimensions),
            "unknown": len(dimensions) - len(known_dimensions),
            "coverage": len(known_dimensions) / len(dimensions) if dimensions else None,
            "fine_dimension_count": len(set(known_dimensions)),
            "primary_dimension_count": len(
                {primary_dimension(item) for item in known_dimensions if primary_dimension(item) != UNKNOWN_DIMENSION}
            ),
            "note": (
                "Source records have no dimension metadata; no dimension was inferred."
                if dimensions and not known_dimensions
                else None
            ),
        },
    }


def validate_dataset_expectations(audit: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    checks = {
        "records": expected.get("expected_records"),
        "safe_records": expected.get("expected_safe_records"),
    }
    failures: list[str] = []
    for field, wanted in checks.items():
        if wanted is not None and audit.get(field) != wanted:
            failures.append(f"{field}: observed={audit.get(field)!r}, expected={wanted!r}")
    wanted_qtypes = expected.get("expected_qtype_counts")
    if wanted_qtypes is not None and dict(audit.get("qtypes") or {}) != dict(wanted_qtypes):
        failures.append(f"qtypes: observed={audit.get('qtypes')!r}, expected={wanted_qtypes!r}")
    wanted_formats = expected.get("expected_task_format_counts")
    if wanted_formats is not None and dict(audit.get("task_formats") or {}) != dict(wanted_formats):
        failures.append(
            f"task_formats: observed={audit.get('task_formats')!r}, expected={wanted_formats!r}"
        )
    wanted_label_audit = expected.get("expected_label_audit")
    if wanted_label_audit is not None and dict(audit.get("label_audit") or {}) != dict(wanted_label_audit):
        failures.append(
            f"label_audit: observed={audit.get('label_audit')!r}, expected={wanted_label_audit!r}"
        )
    wanted_dimension_count = expected.get("expected_fine_dimensions")
    observed_dimension_count = (audit.get("dimension_coverage") or {}).get("fine_dimension_count")
    if wanted_dimension_count is not None and observed_dimension_count != wanted_dimension_count:
        failures.append(
            f"fine_dimension_count: observed={observed_dimension_count!r}, expected={wanted_dimension_count!r}"
        )
    wanted_coverage = expected.get("expected_dimension_coverage")
    observed_coverage = (audit.get("dimension_coverage") or {}).get("coverage")
    if wanted_coverage is not None and observed_coverage != wanted_coverage:
        failures.append(f"dimension_coverage: observed={observed_coverage!r}, expected={wanted_coverage!r}")
    if failures:
        raise ValueError("dataset validation failed: " + "; ".join(failures))


def _request_one(client: Any, config: Mapping[str, Any], record: Mapping[str, Any], seed: int) -> dict[str, Any]:
    generation = config["generation"]
    max_retries = int(config["api"].get("max_retries", 3))
    started = time.monotonic()
    error: str | None = None
    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=config["model"],
                messages=build_messages(record, config["prompt"]),
                temperature=float(generation["temperature"]),
                top_p=float(generation["top_p"]),
                max_tokens=int(generation["max_tokens"]),
                n=1,
                seed=int(seed),
                extra_body=dict(generation.get("extra_body") or {}),
            )
            message = response.choices[0].message
            content = message.content or ""
            reasoning = getattr(message, "reasoning_content", None)
            usage = getattr(response, "usage", None)
            audit = audit_gold(record)
            predicted = extract_last_boxed(content)
            return {
                "sample_id": str(record.get("sample_id")),
                "seed": int(seed),
                "qtype": record_qtype(record),
                "dimension": record_dimension(record),
                "primary_dimension": primary_dimension(record_dimension(record)),
                "predicted_letters": list(predicted) if predicted is not None else None,
                "gold_letters": list(audit.reported),
                "mapped_letters": list(audit.mapped) if audit.mapped is not None else None,
                "safe_gold": audit.safe,
                "gold_audit_reason": audit.reason,
                "content": content,
                "reasoning_content": reasoning,
                "finish_reason": response.choices[0].finish_reason,
                "usage": usage.model_dump() if usage is not None and hasattr(usage, "model_dump") else None,
                "latency_seconds": time.monotonic() - started,
                "error": None,
            }
        except Exception as exc:  # API error types vary across openai versions
            error = f"{type(exc).__name__}: {exc}"
            if attempt < max_retries:
                time.sleep(float(config["api"].get("retry_backoff_seconds", 2.0)) * (2**attempt))
    audit = audit_gold(record)
    return {
        "sample_id": str(record.get("sample_id")),
        "seed": int(seed),
        "qtype": record_qtype(record),
        "dimension": record_dimension(record),
        "primary_dimension": primary_dimension(record_dimension(record)),
        "predicted_letters": None,
        "gold_letters": list(audit.reported),
        "mapped_letters": list(audit.mapped) if audit.mapped is not None else None,
        "safe_gold": audit.safe,
        "gold_audit_reason": audit.reason,
        "content": "",
        "reasoning_content": None,
        "finish_reason": None,
        "usage": None,
        "latency_seconds": time.monotonic() - started,
        "error": error,
    }


def run(config: Mapping[str, Any]) -> None:
    require_keys(config, ["model", "dataset", "output_dir"])
    generation = config["generation"]
    expected = {"protocol": "cot", "temperature": 0.6, "top_p": 0.95, "max_tokens": 8192, "n": 1}
    mismatches = {key: (generation.get(key), value) for key, value in expected.items() if generation.get(key) != value}
    if mismatches:
        raise ValueError(f"generation settings diverge from the official protocol: {mismatches}")
    extra_body = generation.get("extra_body") or {}
    forbidden_extra = sorted(
        set(extra_body) & {"protocol", "temperature", "top_p", "max_tokens", "n", "seed"}
    )
    if forbidden_extra:
        raise ValueError(f"generation.extra_body may not override fixed sampling keys: {forbidden_extra}")
    seeds = [int(seed) for seed in generation.get("seeds", [])]
    if len(seeds) != 3 or len(seeds) != len(set(seeds)):
        raise ValueError("generation.seeds must contain exactly three unique integers")

    dataset_path = resolve_path(config["dataset"])
    records = list(iter_jsonl(dataset_path))
    sample_ids = [str(record.get("sample_id")) for record in records]
    if any(sample_id in {"", "None"} for sample_id in sample_ids):
        raise ValueError("every evaluation record must have sample_id")
    duplicates = [item for item, count in Counter(sample_ids).items() if count > 1]
    if duplicates:
        raise ValueError(f"duplicate sample_id values: {duplicates[:10]}")
    dataset_audit = audit_dataset(records)
    validate_dataset_expectations(dataset_audit, config.get("validation") or {})

    output_dir = resolve_path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.jsonl"
    manifest_path = output_dir / "evaluation_manifest.json"
    manifest = {
        "model": config["model"],
        "dataset": str(dataset_path),
        "dataset_sha256": _sha256(dataset_path),
        "record_count": len(records),
        "dataset_audit": dataset_audit,
        "generation": generation,
        "prompt": config["prompt"],
        "config": redact_config(config),
    }
    if predictions_path.exists() and not manifest_path.is_file():
        raise ValueError(
            f"cannot trust or resume {predictions_path} without evaluation_manifest.json; choose a new output_dir"
        )
    if manifest_path.is_file() and bool(config.get("resume", True)):
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        keys = ("model", "dataset_sha256", "generation", "prompt")
        changed = [key for key in keys if previous.get(key) != manifest.get(key)]
        if changed:
            raise ValueError(f"cannot resume because manifest fields changed: {changed}; choose a new output_dir")
    elif predictions_path.exists() and not bool(config.get("resume", True)):
        raise FileExistsError(f"predictions already exist: {predictions_path}; choose a new output_dir or resume=true")
    _write_json(manifest_path, manifest)
    _write_json(output_dir / "dataset_audit.json", dataset_audit)

    expected_keys = {(seed, sample_id) for seed in seeds for sample_id in sample_ids}
    initial_log = load_prediction_log(predictions_path) if bool(config.get("resume", True)) else PredictionLogState({}, 0, 0, 0, {})
    unexpected = sorted(set(initial_log.completed) - expected_keys)
    if unexpected:
        raise ValueError(f"prediction log contains keys outside this dataset/seed manifest: {unexpected[:10]}")
    completed = initial_log.completed
    jobs = [(record, seed) for seed in seeds for record in records if (seed, str(record["sample_id"])) not in completed]
    LOGGER.info("evaluation jobs: %d pending, %d already complete", len(jobs), len(completed))

    if jobs:
        from openai import OpenAI

        api_cfg = config["api"]
        api_key = os.environ.get(str(api_cfg.get("api_key_env", "OPENAI_API_KEY"))) or str(
            api_cfg.get("api_key_fallback", "EMPTY")
        )
        client = OpenAI(
            base_url=str(api_cfg["base_url"]),
            api_key=api_key,
            timeout=float(api_cfg.get("timeout_seconds", 900)),
            max_retries=0,
        )
        output_mode = "a" if predictions_path.exists() else "w"
        with predictions_path.open(output_mode, encoding="utf-8") as sink:
            with ThreadPoolExecutor(max_workers=max(1, int(api_cfg.get("workers", 4)))) as pool:
                future_to_key = {
                    pool.submit(_request_one, client, config, record, seed): (seed, str(record["sample_id"]))
                    for record, seed in jobs
                }
                for index, future in enumerate(as_completed(future_to_key), start=1):
                    expected_key = future_to_key[future]
                    row = future.result()
                    observed_key = (int(row["seed"]), str(row["sample_id"]))
                    if observed_key != expected_key:
                        raise RuntimeError(
                            f"worker returned prediction key {observed_key}, expected {expected_key}"
                        )
                    sink.write(json.dumps(row, ensure_ascii=False) + "\n")
                    sink.flush()
                    if row.get("error"):
                        LOGGER.error("request failed for seed=%s sample=%s: %s", row["seed"], row["sample_id"], row["error"])
                        if bool(config.get("fail_fast", False)):
                            raise RuntimeError(row["error"])
                    if index % 10 == 0 or index == len(jobs):
                        LOGGER.info("completed %d/%d pending requests", index, len(jobs))

    # Keep the append-only attempt log for failure audit, while writing a second
    # canonical file with exactly one successful row per expected key.
    final_log = load_prediction_log(predictions_path)
    unexpected = sorted(set(final_log.completed) - expected_keys)
    if unexpected:
        raise RuntimeError(f"prediction log gained unexpected keys: {unexpected[:10]}")
    missing_keys = expected_keys - set(final_log.completed)
    final_rows = [
        final_log.completed[(seed, sample_id)]
        for seed in seeds
        for sample_id in sample_ids
        if (seed, sample_id) in final_log.completed
    ]
    _write_jsonl_atomic(output_dir / "predictions_canonical.jsonl", final_rows)
    sample_stability = build_sample_stability(final_rows, seeds, sample_order=sample_ids)
    _write_jsonl_atomic(output_dir / "sample_stability.jsonl", sample_stability)
    metrics = summarize_multi_seed(final_rows, seeds, sample_order=sample_ids)
    metrics["run"] = {
        "expected_predictions": len(records) * len(seeds),
        "completed_predictions": len(final_rows),
        "missing_or_failed": len(missing_keys),
        "missing_keys": [f"{seed}:{sample_id}" for seed, sample_id in sorted(missing_keys)],
        "raw_attempt_log_rows": final_log.total_log_rows,
        "raw_success_rows": final_log.successful_log_rows,
        "raw_failed_rows": final_log.failed_log_rows,
        "duplicate_success_keys": final_log.duplicate_success_keys,
        "canonical_prediction_rows": len(final_rows),
        "sample_stability_rows": len(sample_stability),
        "all_records_retained": len(sample_stability) == len(records),
    }
    _write_json(output_dir / "metrics.json", metrics)
    LOGGER.info("metrics written to %s", output_dir / "metrics.json")
    if missing_keys:
        raise RuntimeError(f"evaluation incomplete: {len(missing_keys)} requests failed; rerun with resume=true")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config = deep_merge(DEFAULTS, load_config(args.config))
    config = apply_overrides(config, args.set)
    run(config)


if __name__ == "__main__":
    main()
