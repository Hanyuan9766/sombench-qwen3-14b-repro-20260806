"""Build a deterministic Stage-3 calibration mix from training data only.

The sampler deliberately has no public-test input.  It combines a
fine-grained-dimension-balanced replay slice with two calibration slices:

* low-cardinality Q2 examples, emphasizing single-letter gold answers; and
* Q3 examples whose gold option explicitly means ``uncertain``.

Q3 semantics are derived from the option text in each prompt, never from a
fixed answer letter.  The mapping is intentionally fail-closed: every Q3 row
must contain exactly one option for yes, no, and uncertain.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from .core import file_sha256, read_jsonl, write_json, write_jsonl


OPTION_TEXT_RE = re.compile(r"(?m)^\s*([A-Z])\s*[.．、)]\s*(.*?)\s*$")
Q3_SEMANTIC_TEXT = {
    "是": "yes",
    "否": "no",
    "无法确定": "uncertain",
    "证据不足": "uncertain",
    "信息不足": "uncertain",
    "无法判断": "uncertain",
}
SOURCE_BUCKETS = ("balanced_replay", "q2_low_cardinality", "q3_uncertain")


@dataclass(frozen=True)
class CalibrationConfig:
    """Controls the training-only calibration mixture."""

    total_rows: int = 2982
    replay_fraction: float = 0.70
    q2_fraction: float = 0.15
    q3_fraction: float = 0.15
    q2_single_share: float = 0.70
    seed: int = 42
    required_dimensions: int = 71

    def validate(self) -> None:
        if self.total_rows <= 0:
            raise ValueError("total_rows must be positive")
        fractions = (self.replay_fraction, self.q2_fraction, self.q3_fraction)
        if any(value <= 0.0 for value in fractions):
            raise ValueError("all source fractions must be positive")
        if not math.isclose(sum(fractions), 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("replay, Q2, and Q3 fractions must sum to 1")
        if not 0.0 < self.q2_single_share <= 1.0:
            raise ValueError("q2_single_share must be in (0, 1]")
        if self.required_dimensions <= 0:
            raise ValueError("required_dimensions must be positive")


def _stable_distribution(values: Sequence[Any]) -> Dict[str, int]:
    return dict(sorted(Counter(str(value) for value in values).items()))


def _quota_summary(quotas: Mapping[str, int]) -> Dict[str, Any]:
    values = list(quotas.values())
    return {
        "dimension_count": len(quotas),
        "rows": sum(values),
        "min_rows": min(values, default=0),
        "max_rows": max(values, default=0),
        "spread": max(values, default=0) - min(values, default=0),
        "distribution": dict(sorted(quotas.items())),
    }


def _question_type(record: Mapping[str, Any]) -> str:
    value = record.get("question_type")
    if value:
        return str(value)
    meta = record.get("meta")
    return str(meta.get("qtype", "")) if isinstance(meta, Mapping) else ""


def _dimension(record: Mapping[str, Any]) -> str:
    value = record.get("dimension")
    if value:
        return str(value)
    meta = record.get("meta")
    return str(meta.get("dim", "")) if isinstance(meta, Mapping) else ""


def _gold_letters(record: Mapping[str, Any]) -> List[str]:
    values = record.get("gold_letters")
    if not isinstance(values, list):
        return []
    return sorted(
        {
            str(value).strip().upper()
            for value in values
            if re.fullmatch(r"[A-Z]", str(value).strip().upper())
        }
    )


def _source_key(record: Mapping[str, Any]) -> str:
    sample_id = record.get("sample_id")
    pipeline = record.get("_pipeline")
    source_line = (
        pipeline.get("source_line")
        if isinstance(pipeline, Mapping) and pipeline.get("source_line") is not None
        else None
    )
    # The reference corpus contains repeated synthetic sample_id values, in
    # some cases attached to different rows/dimensions.  The cleaning pipeline
    # preserves the original source line in both train and balanced replay, so
    # their pair is the stable row identity.
    if sample_id is not None and str(sample_id) and source_line is not None:
        return f"sample_id:{sample_id}|source_line:{source_line}"
    if sample_id is not None and str(sample_id):
        return f"sample_id:{sample_id}"
    if source_line is not None:
        return f"source_line:{source_line}"
    payload = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "record_sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _last_message(record: Mapping[str, Any], role: str) -> str:
    messages = record.get("messages")
    if not isinstance(messages, list):
        return ""
    for message in reversed(messages):
        if isinstance(message, Mapping) and message.get("role") == role:
            content = message.get("content")
            return content if isinstance(content, str) else ""
    return ""


def _normalize_semantic_text(text: str) -> str:
    value = unicodedata.normalize("NFKC", text)
    value = re.sub(r"\s+", "", value)
    return value.strip("。.!！?？:：;；,，()（）[]【】")


def map_q3_gold_semantic(record: Mapping[str, Any]) -> Dict[str, Any]:
    """Map a Q3 gold letter to yes/no/uncertain using strict option text.

    This function raises instead of guessing if the prompt schema is incomplete
    or contains an option text outside the explicit semantic whitelist.
    """

    if _question_type(record) != "Q3":
        raise ValueError("map_q3_gold_semantic requires a Q3 record")
    sample = _source_key(record)
    prompt = _last_message(record, "user")
    options_section = prompt
    for marker in ("选项：", "选项:"):
        if marker in prompt:
            options_section = prompt.rsplit(marker, 1)[-1]
            break
    pairs = OPTION_TEXT_RE.findall(options_section)
    option_text: Dict[str, str] = {}
    for letter, text in pairs:
        if letter in option_text:
            raise ValueError(f"{sample}: duplicate Q3 option letter {letter}")
        option_text[letter] = text.strip()
    if len(option_text) != 3:
        raise ValueError(f"{sample}: expected exactly three Q3 options")

    option_semantics: Dict[str, str] = {}
    for letter, text in option_text.items():
        normalized = _normalize_semantic_text(text)
        semantic = Q3_SEMANTIC_TEXT.get(normalized)
        if semantic is None:
            raise ValueError(f"{sample}: unsupported Q3 option text {text!r}")
        option_semantics[letter] = semantic
    if set(option_semantics.values()) != {"yes", "no", "uncertain"}:
        raise ValueError(f"{sample}: Q3 options do not map one-to-one to yes/no/uncertain")

    gold = _gold_letters(record)
    if len(gold) != 1 or gold[0] not in option_semantics:
        raise ValueError(f"{sample}: Q3 requires one valid gold letter")
    gold_letter = gold[0]
    return {
        "gold_letter": gold_letter,
        "gold_option_text": option_text[gold_letter],
        "gold_semantic": option_semantics[gold_letter],
        "option_semantics": dict(sorted(option_semantics.items())),
    }


def _largest_remainder_budgets(config: CalibrationConfig) -> Dict[str, int]:
    fractions = {
        "balanced_replay": config.replay_fraction,
        "q2_low_cardinality": config.q2_fraction,
        "q3_uncertain": config.q3_fraction,
    }
    raw = {key: config.total_rows * value for key, value in fractions.items()}
    budgets = {key: math.floor(value) for key, value in raw.items()}
    remaining = config.total_rows - sum(budgets.values())
    order = sorted(fractions, key=lambda key: (-(raw[key] - budgets[key]), key))
    for key in order[:remaining]:
        budgets[key] += 1
    return budgets


def _cyclic_sample(
    records: Sequence[Mapping[str, Any]], count: int, *, seed: int, label: str
) -> List[Mapping[str, Any]]:
    if count <= 0:
        return []
    if not records:
        raise ValueError(f"no candidates for non-empty calibration bucket {label}")
    selected: List[Mapping[str, Any]] = []
    cycle = 0
    while len(selected) < count:
        indices = list(range(len(records)))
        random.Random(f"{seed}:{label}:{cycle}").shuffle(indices)
        for index in indices:
            selected.append(records[index])
            if len(selected) == count:
                break
        cycle += 1
    return selected


def _capacity_balanced_sample(
    records: Sequence[Mapping[str, Any]],
    count: int,
    *,
    seed: int,
    label: str,
    max_occurrences_per_source: int,
    shared_dimension_counts: Optional[MutableMapping[str, int]] = None,
    baseline_dimension_counts: Optional[Mapping[str, int]] = None,
) -> Tuple[List[Mapping[str, Any]], Dict[str, int]]:
    """Sample near-uniformly across candidate dimensions with repeat caps.

    ``shared_dimension_counts`` lets the two Q2 cardinality strata optimize the
    combined Q2 bucket instead of independently doubling the same dimensions.
    Sparse dimensions are allowed to saturate their source-repeat capacity;
    the remaining budget is then water-filled across dimensions with capacity.
    """

    if count <= 0:
        return [], {}
    if max_occurrences_per_source <= 0:
        raise ValueError("max_occurrences_per_source must be positive")
    by_dimension: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        dimension = _dimension(record)
        if not dimension:
            raise ValueError(f"{_source_key(record)}: calibration row has no dimension")
        by_dimension[dimension].append(record)
    if not by_dimension:
        raise ValueError(f"no candidates for non-empty calibration bucket {label}")
    capacities = {
        dimension: len(rows) * max_occurrences_per_source
        for dimension, rows in by_dimension.items()
    }
    if sum(capacities.values()) < count:
        raise ValueError(
            f"{label} requires {count} rows but repeat-capped capacity is "
            f"{sum(capacities.values())}"
        )

    shared = shared_dimension_counts if shared_dimension_counts is not None else Counter()
    baseline = baseline_dimension_counts or {}
    allocated: Counter[str] = Counter()
    tie_breakers = {
        dimension: hashlib.sha256(
            f"{seed}:dimension-waterfill:{label}:{dimension}".encode("utf-8")
        ).hexdigest()
        for dimension in by_dimension
    }
    for _ in range(count):
        available = [
            dimension
            for dimension in by_dimension
            if allocated[dimension] < capacities[dimension]
        ]
        if not available:
            raise AssertionError(f"{label}: exhausted dimension capacity")
        chosen = min(
            available,
            key=lambda dimension: (
                shared.get(dimension, 0),
                baseline.get(dimension, 0) + shared.get(dimension, 0),
                tie_breakers[dimension],
            ),
        )
        allocated[chosen] += 1
        shared[chosen] = shared.get(chosen, 0) + 1

    selected: List[Mapping[str, Any]] = []
    for dimension in sorted(by_dimension):
        quota = allocated[dimension]
        if not quota:
            continue
        selected.extend(
            _cyclic_sample(
                by_dimension[dimension],
                quota,
                seed=seed,
                label=f"{label}:{dimension}",
            )
        )
    return selected, dict(sorted(allocated.items()))


def _balanced_replay_sample(
    records: Sequence[Mapping[str, Any]], count: int, *, seed: int
) -> Tuple[List[Mapping[str, Any]], Dict[str, int]]:
    by_dimension: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        dimension = _dimension(record)
        if not dimension:
            raise ValueError(f"{_source_key(record)}: replay row has no dimension")
        by_dimension[dimension].append(record)
    dimensions = sorted(by_dimension)
    if count < len(dimensions):
        raise ValueError("replay budget is too small to cover every dimension")
    base, remainder = divmod(count, len(dimensions))
    ranked = sorted(
        dimensions,
        key=lambda value: hashlib.sha256(
            f"{seed}:dimension-remainder:{value}".encode("utf-8")
        ).hexdigest(),
    )
    quotas = {dimension: base for dimension in dimensions}
    for dimension in ranked[:remainder]:
        quotas[dimension] += 1

    selected: List[Mapping[str, Any]] = []
    for dimension in dimensions:
        selected.extend(
            _cyclic_sample(
                by_dimension[dimension],
                quotas[dimension],
                seed=seed,
                label=f"balanced-replay:{dimension}",
            )
        )
    return selected, dict(sorted(quotas.items()))


def _q3_analysis(
    records: Sequence[Mapping[str, Any]],
) -> Tuple[List[Tuple[Mapping[str, Any], Dict[str, Any]]], Dict[str, Any]]:
    mapped: List[Tuple[Mapping[str, Any], Dict[str, Any]]] = []
    errors: List[str] = []
    for record in records:
        if _question_type(record) != "Q3":
            continue
        try:
            mapped.append((record, map_q3_gold_semantic(record)))
        except ValueError as exc:
            errors.append(str(exc))
    if errors:
        examples = "; ".join(errors[:5])
        raise ValueError(
            f"Q3 semantic mapping failed for {len(errors)} row(s); refusing to sample: {examples}"
        )
    gold_semantics = [mapping["gold_semantic"] for _, mapping in mapped]
    gold_text = [mapping["gold_option_text"] for _, mapping in mapped]
    audit = {
        "policy": "strict_option_text_fail_closed",
        "accepted_semantic_text": dict(sorted(Q3_SEMANTIC_TEXT.items())),
        "q3_rows": len(mapped),
        "mapped_rows": len(mapped),
        "mapping_coverage": 1.0 if mapped else 0.0,
        "mapping_errors": [],
        "gold_semantic_distribution": _stable_distribution(gold_semantics),
        "gold_option_text_distribution": _stable_distribution(gold_text),
        "uncertain_candidates": gold_semantics.count("uncertain"),
    }
    return mapped, audit


def _decorate_instances(
    selections: Sequence[Tuple[str, Mapping[str, Any], Mapping[str, Any]]], *, seed: int
) -> List[Dict[str, Any]]:
    global_occurrence: Counter[str] = Counter()
    bucket_occurrence: Counter[Tuple[str, str]] = Counter()
    output: List[Dict[str, Any]] = []
    for ordinal, (bucket, source, annotations) in enumerate(selections):
        copied = copy.deepcopy(dict(source))
        key = _source_key(source)
        pipeline = copied.setdefault("_pipeline", {})
        if not isinstance(pipeline, dict):
            pipeline = {}
            copied["_pipeline"] = pipeline
        old_instance = pipeline.get("sample_instance_id")
        old_occurrence = pipeline.get("sampling_occurrence")
        occurrence = global_occurrence[key]
        per_bucket_occurrence = bucket_occurrence[(bucket, key)]
        global_occurrence[key] += 1
        bucket_occurrence[(bucket, key)] += 1
        digest_payload = (
            f"stage3:{seed}:{bucket}:{key}:{per_bucket_occurrence}:{ordinal}"
        )
        instance_id = "s3_" + hashlib.sha256(digest_payload.encode("utf-8")).hexdigest()[:20]
        pipeline["sample_instance_id"] = instance_id
        pipeline["sampling_occurrence"] = occurrence
        pipeline["calibration"] = {
            "schema_version": 1,
            "source_bucket": bucket,
            "source_key": key,
            "source_sample_instance_id": old_instance,
            "source_sampling_occurrence": old_occurrence,
            "source_occurrence_in_output": occurrence,
            "source_occurrence_in_bucket": per_bucket_occurrence,
            **dict(annotations),
        }
        output.append(copied)
    random.Random(seed + 3003).shuffle(output)
    return output


def _selected_audit(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    source_buckets: List[str] = []
    qtypes: List[str] = []
    dimensions: List[str] = []
    instance_ids: List[str] = []
    source_keys: List[str] = []
    q2_cardinalities: List[str] = []
    q3_semantics: List[str] = []
    per_bucket_qtype: Dict[str, List[str]] = defaultdict(list)
    per_bucket_dimension: Dict[str, List[str]] = defaultdict(list)
    per_bucket_sources: Dict[str, List[str]] = defaultdict(list)

    for record in records:
        pipeline = record.get("_pipeline")
        calibration = pipeline.get("calibration") if isinstance(pipeline, Mapping) else None
        if not isinstance(calibration, Mapping):
            raise AssertionError("output row is missing calibration provenance")
        bucket = str(calibration.get("source_bucket"))
        qtype = _question_type(record)
        dimension = _dimension(record)
        source = str(calibration.get("source_key"))
        source_buckets.append(bucket)
        qtypes.append(qtype)
        dimensions.append(dimension)
        source_keys.append(source)
        per_bucket_qtype[bucket].append(qtype)
        per_bucket_dimension[bucket].append(dimension)
        per_bucket_sources[bucket].append(source)
        instance_ids.append(str(pipeline.get("sample_instance_id")))
        if qtype == "Q2":
            q2_cardinalities.append(str(len(_gold_letters(record))))
        if qtype == "Q3":
            q3_semantics.append(map_q3_gold_semantic(record)["gold_semantic"])

    source_counts = Counter(source_keys)

    def dimension_summary(values: Sequence[str]) -> Dict[str, Any]:
        counts = Counter(values)
        observed = list(counts.values())
        return {
            "count": len(counts),
            "min_rows": min(observed, default=0),
            "max_rows": max(observed, default=0),
            "spread": max(observed, default=0) - min(observed, default=0),
            "distribution": dict(sorted(counts.items())),
        }

    def duplicate_summary(keys: Sequence[str]) -> Dict[str, Any]:
        counts = Counter(keys)
        histogram = Counter(counts.values())
        return {
            "unique_source_rows": len(counts),
            "duplicate_output_rows": len(keys) - len(counts),
            "sources_used_more_than_once": sum(value > 1 for value in counts.values()),
            "max_source_occurrence": max(counts.values(), default=0),
            "source_occurrence_histogram": dict(
                sorted((str(key), value) for key, value in histogram.items())
            ),
        }

    per_bucket = {}
    for bucket in SOURCE_BUCKETS:
        bucket_dimensions = dimension_summary(per_bucket_dimension[bucket])
        per_bucket[bucket] = {
            "rows": source_buckets.count(bucket),
            "question_type_distribution": _stable_distribution(per_bucket_qtype[bucket]),
            "dimension_count": bucket_dimensions["count"],
            "dimension_distribution": bucket_dimensions["distribution"],
            "dimensions": bucket_dimensions,
            "duplication": duplicate_summary(per_bucket_sources[bucket]),
        }
    return {
        "rows": len(records),
        "source_distribution": _stable_distribution(source_buckets),
        "question_type_distribution": _stable_distribution(qtypes),
        "q2_gold_cardinality_distribution": _stable_distribution(q2_cardinalities),
        "q3_gold_semantic_distribution": _stable_distribution(q3_semantics),
        "dimensions": dimension_summary(dimensions),
        "sample_instance_ids": {
            "rows": len(instance_ids),
            "unique": len(set(instance_ids)),
            "duplicates": len(instance_ids) - len(set(instance_ids)),
        },
        "duplication": {
            **duplicate_summary(source_keys),
            "most_repeated_sources": [
                {"source_key": key, "occurrences": count}
                for key, count in sorted(
                    source_counts.items(), key=lambda item: (-item[1], item[0])
                )[:20]
            ],
        },
        "per_bucket": per_bucket,
    }


def build_calibration_dataset(
    train_records: Sequence[Mapping[str, Any]],
    balanced_replay_records: Sequence[Mapping[str, Any]],
    config: CalibrationConfig = CalibrationConfig(),
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Create the Stage-3 calibration dataset and a complete audit."""

    config.validate()
    if not train_records:
        raise ValueError("training records are empty")
    if not balanced_replay_records:
        raise ValueError("balanced replay records are empty")

    train_sources = {_source_key(record) for record in train_records}
    replay_sources = {_source_key(record) for record in balanced_replay_records}
    external_replay_sources = sorted(replay_sources - train_sources)
    if external_replay_sources:
        raise ValueError(
            "balanced replay contains rows that are not in the training split: "
            + ", ".join(external_replay_sources[:5])
        )

    train_dimensions = sorted({_dimension(record) for record in train_records if _dimension(record)})
    replay_dimensions = sorted(
        {_dimension(record) for record in balanced_replay_records if _dimension(record)}
    )
    if len(replay_dimensions) != config.required_dimensions:
        raise ValueError(
            f"expected {config.required_dimensions} replay dimensions, "
            f"found {len(replay_dimensions)}"
        )

    q3_mapped, q3_mapping_audit = _q3_analysis(train_records)
    if not q3_mapped:
        raise ValueError("training split contains no Q3 rows")
    uncertain_q3 = [row for row, mapping in q3_mapped if mapping["gold_semantic"] == "uncertain"]

    q2_by_cardinality: Dict[int, List[Mapping[str, Any]]] = defaultdict(list)
    for record in train_records:
        if _question_type(record) != "Q2":
            continue
        cardinality = len(_gold_letters(record))
        if cardinality <= 0:
            raise ValueError(f"{_source_key(record)}: Q2 row has no gold letters")
        q2_by_cardinality[cardinality].append(record)
    if not q2_by_cardinality.get(1):
        raise ValueError("training split has no single-letter Q2 calibration candidates")
    secondary_cardinalities = sorted(value for value in q2_by_cardinality if value > 1)
    if not secondary_cardinalities:
        raise ValueError("training split has no secondary low-cardinality Q2 candidates")
    secondary_cardinality = secondary_cardinalities[0]

    budgets = _largest_remainder_budgets(config)
    replay, replay_quotas = _balanced_replay_sample(
        balanced_replay_records, budgets["balanced_replay"], seed=config.seed
    )
    q2_budget = budgets["q2_low_cardinality"]
    q2_single_budget = round(q2_budget * config.q2_single_share)
    q2_secondary_budget = q2_budget - q2_single_budget
    q2_dimension_counts: Counter[str] = Counter()
    q2_single, q2_single_quotas = _capacity_balanced_sample(
        q2_by_cardinality[1],
        q2_single_budget,
        seed=config.seed,
        label="q2-cardinality-1",
        max_occurrences_per_source=4,
        shared_dimension_counts=q2_dimension_counts,
        baseline_dimension_counts=replay_quotas,
    )
    q2_secondary, q2_secondary_quotas = _capacity_balanced_sample(
        q2_by_cardinality[secondary_cardinality],
        q2_secondary_budget,
        seed=config.seed,
        label=f"q2-cardinality-{secondary_cardinality}",
        max_occurrences_per_source=2,
        shared_dimension_counts=q2_dimension_counts,
        baseline_dimension_counts=replay_quotas,
    )
    q3_baseline_counts: Counter[str] = Counter(replay_quotas)
    q3_baseline_counts.update(q2_dimension_counts)
    q3_dimension_counts: Counter[str] = Counter()
    q3_selected, q3_quotas = _capacity_balanced_sample(
        uncertain_q3,
        budgets["q3_uncertain"],
        seed=config.seed,
        label="q3-uncertain",
        max_occurrences_per_source=2,
        shared_dimension_counts=q3_dimension_counts,
        baseline_dimension_counts=q3_baseline_counts,
    )

    selections: List[Tuple[str, Mapping[str, Any], Mapping[str, Any]]] = []
    selections.extend(("balanced_replay", row, {}) for row in replay)
    selections.extend(
        (
            "q2_low_cardinality",
            row,
            {"q2_gold_cardinality": len(_gold_letters(row))},
        )
        for row in q2_single + q2_secondary
    )
    selections.extend(
        (
            "q3_uncertain",
            row,
            {"q3_gold_semantic": "uncertain"},
        )
        for row in q3_selected
    )
    output = _decorate_instances(selections, seed=config.seed)
    selected_audit = _selected_audit(output)

    expected_sources = dict(sorted(budgets.items()))
    actual_sources = selected_audit["source_distribution"]
    invariants = {
        "total_rows_exact": len(output) == config.total_rows,
        "source_budgets_exact": actual_sources == expected_sources,
        "required_dimensions_covered": (
            selected_audit["dimensions"]["count"] == config.required_dimensions
        ),
        "balanced_replay_dimension_count": (
            selected_audit["per_bucket"]["balanced_replay"]["dimension_count"]
            == config.required_dimensions
        ),
        "balanced_replay_quota_spread_at_most_one": (
            max(replay_quotas.values()) - min(replay_quotas.values()) <= 1
        ),
        "unique_sample_instance_ids": (
            selected_audit["sample_instance_ids"]["duplicates"] == 0
        ),
        "q3_semantic_mapping_complete": (
            q3_mapping_audit["mapped_rows"] == q3_mapping_audit["q3_rows"]
        ),
        "q3_calibration_is_uncertain_only": (
            all(map_q3_gold_semantic(row)["gold_semantic"] == "uncertain" for row in q3_selected)
        ),
        "q2_special_source_repeat_cap_respected": (
            selected_audit["per_bucket"]["q2_low_cardinality"]["duplication"][
                "max_source_occurrence"
            ]
            <= 4
        ),
        "q3_special_source_repeat_cap_respected": (
            selected_audit["per_bucket"]["q3_uncertain"]["duplication"][
                "max_source_occurrence"
            ]
            <= 2
        ),
        "replay_sources_are_training_only": not external_replay_sources,
        "public_test_input_absent": True,
    }
    if not all(value is True for value in invariants.values()):
        failed = [key for key, value in invariants.items() if value is not True]
        raise AssertionError("calibration invariant failure: " + ", ".join(failed))

    audit = {
        "schema_version": 1,
        "config": asdict(config),
        "budgets": expected_sources,
        "input": {
            "train_rows": len(train_records),
            "balanced_replay_rows": len(balanced_replay_records),
            "train_question_type_distribution": _stable_distribution(
                [_question_type(record) for record in train_records]
            ),
            "train_dimension_count": len(train_dimensions),
            "balanced_replay_dimension_count": len(replay_dimensions),
            "balanced_replay_sources_not_in_training": len(external_replay_sources),
            "q2_candidate_cardinality_distribution": dict(
                sorted((str(key), len(value)) for key, value in q2_by_cardinality.items())
            ),
            "q2_secondary_cardinality": secondary_cardinality,
        },
        "q3_semantic_mapping": q3_mapping_audit,
        "replay_dimension_quotas": replay_quotas,
        "target_bucket_dimension_sampling": {
            "policy": "capacity_aware_dimension_waterfill",
            "q2": {
                "single_source_repeat_cap": 4,
                "secondary_source_repeat_cap": 2,
                "single": _quota_summary(q2_single_quotas),
                "secondary": _quota_summary(q2_secondary_quotas),
                "combined": _quota_summary(dict(q2_dimension_counts)),
            },
            "q3_uncertain": {
                "source_repeat_cap": 2,
                **_quota_summary(q3_quotas),
            },
        },
        "output": selected_audit,
        "data_leakage": {
            "public_test_input_used": False,
            "public_leakage_check": "not_applicable_no_public_input",
            "provenance_rule": "both inputs must resolve to the same training split source IDs",
        },
        "invariants": invariants,
    }
    return output, audit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a training-only Stage-3 replay/Q2/Q3 calibration mix."
    )
    parser.add_argument("--train", required=True, type=Path, help="Processed train split")
    parser.add_argument(
        "--balanced-replay", required=True, type=Path, help="71-dimension balanced train replay"
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--total-rows", type=int, default=2982)
    parser.add_argument("--replay-fraction", type=float, default=0.70)
    parser.add_argument("--q2-fraction", type=float, default=0.15)
    parser.add_argument("--q3-fraction", type=float, default=0.15)
    parser.add_argument("--q2-single-share", type=float, default=0.70)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--required-dimensions", type=int, default=71)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    output_path = args.output_dir / "train_stage3_calibration.jsonl"
    audit_path = args.output_dir / "stage3_calibration_audit.json"
    existing = [path for path in (output_path, audit_path) if path.exists()]
    if existing and not args.force:
        raise FileExistsError(
            "refusing to replace existing calibration output without --force: "
            + ", ".join(str(path) for path in existing)
        )

    config = CalibrationConfig(
        total_rows=args.total_rows,
        replay_fraction=args.replay_fraction,
        q2_fraction=args.q2_fraction,
        q3_fraction=args.q3_fraction,
        q2_single_share=args.q2_single_share,
        seed=args.seed,
        required_dimensions=args.required_dimensions,
    )
    train_records = read_jsonl(args.train)
    replay_records = read_jsonl(args.balanced_replay)
    output, audit = build_calibration_dataset(train_records, replay_records, config)
    audit["input"]["files"] = {
        "train": {"path": str(args.train.resolve()), "sha256": file_sha256(args.train)},
        "balanced_replay": {
            "path": str(args.balanced_replay.resolve()),
            "sha256": file_sha256(args.balanced_replay),
        },
    }
    write_jsonl(output_path, output)
    audit["output"]["file"] = {
        "path": str(output_path.resolve()),
        "sha256": file_sha256(output_path),
    }
    write_json(audit_path, audit)
    print(
        json.dumps(
            {
                "output": str(output_path.resolve()),
                "audit": str(audit_path.resolve()),
                "rows": len(output),
                "source_distribution": audit["output"]["source_distribution"],
                "dimensions": audit["output"]["dimensions"]["count"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
