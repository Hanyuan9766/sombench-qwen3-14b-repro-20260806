"""Read public/dev evaluation outputs and produce compact JSON/Markdown.

This command never calls a model endpoint and never modifies either evaluation
directory. It reads ``metrics.json``, ``sample_stability.jsonl``, and the
canonical prediction artifact, then writes only the explicitly requested
summary files.

Example:
    python -m training.summarize_evaluation \
      --public-dir /root/autodl-tmp/sombench/outputs/eval-public \
      --dev-dir /root/autodl-tmp/sombench/outputs/eval-dev \
      --json-out /root/autodl-tmp/sombench/outputs/evaluation-summary.json \
      --markdown-out /root/autodl-tmp/sombench/outputs/evaluation-summary.md
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .data import iter_jsonl


class SummaryError(ValueError):
    """Raised when an evaluation output does not match the expected schema."""


LABEL_SCOPES: dict[str, tuple[str, str, str]] = {
    # compact name: (per-seed/majority key, aggregate key, grouped key)
    "reported_letters": (
        "reported_letter_overall",
        "reported_letter_accuracy",
        "reported_letters",
    ),
    "safe_agreement": (
        "safe_agreement_subset",
        "safe_agreement_accuracy",
        "safe_agreement",
    ),
    "mapped_answer_text": (
        "mapped_answer_text_available",
        "mapped_answer_text_accuracy",
        "mapped_answer_text",
    ),
}
EXPECTED_QTYPES = ("Q1", "Q2", "Q3")
PRIMARY_DIMENSIONS: dict[str, str] = {
    "1:mentalization": "Mentalization",
    "2:strategic_social_interaction": "Strategic social interaction",
    "3:sociocultural_norms": "Sociocultural norms",
}
# Inferred from the A-leaderboard score increments and the 71-dimension design:
# 30/27/14 fine dimensions x 25 items = 750/675/350 examples.  This is an
# explicitly inferred diagnostic, not an official evaluation claim.
A_LEADERBOARD_INFERRED_COUNTS: dict[str, int] = {
    "1:mentalization": 750,
    "2:strategic_social_interaction": 675,
    "3:sociocultural_norms": 350,
}


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SummaryError(f"missing required file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SummaryError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SummaryError(f"{path} must contain a JSON object")
    return value


def _metric_block(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {"total": 0, "correct": 0, "accuracy": None, "parse_rate": None}
    return {
        "total": int(value.get("total") or 0),
        "correct": int(value.get("correct") or 0),
        "accuracy": value.get("accuracy"),
        "parse_rate": value.get("parse_rate"),
    }


def _summary_block(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {"mean": None, "std": None, "min": None, "max": None}
    return {key: value.get(key) for key in ("mean", "std", "min", "max")}


def _seed_sort_key(value: str) -> tuple[int, str]:
    try:
        return int(value), value
    except ValueError:
        return 2**63 - 1, value


def _group_summary(
    per_seed: Mapping[str, Any],
    aggregate_groups: Mapping[str, Any],
    seed_ids: Sequence[str],
    *,
    container: str,
    group: str,
) -> dict[str, Any]:
    """Keep aggregate compatibility while adding auditable per-seed groups."""

    aggregate_source = aggregate_groups.get(group)
    aggregate_source = aggregate_source if isinstance(aggregate_source, Mapping) else {}
    result: dict[str, Any] = {
        label: _summary_block(aggregate_source.get(label)) for label in LABEL_SCOPES
    }
    result["per_seed"] = {}
    for seed in seed_ids:
        seed_source = per_seed.get(seed)
        seed_source = seed_source if isinstance(seed_source, Mapping) else {}
        grouped = seed_source.get(container)
        grouped = grouped if isinstance(grouped, Mapping) else {}
        group_source = grouped.get(group)
        group_source = group_source if isinstance(group_source, Mapping) else {}
        result["per_seed"][seed] = {
            label: _metric_block(group_source.get(label)) for label in LABEL_SCOPES
        }
    return result


def _inferred_a_leaderboard_score(
    primary_dimensions: Mapping[str, Any], seed_ids: Sequence[str]
) -> dict[str, Any]:
    """Calculate the inferred A-leaderboard weighting without renormalizing gaps."""

    denominator = sum(A_LEADERBOARD_INFERRED_COUNTS.values())
    weights = {
        dimension: count / denominator
        for dimension, count in A_LEADERBOARD_INFERRED_COUNTS.items()
    }
    per_seed: dict[str, Any] = {}
    complete_scores: list[float] = []
    for seed in seed_ids:
        contributions: dict[str, Any] = {}
        missing: list[str] = []
        score = 0.0
        covered_weight = 0.0
        for dimension, weight in weights.items():
            group = primary_dimensions.get(dimension)
            group = group if isinstance(group, Mapping) else {}
            seed_groups = group.get("per_seed")
            seed_groups = seed_groups if isinstance(seed_groups, Mapping) else {}
            seed_group = seed_groups.get(seed)
            seed_group = seed_group if isinstance(seed_group, Mapping) else {}
            metric = seed_group.get("reported_letters")
            metric = metric if isinstance(metric, Mapping) else {}
            accuracy = metric.get("accuracy")
            total = int(metric.get("total") or 0)
            contribution = float(accuracy) * weight if accuracy is not None and total else None
            contributions[dimension] = {
                "accuracy": accuracy,
                "total": total,
                "weight": weight,
                "weighted_contribution": contribution,
            }
            if contribution is None:
                missing.append(dimension)
            else:
                score += contribution
                covered_weight += weight
        complete = not missing
        final_score = score if complete else None
        if final_score is not None:
            complete_scores.append(final_score)
        per_seed[seed] = {
            "score": final_score,
            "complete": complete,
            "covered_weight": covered_weight,
            "missing_dimensions": missing,
            "contributions": contributions,
        }
    return {
        "status": "complete" if len(complete_scores) == len(seed_ids) else "not_applicable",
        "label_policy": "reported_letters",
        "inference_basis": {
            "total_items": denominator,
            "dimension_item_counts": dict(A_LEADERBOARD_INFERRED_COUNTS),
            "dimension_weights": weights,
            "fine_dimension_counts": {
                "1:mentalization": 30,
                "2:strategic_social_interaction": 27,
                "3:sociocultural_norms": 14,
            },
            "items_per_fine_dimension": 25,
        },
        "per_seed": per_seed,
        "aggregate_score": _summary_block(
            {
                "mean": sum(complete_scores) / len(complete_scores) if complete_scores else None,
                "std": (
                    sum((value - sum(complete_scores) / len(complete_scores)) ** 2 for value in complete_scores)
                    / len(complete_scores)
                )
                ** 0.5
                if complete_scores
                else None,
                "min": min(complete_scores) if complete_scores else None,
                "max": max(complete_scores) if complete_scores else None,
            }
        ),
        "note": (
            "Diagnostic weighted score inferred from A-leaderboard increments; "
            "it is emitted only when all three primary dimensions are present."
        ),
    }


def _stability_audit(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {
            "file_present": False,
            "samples": 0,
            "complete_samples": 0,
            "consistent_samples": 0,
            "tie_samples": 0,
            "consistency_rate": None,
        }
    rows = list(iter_jsonl(path))
    sample_ids = [str(row.get("sample_id")) for row in rows]
    if any(item in {"", "None"} for item in sample_ids):
        raise SummaryError(f"{path} contains a row without sample_id")
    if len(sample_ids) != len(set(sample_ids)):
        raise SummaryError(f"{path} contains duplicate sample_id rows")
    complete = [row for row in rows if row.get("complete")]
    consistent = sum(bool(row.get("all_seed_consistent")) for row in complete)
    ties = sum(bool(row.get("majority_tie")) for row in complete)
    return {
        "file_present": True,
        "samples": len(rows),
        "complete_samples": len(complete),
        "consistent_samples": consistent,
        "tie_samples": ties,
        "consistency_rate": consistent / len(complete) if complete else None,
    }


def _prediction_artifact_audit(
    path: Path,
    *,
    seed_ids: Sequence[str],
    record_count: int,
) -> dict[str, Any]:
    """Validate the canonical seed/sample grid and summarize response endings.

    The audit deliberately uses only ``predictions_canonical.jsonl`` for these
    fields. A majority-vote tie is not a prediction row failure and therefore
    has no bearing on the empty-prediction count.
    """

    if not path.is_file():
        raise SummaryError(f"missing required file: {path}")
    rows = list(iter_jsonl(path))
    expected_seeds = set(seed_ids)
    keys: list[tuple[str, str]] = []
    sample_ids: set[str] = set()
    finish_reason_counts: dict[str, int] = {}
    length_truncation_count = 0
    empty_predicted_letters_count = 0

    for index, row in enumerate(rows, start=1):
        seed = str(row.get("seed"))
        sample_id = str(row.get("sample_id"))
        if seed in {"", "None"}:
            raise SummaryError(f"{path}:{index} contains a row without seed")
        if sample_id in {"", "None"}:
            raise SummaryError(f"{path}:{index} contains a row without sample_id")
        if seed not in expected_seeds:
            raise SummaryError(
                f"{path}:{index} contains unexpected seed {seed!r}; expected {list(seed_ids)!r}"
            )
        keys.append((seed, sample_id))
        sample_ids.add(sample_id)

        raw_reason = row.get("finish_reason")
        if raw_reason is None:
            reason = "<missing>"
        else:
            reason = str(raw_reason).strip() or "<empty>"
        finish_reason_counts[reason] = finish_reason_counts.get(reason, 0) + 1
        if reason.casefold() == "length":
            length_truncation_count += 1

        predicted = row.get("predicted_letters")
        if predicted is None or (hasattr(predicted, "__len__") and len(predicted) == 0):
            empty_predicted_letters_count += 1

    duplicate_keys = sorted(key for key in set(keys) if keys.count(key) > 1)
    if duplicate_keys:
        raise SummaryError(
            f"{path} contains duplicate (seed, sample_id) rows: {duplicate_keys[:10]!r}"
        )
    if len(sample_ids) != record_count:
        raise SummaryError(
            f"{path} sample grid has {len(sample_ids)} unique sample_ids, "
            f"but evaluation_manifest.json declares {record_count}"
        )
    expected_keys = {(seed, sample_id) for seed in seed_ids for sample_id in sample_ids}
    observed_keys = set(keys)
    missing_keys = sorted(expected_keys - observed_keys)
    if missing_keys:
        raise SummaryError(
            f"{path} is missing (seed, sample_id) grid rows: {missing_keys[:10]!r}"
        )
    unexpected_keys = sorted(observed_keys - expected_keys)
    if unexpected_keys:
        raise SummaryError(
            f"{path} contains unexpected (seed, sample_id) grid rows: {unexpected_keys[:10]!r}"
        )

    row_count = len(rows)
    expected_grid_rows = record_count * len(seed_ids)
    if row_count != expected_grid_rows:
        raise SummaryError(
            f"{path} contains {row_count} rows, expected a {len(seed_ids)} x "
            f"{record_count} grid ({expected_grid_rows} rows)"
        )
    return {
        "file_present": True,
        "rows": row_count,
        "sample_count": len(sample_ids),
        "seed_count": len(seed_ids),
        "expected_grid_rows": expected_grid_rows,
        "grid_complete": True,
        "finish_reason_counts": dict(sorted(finish_reason_counts.items())),
        "length_truncation_count": length_truncation_count,
        "length_truncation_rate": (
            length_truncation_count / row_count if row_count else None
        ),
        "empty_predicted_letters_count": empty_predicted_letters_count,
        "empty_predicted_letters_rate": (
            empty_predicted_letters_count / row_count if row_count else None
        ),
    }


def compact_evaluation(directory: str | Path, *, name: str) -> dict[str, Any]:
    """Extract stable summary fields from one evaluation output directory."""

    root = Path(directory).expanduser().resolve()
    if not root.is_dir():
        raise SummaryError(f"{name} evaluation directory does not exist: {root}")
    metrics = _read_object(root / "metrics.json")
    manifest = _read_object(root / "evaluation_manifest.json")
    per_seed = metrics.get("per_seed")
    aggregate = metrics.get("aggregate")
    stability = metrics.get("stability")
    run = metrics.get("run")
    if not all(isinstance(value, Mapping) for value in (per_seed, aggregate, stability, run)):
        raise SummaryError(f"{root / 'metrics.json'} lacks per_seed/aggregate/stability/run objects")
    seed_ids = sorted((str(seed) for seed in per_seed), key=_seed_sort_key)
    if len(seed_ids) != 3:
        raise SummaryError(f"{name} evaluation must contain exactly three seeds, found {seed_ids}")
    record_count = int(manifest.get("record_count") or 0)
    if record_count <= 0:
        raise SummaryError(f"{root / 'evaluation_manifest.json'} has invalid record_count")
    prediction_artifact = _prediction_artifact_audit(
        root / "predictions_canonical.jsonl",
        seed_ids=seed_ids,
        record_count=record_count,
    )

    majority = stability.get("majority_vote")
    if not isinstance(majority, Mapping):
        raise SummaryError(f"{name} stability.majority_vote is missing")
    labels: dict[str, Any] = {}
    for label, (seed_key, aggregate_key, _) in LABEL_SCOPES.items():
        labels[label] = {
            "per_seed": {
                seed: _metric_block(per_seed[seed].get(seed_key) if isinstance(per_seed[seed], Mapping) else None)
                for seed in seed_ids
            },
            "aggregate_accuracy": _summary_block(aggregate.get(aggregate_key)),
            "majority_vote": _metric_block(majority.get(seed_key)),
        }

    aggregate_qtypes = aggregate.get("by_qtype")
    aggregate_qtypes = aggregate_qtypes if isinstance(aggregate_qtypes, Mapping) else {}
    seed_qtypes = {
        str(qtype)
        for seed in seed_ids
        if isinstance(per_seed.get(seed), Mapping)
        for qtype in (
            per_seed[seed].get("by_qtype", {})
            if isinstance(per_seed[seed].get("by_qtype"), Mapping)
            else {}
        )
    }
    all_qtypes = list(EXPECTED_QTYPES) + sorted(
        (set(map(str, aggregate_qtypes)) | seed_qtypes) - set(EXPECTED_QTYPES)
    )
    qtypes = {
        qtype: _group_summary(
            per_seed,
            aggregate_qtypes,
            seed_ids,
            container="by_qtype",
            group=qtype,
        )
        for qtype in all_qtypes
    }

    aggregate_primaries = aggregate.get("by_primary_dimension")
    aggregate_primaries = aggregate_primaries if isinstance(aggregate_primaries, Mapping) else {}
    seed_primaries = {
        str(dimension)
        for seed in seed_ids
        if isinstance(per_seed.get(seed), Mapping)
        for dimension in (
            per_seed[seed].get("by_primary_dimension", {})
            if isinstance(per_seed[seed].get("by_primary_dimension"), Mapping)
            else {}
        )
    }
    primary_keys = list(PRIMARY_DIMENSIONS) + sorted(
        (set(map(str, aggregate_primaries)) | seed_primaries) - set(PRIMARY_DIMENSIONS)
    )
    primary_dimensions = {
        dimension: _group_summary(
            per_seed,
            aggregate_primaries,
            seed_ids,
            container="by_primary_dimension",
            group=dimension,
        )
        for dimension in primary_keys
        if dimension in aggregate_primaries or dimension in seed_primaries
    }
    leaderboard_a = _inferred_a_leaderboard_score(primary_dimensions, seed_ids)

    stability_file = _stability_audit(root / "sample_stability.jsonl")
    metric_complete = int(stability.get("complete_samples") or 0)
    metric_ties = int(stability.get("majority_tie_count") or 0)
    metric_consistency = stability.get("all_seed_consistency_rate")
    warnings: list[str] = []
    if stability_file["file_present"]:
        if stability_file["complete_samples"] != metric_complete:
            warnings.append("sample_stability complete count differs from metrics.json")
        if stability_file["tie_samples"] != metric_ties:
            warnings.append("sample_stability tie count differs from metrics.json")
        file_rate = stability_file["consistency_rate"]
        if file_rate is not None and metric_consistency is not None and abs(file_rate - metric_consistency) > 1e-12:
            warnings.append("sample_stability consistency rate differs from metrics.json")
    else:
        warnings.append("sample_stability.jsonl is missing")

    expected_predictions = int(run.get("expected_predictions") or 0)
    completed_predictions = int(run.get("completed_predictions") or 0)
    missing = int(run.get("missing_or_failed") or 0)
    if completed_predictions + missing != expected_predictions:
        warnings.append("completed_predictions + missing_or_failed != expected_predictions")
    if prediction_artifact["rows"] != completed_predictions:
        warnings.append("canonical prediction row count differs from completed_predictions")
    if prediction_artifact["rows"] != int(run.get("canonical_prediction_rows") or 0):
        warnings.append("canonical prediction row count differs from metrics.json")
    if prediction_artifact["expected_grid_rows"] != expected_predictions:
        warnings.append("canonical prediction grid size differs from expected_predictions")
    retained_from_artifact = (
        stability_file["file_present"]
        and stability_file["samples"] == record_count
    )
    if bool(run.get("all_records_retained")) != retained_from_artifact:
        warnings.append("all_records_retained flag is inconsistent with stability artifact presence")

    dimension = aggregate.get("dimension_coverage")
    dimension = dict(dimension) if isinstance(dimension, Mapping) else {}
    first_seed = per_seed[seed_ids[0]] if isinstance(per_seed[seed_ids[0]], Mapping) else {}
    return {
        "name": name,
        "source_dir": str(root),
        "model": manifest.get("model"),
        "dataset": manifest.get("dataset"),
        "seeds": seed_ids,
        "run": {
            "expected_predictions": expected_predictions,
            "completed_predictions": completed_predictions,
            "missing_or_failed": missing,
            "canonical_prediction_rows": run.get("canonical_prediction_rows"),
            "all_records_retained": run.get("all_records_retained"),
        },
        "labels": labels,
        "qtypes": qtypes,
        "primary_dimensions": primary_dimensions,
        "leaderboard_a_inferred_weighted_score": leaderboard_a,
        "parse_rate": {
            "per_seed": {
                seed: labels["reported_letters"]["per_seed"][seed]["parse_rate"] for seed in seed_ids
            },
            "aggregate": _summary_block(aggregate.get("parse_rate")),
            "majority": labels["reported_letters"]["majority_vote"]["parse_rate"],
        },
        "stability": {
            "complete_samples": metric_complete,
            "incomplete_samples": int(stability.get("incomplete_samples") or 0),
            "consistency_rate": metric_consistency,
            "majority_tie_count": metric_ties,
            "majority_determinable_samples": metric_complete - metric_ties,
            "majority_determinable_rate": (
                (metric_complete - metric_ties) / metric_complete if metric_complete else None
            ),
            "artifact_audit": stability_file,
        },
        "prediction_artifact_audit": prediction_artifact,
        "dimension_coverage": dimension,
        "label_audit": dict(first_seed.get("label_audit") or {}),
        "warnings": warnings,
    }


def build_summary(public_dir: str | Path, dev_dir: str | Path) -> dict[str, Any]:
    return {
        "schema_version": 3,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "evaluations": {
            "public": compact_evaluation(public_dir, name="public"),
            "dev": compact_evaluation(dev_dir, name="dev"),
        },
    }


def _percent(value: Any) -> str:
    return "—" if value is None else f"{float(value) * 100:.2f}%"


def _mean_std(value: Mapping[str, Any]) -> str:
    if value.get("mean") is None:
        return "—"
    return f"{_percent(value['mean'])} ± {float(value.get('std') or 0) * 100:.2f}pp"


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    return [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(row) + " |" for row in rows),
    ]


def render_markdown(summary: Mapping[str, Any]) -> str:
    evaluations = summary["evaluations"]
    lines = ["# SoMBench Evaluation Summary", ""]
    integrity_rows = []
    for name in ("public", "dev"):
        item = evaluations[name]
        run = item["run"]
        stability = item["stability"]
        prediction_audit = item["prediction_artifact_audit"]
        finish_reasons = ", ".join(
            f"{reason}={count}"
            for reason, count in prediction_audit["finish_reason_counts"].items()
        )
        integrity_rows.append(
            [
                name,
                f"{run['completed_predictions']}/{run['expected_predictions']}",
                str(run["missing_or_failed"]),
                finish_reasons or "—",
                (
                    f"{prediction_audit['length_truncation_count']} "
                    f"({_percent(prediction_audit['length_truncation_rate'])})"
                ),
                (
                    f"{prediction_audit['empty_predicted_letters_count']} "
                    f"({_percent(prediction_audit['empty_predicted_letters_rate'])})"
                ),
                _percent(item["parse_rate"]["aggregate"]["mean"]),
                _percent(stability["consistency_rate"]),
                _percent(stability["majority_determinable_rate"]),
                str(stability["majority_tie_count"]),
            ]
        )
    lines.extend(
        _markdown_table(
            [
                "Dataset",
                "Predictions",
                "Missing",
                "Finish reasons",
                "Length truncations",
                "Empty predicted letters",
                "Parse rate",
                "3-seed consistency",
                "Majority determinable",
                "Majority ties",
            ],
            integrity_rows,
        )
    )

    label_titles = {
        "reported_letters": "Reported letters",
        "safe_agreement": "Safe agreement",
        "mapped_answer_text": "Mapped answer text",
    }
    for name in ("public", "dev"):
        item = evaluations[name]
        seeds = item["seeds"]
        lines.extend(["", f"## {name.title()}", ""])
        label_rows = []
        for label, title in label_titles.items():
            value = item["labels"][label]
            seed_values = [
                f"{_percent(value['per_seed'][seed]['accuracy'])} / {_percent(value['per_seed'][seed]['parse_rate'])}"
                for seed in seeds
            ]
            label_rows.append(
                [
                    title,
                    *seed_values,
                    _mean_std(value["aggregate_accuracy"]),
                    _percent(value["majority_vote"]["accuracy"]),
                    str(value["per_seed"][seeds[0]]["total"]),
                ]
            )
        lines.extend(
            _markdown_table(
                ["Label policy", *(f"Seed {seed} acc/parse" for seed in seeds), "Mean ± std", "Majority", "N"],
                label_rows,
            )
        )
        lines.extend(["", "### Q1/Q2/Q3 accuracy", ""])
        qtype_rows = []
        for qtype in EXPECTED_QTYPES:
            qtype_value = item["qtypes"][qtype]
            qtype_rows.append(
                [
                    qtype,
                    *(
                        _percent(qtype_value["per_seed"][seed]["reported_letters"]["accuracy"])
                        for seed in seeds
                    ),
                    _percent(qtype_value["reported_letters"]["mean"]),
                    _percent(qtype_value["safe_agreement"]["mean"]),
                    _percent(qtype_value["mapped_answer_text"]["mean"]),
                ]
            )
        lines.extend(
            _markdown_table(
                [
                    "QType",
                    *(f"Seed {seed} reported" for seed in seeds),
                    "Reported mean",
                    "Safe mean",
                    "Mapped mean",
                ],
                qtype_rows,
            )
        )
        primary_dimensions = item["primary_dimensions"]
        if primary_dimensions:
            lines.extend(["", "### Primary dimensions (reported-letter accuracy)", ""])
            primary_rows = []
            weights = item["leaderboard_a_inferred_weighted_score"]["inference_basis"][
                "dimension_weights"
            ]
            for dimension in PRIMARY_DIMENSIONS:
                if dimension not in primary_dimensions:
                    continue
                value = primary_dimensions[dimension]
                primary_rows.append(
                    [
                        f"{dimension} ({PRIMARY_DIMENSIONS[dimension]})",
                        *(
                            _percent(value["per_seed"][seed]["reported_letters"]["accuracy"])
                            for seed in seeds
                        ),
                        _percent(value["reported_letters"]["mean"]),
                        _percent(weights[dimension]),
                        str(value["per_seed"][seeds[0]]["reported_letters"]["total"]),
                    ]
                )
            lines.extend(
                _markdown_table(
                    [
                        "Primary dimension",
                        *(f"Seed {seed}" for seed in seeds),
                        "Mean",
                        "Inferred A weight",
                        "N/seed",
                    ],
                    primary_rows,
                )
            )
            leaderboard = item["leaderboard_a_inferred_weighted_score"]
            weighted_rows = [
                [seed, _percent(leaderboard["per_seed"][seed]["score"])] for seed in seeds
            ]
            weighted_rows.append(["Mean ± std", _mean_std(leaderboard["aggregate_score"])])
            lines.extend(
                ["", "### A-leaderboard inferred weighted score", ""]
                + _markdown_table(["Seed", "Score"], weighted_rows)
                + ["", leaderboard["note"]]
            )
        coverage = item["dimension_coverage"]
        lines.extend(
            [
                "",
                f"Dimension coverage: `{coverage.get('known', 0)}/{coverage.get('total', 0)}`; "
                f"fine dimensions: `{coverage.get('fine_dimension_count', 0)}`.",
            ]
        )
        if item["warnings"]:
            lines.extend(["", "Warnings: " + "; ".join(item["warnings"]) + "."])
    return "\n".join(lines) + "\n"


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-dir", required=True)
    parser.add_argument("--dev-dir", required=True)
    parser.add_argument("--json-out")
    parser.add_argument("--markdown-out")
    args = parser.parse_args()
    summary = build_summary(args.public_dir, args.dev_dir)
    json_text = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    markdown = render_markdown(summary)
    if args.json_out:
        _atomic_text(Path(args.json_out).expanduser().resolve(), json_text)
    if args.markdown_out:
        _atomic_text(Path(args.markdown_out).expanduser().resolve(), markdown)
    if not args.json_out and not args.markdown_out:
        print(markdown, end="")


if __name__ == "__main__":
    main()
