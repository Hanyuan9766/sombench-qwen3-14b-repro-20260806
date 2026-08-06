"""Compare two completed SoMBench evaluations on the exact same prediction grid.

The command is read-only with respect to both evaluation directories.  It
fails closed unless the dataset hash, prompt, generation protocol, seeds, and
``(seed, sample_id)`` grid agree.  Accuracy deltas are therefore paired deltas,
not a comparison of unrelated aggregate reports.

Example:
    python -m training.compare_evaluations \
      --baseline-dir /path/to/m1-dev \
      --candidate-dir /path/to/m2-dev \
      --baseline-name M1 --candidate-name M2 \
      --json-out /path/to/m1-vs-m2.json \
      --markdown-out /path/to/m1-vs-m2.md
"""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .data import iter_jsonl
from .evaluate import PRIMARY_DIMENSION_NAMES, primary_dimension
from .summarize_evaluation import (
    A_LEADERBOARD_INFERRED_COUNTS,
    PRIMARY_DIMENSIONS,
    SummaryError,
    compact_evaluation,
)


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


def _stats(values: Sequence[float | None]) -> dict[str, float | None]:
    usable = [float(value) for value in values if value is not None]
    if not usable:
        return {"mean": None, "std": None, "min": None, "max": None}
    return {
        "mean": statistics.fmean(usable),
        "std": statistics.pstdev(usable),
        "min": min(usable),
        "max": max(usable),
    }


def _canonical_rows(
    root: Path, *, expected_seeds: Sequence[str]
) -> dict[tuple[str, str], dict[str, Any]]:
    path = root / "predictions_canonical.jsonl"
    if not path.is_file():
        raise SummaryError(f"missing required file: {path}")
    expected = set(expected_seeds)
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for line_number, row in enumerate(iter_jsonl(path), start=1):
        seed = str(row.get("seed"))
        sample_id = str(row.get("sample_id"))
        if seed not in expected:
            raise SummaryError(
                f"{path}:{line_number} has seed {seed!r}, expected {list(expected_seeds)!r}"
            )
        if sample_id in {"", "None"}:
            raise SummaryError(f"{path}:{line_number} has no sample_id")
        key = (seed, sample_id)
        if key in result:
            raise SummaryError(f"{path} contains duplicate key {key!r}")
        prediction = row.get("predicted_letters")
        if prediction is not None and (
            not isinstance(prediction, list)
            or not prediction
            or not all(isinstance(letter, str) and letter for letter in prediction)
        ):
            raise SummaryError(
                f"{path}:{line_number} predicted_letters must be null or a non-empty string list"
            )
        gold = row.get("gold_letters")
        if not isinstance(gold, list) or not gold:
            raise SummaryError(f"{path}:{line_number} has invalid gold_letters")
        result[key] = row
    return result


def _same_protocol(
    baseline_manifest: Mapping[str, Any],
    candidate_manifest: Mapping[str, Any],
    baseline_seeds: Sequence[str],
    candidate_seeds: Sequence[str],
) -> dict[str, Any]:
    failures: list[str] = []
    baseline_hash = baseline_manifest.get("dataset_sha256")
    candidate_hash = candidate_manifest.get("dataset_sha256")
    if not isinstance(baseline_hash, str) or not baseline_hash:
        failures.append("baseline evaluation_manifest.json lacks dataset_sha256")
    if not isinstance(candidate_hash, str) or not candidate_hash:
        failures.append("candidate evaluation_manifest.json lacks dataset_sha256")
    if baseline_hash != candidate_hash:
        failures.append("dataset_sha256 differs")
    for field in ("record_count", "generation", "prompt"):
        if baseline_manifest.get(field) != candidate_manifest.get(field):
            failures.append(f"{field} differs")
    if list(baseline_seeds) != list(candidate_seeds):
        failures.append("seed list differs")
    if failures:
        raise SummaryError("evaluations are not paired under one protocol: " + "; ".join(failures))
    return {
        "status": "matched",
        "dataset_sha256": baseline_hash,
        "record_count": baseline_manifest.get("record_count"),
        "generation": baseline_manifest.get("generation"),
        "prompt": baseline_manifest.get("prompt"),
        "seeds": list(baseline_seeds),
    }


def _correct(row: Mapping[str, Any]) -> bool:
    prediction = row.get("predicted_letters")
    return prediction is not None and list(prediction) == list(row.get("gold_letters") or [])


def _parsed(row: Mapping[str, Any]) -> bool:
    return row.get("predicted_letters") is not None


def _length_truncated(row: Mapping[str, Any]) -> bool:
    reason = row.get("finish_reason")
    return isinstance(reason, str) and reason.strip().casefold() == "length"


def _paired_metric(rows: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]]) -> dict[str, Any]:
    total = len(rows)
    baseline_correct = sum(_correct(baseline) for baseline, _ in rows)
    candidate_correct = sum(_correct(candidate) for _, candidate in rows)
    wins = sum(not _correct(baseline) and _correct(candidate) for baseline, candidate in rows)
    losses = sum(_correct(baseline) and not _correct(candidate) for baseline, candidate in rows)
    ties_both_correct = sum(_correct(baseline) and _correct(candidate) for baseline, candidate in rows)
    ties_both_wrong = total - wins - losses - ties_both_correct
    ties = ties_both_correct + ties_both_wrong
    baseline_parsed = sum(_parsed(baseline) for baseline, _ in rows)
    candidate_parsed = sum(_parsed(candidate) for _, candidate in rows)
    parse_wins = sum(not _parsed(baseline) and _parsed(candidate) for baseline, candidate in rows)
    parse_losses = sum(_parsed(baseline) and not _parsed(candidate) for baseline, candidate in rows)
    baseline_truncated = sum(_length_truncated(baseline) for baseline, _ in rows)
    candidate_truncated = sum(_length_truncated(candidate) for _, candidate in rows)
    baseline_only_truncated = sum(
        _length_truncated(baseline) and not _length_truncated(candidate)
        for baseline, candidate in rows
    )
    candidate_only_truncated = sum(
        not _length_truncated(baseline) and _length_truncated(candidate)
        for baseline, candidate in rows
    )
    both_truncated = sum(
        _length_truncated(baseline) and _length_truncated(candidate)
        for baseline, candidate in rows
    )
    changed = sum(
        baseline.get("predicted_letters") != candidate.get("predicted_letters")
        for baseline, candidate in rows
    )
    baseline_accuracy = baseline_correct / total if total else None
    candidate_accuracy = candidate_correct / total if total else None
    baseline_parse_rate = baseline_parsed / total if total else None
    candidate_parse_rate = candidate_parsed / total if total else None
    baseline_truncation_rate = baseline_truncated / total if total else None
    candidate_truncation_rate = candidate_truncated / total if total else None
    discordant = wins + losses
    return {
        "total": total,
        "baseline": {
            "correct": baseline_correct,
            "accuracy": baseline_accuracy,
            "parsed": baseline_parsed,
            "parse_rate": baseline_parse_rate,
            "length_truncations": baseline_truncated,
            "length_truncation_rate": baseline_truncation_rate,
        },
        "candidate": {
            "correct": candidate_correct,
            "accuracy": candidate_accuracy,
            "parsed": candidate_parsed,
            "parse_rate": candidate_parse_rate,
            "length_truncations": candidate_truncated,
            "length_truncation_rate": candidate_truncation_rate,
        },
        "delta": {
            "correct": candidate_correct - baseline_correct,
            "accuracy": (
                candidate_accuracy - baseline_accuracy
                if candidate_accuracy is not None and baseline_accuracy is not None
                else None
            ),
            "parsed": candidate_parsed - baseline_parsed,
            "parse_rate": (
                candidate_parse_rate - baseline_parse_rate
                if candidate_parse_rate is not None and baseline_parse_rate is not None
                else None
            ),
            "length_truncations": candidate_truncated - baseline_truncated,
            "length_truncation_rate": (
                candidate_truncation_rate - baseline_truncation_rate
                if candidate_truncation_rate is not None and baseline_truncation_rate is not None
                else None
            ),
        },
        "outcomes": {
            "candidate_wins": wins,
            "candidate_losses": losses,
            "ties": ties,
            "ties_both_correct": ties_both_correct,
            "ties_both_wrong": ties_both_wrong,
            "net_wins": wins - losses,
            "discordant": discordant,
            "candidate_win_rate_among_discordant": wins / discordant if discordant else None,
        },
        "parse_transitions": {
            "candidate_wins": parse_wins,
            "candidate_losses": parse_losses,
            "ties": total - parse_wins - parse_losses,
        },
        "length_truncation_transitions": {
            "baseline_only": baseline_only_truncated,
            "candidate_only": candidate_only_truncated,
            "both": both_truncated,
            "neither": total - baseline_only_truncated - candidate_only_truncated - both_truncated,
        },
        "prediction_changed": {
            "count": changed,
            "rate": changed / total if total else None,
        },
    }


def _seed_summary(per_seed: Mapping[str, Mapping[str, Any]], seeds: Sequence[str]) -> dict[str, Any]:
    def values(*path: str) -> list[Any]:
        output: list[Any] = []
        for seed in seeds:
            node: Any = per_seed[seed]
            for key in path:
                node = node.get(key) if isinstance(node, Mapping) else None
            output.append(node)
        return output

    return {
        "baseline_accuracy": _stats(values("baseline", "accuracy")),
        "candidate_accuracy": _stats(values("candidate", "accuracy")),
        "candidate_minus_baseline_accuracy": _stats(values("delta", "accuracy")),
        "baseline_parse_rate": _stats(values("baseline", "parse_rate")),
        "candidate_parse_rate": _stats(values("candidate", "parse_rate")),
        "candidate_minus_baseline_parse_rate": _stats(values("delta", "parse_rate")),
        "baseline_length_truncation_rate": _stats(values("baseline", "length_truncation_rate")),
        "candidate_length_truncation_rate": _stats(values("candidate", "length_truncation_rate")),
        "candidate_minus_baseline_length_truncation_rate": _stats(
            values("delta", "length_truncation_rate")
        ),
        "candidate_win_rate": _stats(
            [
                per_seed[seed]["outcomes"]["candidate_wins"] / per_seed[seed]["total"]
                if per_seed[seed]["total"]
                else None
                for seed in seeds
            ]
        ),
        "candidate_loss_rate": _stats(
            [
                per_seed[seed]["outcomes"]["candidate_losses"] / per_seed[seed]["total"]
                if per_seed[seed]["total"]
                else None
                for seed in seeds
            ]
        ),
    }


def _group_comparison(
    pairs: Mapping[tuple[str, str], tuple[Mapping[str, Any], Mapping[str, Any]]],
    seeds: Sequence[str],
    *,
    predicate: Any,
) -> dict[str, Any]:
    selected = {
        key: value for key, value in pairs.items() if predicate(value[0])
    }
    per_seed = {
        seed: _paired_metric(
            [value for (row_seed, _), value in selected.items() if row_seed == seed]
        )
        for seed in seeds
    }
    return {
        "per_seed": per_seed,
        "seed_summary": _seed_summary(per_seed, seeds),
        "pooled_seed_sample_pairs": _paired_metric(list(selected.values())),
    }


def _weighted_comparison(
    by_primary: Mapping[str, Mapping[str, Any]], seeds: Sequence[str]
) -> dict[str, Any]:
    denominator = sum(A_LEADERBOARD_INFERRED_COUNTS.values())
    weights = {
        dimension: count / denominator
        for dimension, count in A_LEADERBOARD_INFERRED_COUNTS.items()
    }
    per_seed: dict[str, Any] = {}
    baseline_scores: list[float | None] = []
    candidate_scores: list[float | None] = []
    delta_scores: list[float | None] = []
    for seed in seeds:
        missing: list[str] = []
        baseline_score = 0.0
        candidate_score = 0.0
        contributions: dict[str, Any] = {}
        for dimension, weight in weights.items():
            group = by_primary.get(dimension)
            metric = group.get("per_seed", {}).get(seed) if isinstance(group, Mapping) else None
            if not isinstance(metric, Mapping) or not metric.get("total"):
                missing.append(dimension)
                contributions[dimension] = None
                continue
            baseline_accuracy = metric["baseline"]["accuracy"]
            candidate_accuracy = metric["candidate"]["accuracy"]
            baseline_score += float(baseline_accuracy) * weight
            candidate_score += float(candidate_accuracy) * weight
            contributions[dimension] = {
                "weight": weight,
                "total": metric["total"],
                "baseline_accuracy": baseline_accuracy,
                "candidate_accuracy": candidate_accuracy,
                "candidate_minus_baseline_contribution": (
                    float(candidate_accuracy) - float(baseline_accuracy)
                )
                * weight,
            }
        complete = not missing
        baseline_value = baseline_score if complete else None
        candidate_value = candidate_score if complete else None
        delta_value = candidate_score - baseline_score if complete else None
        baseline_scores.append(baseline_value)
        candidate_scores.append(candidate_value)
        delta_scores.append(delta_value)
        per_seed[seed] = {
            "complete": complete,
            "missing_dimensions": missing,
            "baseline_score": baseline_value,
            "candidate_score": candidate_value,
            "candidate_minus_baseline": delta_value,
            "contributions": contributions,
        }
    return {
        "status": "complete" if all(item is not None for item in delta_scores) else "not_applicable",
        "label_policy": "reported_letters",
        "inference_basis": {
            "total_items": denominator,
            "dimension_item_counts": dict(A_LEADERBOARD_INFERRED_COUNTS),
            "dimension_weights": weights,
        },
        "per_seed": per_seed,
        "seed_summary": {
            "baseline_score": _stats(baseline_scores),
            "candidate_score": _stats(candidate_scores),
            "candidate_minus_baseline": _stats(delta_scores),
        },
        "note": (
            "Diagnostic weighting inferred from A-leaderboard increments; it is not an "
            "official evaluation score. Missing primary dimensions are never renormalized."
        ),
    }


def build_comparison(
    baseline_dir: str | Path,
    candidate_dir: str | Path,
    *,
    baseline_name: str = "baseline",
    candidate_name: str = "candidate",
) -> dict[str, Any]:
    """Build a strict paired comparison without modifying either input."""

    baseline_root = Path(baseline_dir).expanduser().resolve()
    candidate_root = Path(candidate_dir).expanduser().resolve()
    baseline_summary = compact_evaluation(baseline_root, name=baseline_name)
    candidate_summary = compact_evaluation(candidate_root, name=candidate_name)
    seeds = list(baseline_summary["seeds"])
    protocol = _same_protocol(
        _read_object(baseline_root / "evaluation_manifest.json"),
        _read_object(candidate_root / "evaluation_manifest.json"),
        seeds,
        candidate_summary["seeds"],
    )
    baseline_rows = _canonical_rows(baseline_root, expected_seeds=seeds)
    candidate_rows = _canonical_rows(candidate_root, expected_seeds=seeds)
    if set(baseline_rows) != set(candidate_rows):
        missing_candidate = sorted(set(baseline_rows) - set(candidate_rows))
        missing_baseline = sorted(set(candidate_rows) - set(baseline_rows))
        raise SummaryError(
            "canonical prediction grids differ: "
            f"missing_in_candidate={missing_candidate[:10]!r}; "
            f"missing_in_baseline={missing_baseline[:10]!r}"
        )
    metadata_fields = (
        "qtype",
        "dimension",
        "gold_letters",
        "mapped_letters",
        "safe_gold",
        "gold_audit_reason",
    )
    pairs: dict[tuple[str, str], tuple[Mapping[str, Any], Mapping[str, Any]]] = {}
    for key in sorted(baseline_rows):
        baseline = baseline_rows[key]
        candidate = candidate_rows[key]
        mismatched = [field for field in metadata_fields if baseline.get(field) != candidate.get(field)]
        if mismatched:
            raise SummaryError(f"paired row metadata differs for key={key!r}: {mismatched}")
        pairs[key] = (baseline, candidate)

    overall = _group_comparison(pairs, seeds, predicate=lambda _: True)
    qtypes = sorted({str(row.get("qtype") or "unknown") for row in baseline_rows.values()})
    by_qtype = {
        qtype: _group_comparison(
            pairs,
            seeds,
            predicate=lambda row, expected=qtype: str(row.get("qtype") or "unknown") == expected,
        )
        for qtype in qtypes
    }
    primary_keys = sorted(
        {
            primary_dimension(row.get("dimension"))
            for row in baseline_rows.values()
            if primary_dimension(row.get("dimension")) != "unknown"
        }
    )
    by_primary = {
        f"{dimension}:{PRIMARY_DIMENSION_NAMES[dimension]}": _group_comparison(
            pairs,
            seeds,
            predicate=lambda row, expected=dimension: primary_dimension(row.get("dimension"))
            == expected,
        )
        for dimension in primary_keys
    }
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "baseline_name": baseline_name,
        "candidate_name": candidate_name,
        "protocol_check": protocol,
        "evaluations": {
            "baseline": baseline_summary,
            "candidate": candidate_summary,
        },
        "paired_comparison": {
            "pairing_unit": "seed_sample",
            "accuracy_policy": "exact reported-letter set match",
            "overall": overall,
            "by_qtype": by_qtype,
            "by_primary_dimension": by_primary,
            "leaderboard_a_inferred_weighted_score": _weighted_comparison(by_primary, seeds),
        },
    }


def _percent(value: Any, *, signed: bool = False) -> str:
    if value is None:
        return "—"
    prefix = "+" if signed and float(value) > 0 else ""
    return f"{prefix}{float(value) * 100:.2f}%"


def _mean_std(value: Mapping[str, Any], *, signed: bool = False) -> str:
    if value.get("mean") is None:
        return "—"
    prefix = "+" if signed and float(value["mean"]) > 0 else ""
    return f"{prefix}{float(value['mean']) * 100:.2f}% ± {float(value.get('std') or 0) * 100:.2f}pp"


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    return [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(row) + " |" for row in rows),
    ]


def render_markdown(report: Mapping[str, Any]) -> str:
    baseline_name = str(report["baseline_name"])
    candidate_name = str(report["candidate_name"])
    paired = report["paired_comparison"]
    overall = paired["overall"]
    seeds = report["protocol_check"]["seeds"]
    lines = [
        f"# SoMBench paired evaluation: {candidate_name} vs {baseline_name}",
        "",
        "Protocol check: **matched**. Pairing unit: `(seed, sample_id)`; accuracy uses exact reported-letter set match.",
        "",
    ]
    seed_rows = []
    for seed in seeds:
        metric = overall["per_seed"][seed]
        seed_rows.append(
            [
                seed,
                _percent(metric["baseline"]["accuracy"]),
                _percent(metric["candidate"]["accuracy"]),
                _percent(metric["delta"]["accuracy"], signed=True),
                str(metric["outcomes"]["candidate_wins"]),
                str(metric["outcomes"]["candidate_losses"]),
                str(metric["outcomes"]["ties"]),
                _percent(metric["baseline"]["parse_rate"]),
                _percent(metric["candidate"]["parse_rate"]),
                f"{metric['baseline']['length_truncations']} / {metric['candidate']['length_truncations']}",
            ]
        )
    lines.extend(
        _table(
            [
                "Seed",
                baseline_name,
                candidate_name,
                "Delta",
                "Wins",
                "Losses",
                "Ties",
                f"{baseline_name} parse",
                f"{candidate_name} parse",
                "Length trunc. B/C",
            ],
            seed_rows,
        )
    )
    summary = overall["seed_summary"]
    pooled = overall["pooled_seed_sample_pairs"]
    lines.extend(
        [
            "",
            f"Three-seed accuracy: **{baseline_name} {_mean_std(summary['baseline_accuracy'])}**, "
            f"**{candidate_name} {_mean_std(summary['candidate_accuracy'])}**, "
            f"paired delta **{_mean_std(summary['candidate_minus_baseline_accuracy'], signed=True)}**.",
            "",
            f"Pooled seed-sample outcomes: wins `{pooled['outcomes']['candidate_wins']}`, "
            f"losses `{pooled['outcomes']['candidate_losses']}`, ties `{pooled['outcomes']['ties']}`; "
            f"parse delta `{_percent(pooled['delta']['parse_rate'], signed=True)}`; "
            f"length-truncation delta `{pooled['delta']['length_truncations']:+d}`.",
        ]
    )
    lines.extend(["", "## Q1/Q2/Q3", ""])
    qtype_rows = []
    for qtype, group in paired["by_qtype"].items():
        q_pooled = group["pooled_seed_sample_pairs"]
        q_summary = group["seed_summary"]
        qtype_rows.append(
            [
                qtype,
                _mean_std(q_summary["baseline_accuracy"]),
                _mean_std(q_summary["candidate_accuracy"]),
                _mean_std(q_summary["candidate_minus_baseline_accuracy"], signed=True),
                str(q_pooled["outcomes"]["candidate_wins"]),
                str(q_pooled["outcomes"]["candidate_losses"]),
                str(q_pooled["outcomes"]["ties"]),
            ]
        )
    lines.extend(
        _table(
            ["QType", baseline_name, candidate_name, "Delta", "Wins", "Losses", "Ties"],
            qtype_rows,
        )
    )
    if paired["by_primary_dimension"]:
        lines.extend(["", "## Primary dimensions", ""])
        primary_rows = []
        for dimension in PRIMARY_DIMENSIONS:
            group = paired["by_primary_dimension"].get(dimension)
            if not group:
                continue
            p_pooled = group["pooled_seed_sample_pairs"]
            p_summary = group["seed_summary"]
            primary_rows.append(
                [
                    dimension,
                    _mean_std(p_summary["baseline_accuracy"]),
                    _mean_std(p_summary["candidate_accuracy"]),
                    _mean_std(p_summary["candidate_minus_baseline_accuracy"], signed=True),
                    str(p_pooled["outcomes"]["candidate_wins"]),
                    str(p_pooled["outcomes"]["candidate_losses"]),
                    str(p_pooled["outcomes"]["ties"]),
                ]
            )
        lines.extend(
            _table(
                ["Primary dimension", baseline_name, candidate_name, "Delta", "Wins", "Losses", "Ties"],
                primary_rows,
            )
        )
        weighted = paired["leaderboard_a_inferred_weighted_score"]
        lines.extend(
            [
                "",
                "## A-leaderboard inferred weighted score",
                "",
                f"{baseline_name}: `{_mean_std(weighted['seed_summary']['baseline_score'])}`; "
                f"{candidate_name}: `{_mean_std(weighted['seed_summary']['candidate_score'])}`; "
                f"delta: `{_mean_std(weighted['seed_summary']['candidate_minus_baseline'], signed=True)}`.",
                "",
                weighted["note"],
            ]
        )
    return "\n".join(lines) + "\n"


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument("--baseline-name", default="baseline")
    parser.add_argument("--candidate-name", default="candidate")
    parser.add_argument("--json-out")
    parser.add_argument("--markdown-out")
    args = parser.parse_args()
    report = build_comparison(
        args.baseline_dir,
        args.candidate_dir,
        baseline_name=args.baseline_name,
        candidate_name=args.candidate_name,
    )
    json_text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    markdown = render_markdown(report)
    if args.json_out:
        _atomic_text(Path(args.json_out).expanduser().resolve(), json_text)
    if args.markdown_out:
        _atomic_text(Path(args.markdown_out).expanduser().resolve(), markdown)
    if not args.json_out and not args.markdown_out:
        print(markdown, end="")


if __name__ == "__main__":
    main()
