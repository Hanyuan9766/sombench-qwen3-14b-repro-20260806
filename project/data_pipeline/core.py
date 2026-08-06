"""SoMBench data audit, cleaning, grouped splitting, and balancing.

The implementation intentionally uses only the Python standard library.  It is
safe to run before the GPU training environment has been created and produces
plain JSON/JSONL artifacts that can be inspected or consumed by datasets/TRL.
"""

from __future__ import annotations

import copy
import hashlib
import heapq
import json
import math
import os
import random
import re
import statistics
import tempfile
import unicodedata
import zlib
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple, Union


BOXED_RE = re.compile(r"\\boxed\s*\{([^{}]*)\}", re.IGNORECASE)
THINK_RE = re.compile(r"<think>(.*?)</think>", re.IGNORECASE | re.DOTALL)
OPTION_RE = re.compile(r"(?m)^\s*([A-Z])\s*[.．、)]\s*")
STORY_RE = re.compile(
    r"(?:^|\n)\s*故事\s*[：:]\s*(.*?)(?=\n\s*问题\s*[：:])",
    re.DOTALL,
)
VALID_LETTER_RE = re.compile(r"^[A-Z]$")


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration for a complete data preparation run."""

    seed: int = 42
    val_fraction: float = 0.10
    fuzzy_story_threshold: float = 0.92
    shingle_size: int = 5
    sketch_size: int = 12
    max_cot_chars: int = 1800
    cot_style: str = "compressed"
    balance_target: str = "median"
    safe_dev_policy: str = "agreement_only"
    safe_dev_require_complete_story: bool = False
    allow_pred_mismatch: bool = False

    def validate(self) -> None:
        if not 0.0 < self.val_fraction < 1.0:
            raise ValueError("val_fraction must be between 0 and 1")
        if not 0.0 < self.fuzzy_story_threshold <= 1.0:
            raise ValueError("fuzzy_story_threshold must be in (0, 1]")
        if self.shingle_size < 2:
            raise ValueError("shingle_size must be at least 2")
        if self.sketch_size < 2:
            raise ValueError("sketch_size must be at least 2")
        if self.max_cot_chars < 0:
            raise ValueError("max_cot_chars cannot be negative")
        if self.cot_style not in {"compressed", "answer_only"}:
            raise ValueError("cot_style must be compressed or answer_only")
        if self.safe_dev_policy not in {"agreement_only", "mapped_answer_text"}:
            raise ValueError(
                "safe_dev_policy must be agreement_only or mapped_answer_text"
            )
        _parse_balance_target(self.balance_target, [1])


PathLike = Union[os.PathLike, str]


def read_jsonl(path: PathLike) -> List[Dict[str, Any]]:
    """Read a JSONL file and report the exact malformed line, if any."""

    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"expected an object at {path}:{line_number}")
            item = dict(item)
            item.setdefault("_source_line", line_number)
            records.append(item)
    return records


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def write_jsonl(path: PathLike, records: Iterable[Mapping[str, Any]]) -> None:
    payload = "".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        for record in records
    )
    _atomic_write_text(Path(path), payload)


def write_json(path: PathLike, value: Mapping[str, Any]) -> None:
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    _atomic_write_text(Path(path), payload)


def file_sha256(path: PathLike) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _message_content(record: Mapping[str, Any], role: str) -> str:
    messages = record.get("messages")
    if not isinstance(messages, list):
        return ""
    values = [
        message.get("content", "")
        for message in messages
        if isinstance(message, dict) and message.get("role") == role
    ]
    return values[-1] if values and isinstance(values[-1], str) else ""


def extract_story_from_prompt(prompt: str) -> str:
    match = STORY_RE.search(prompt)
    if match:
        return match.group(1).strip()
    # A conservative fallback supports small hand-authored fixtures and future
    # prompts without silently folding the answer choices into the story.
    if "问题：" in prompt:
        return prompt.split("问题：", 1)[0].replace("故事：", "", 1).strip()
    return ""


def extract_option_letters(prompt: str) -> List[str]:
    options_section = prompt
    for marker in ("选项：", "选项:"):
        if marker in prompt:
            options_section = prompt.rsplit(marker, 1)[-1]
            break
    return sorted(set(OPTION_RE.findall(options_section)))


def normalize_story(story: str) -> str:
    text = unicodedata.normalize("NFKC", story).lower()
    # Numbering is presentation rather than story identity.  Keeping only
    # letters/numbers also makes whitespace and punctuation edits clusterable.
    return "".join(character for character in text if character.isalnum())


def _story_exact_hash(normalized_story: str) -> str:
    return hashlib.sha256(normalized_story.encode("utf-8")).hexdigest()


def _hashed_shingles(text: str, size: int) -> Set[int]:
    if len(text) <= size:
        return {zlib.crc32(text.encode("utf-8")) & 0xFFFFFFFF}
    return {
        zlib.crc32(text[index : index + size].encode("utf-8")) & 0xFFFFFFFF
        for index in range(len(text) - size + 1)
    }


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def assign_story_clusters(
    records: Sequence[MutableMapping[str, Any]],
    *,
    threshold: float = 0.92,
    shingle_size: int = 5,
    sketch_size: int = 12,
    prefix: str = "story",
) -> Dict[str, Any]:
    """Assign deterministic normalized/fuzzy story cluster IDs in-place.

    Exact normalized stories are collapsed first.  For fuzzy matching, a
    bottom-k shingle sketch provides candidates and full shingle Jaccard is the
    final guard.  This captures punctuation edits and close rewrites without an
    O(n^2) all-pairs scan.
    """

    if not 0.0 < threshold <= 1.0:
        raise ValueError("threshold must be in (0, 1]")

    normalized_to_rows: Dict[str, List[int]] = defaultdict(list)
    for row_index, record in enumerate(records):
        story = record.get("story", "")
        normalized = normalize_story(story if isinstance(story, str) else "")
        if not normalized:
            # The unique placeholder prevents unrelated missing stories from
            # leaking across a grouped split.
            normalized = f"__missing_story_{row_index}"
        normalized_to_rows[normalized].append(row_index)

    normalized_stories = sorted(
        normalized_to_rows,
        key=lambda value: _story_exact_hash(value),
    )
    exact_hashes = [_story_exact_hash(value) for value in normalized_stories]
    union_find = _UnionFind(len(normalized_stories))
    shingle_sets: List[Set[int]] = []
    sketches: List[List[int]] = []
    inverted_sketch: Dict[int, List[int]] = defaultdict(list)
    compared_pairs = 0

    for index, normalized in enumerate(normalized_stories):
        shingles = _hashed_shingles(normalized, shingle_size)
        shingle_sets.append(shingles)
        sketch = heapq.nsmallest(sketch_size, shingles)
        sketches.append(sketch)

        if threshold < 1.0:
            candidate_counts: Counter[int] = Counter()
            for signature in sketch:
                candidate_counts.update(inverted_sketch.get(signature, ()))
            minimum_shared = 1 if len(sketch) < 4 else 2
            for candidate, shared in candidate_counts.items():
                if shared < minimum_shared:
                    continue
                candidate_text = normalized_stories[candidate]
                length_ratio = min(len(normalized), len(candidate_text)) / max(
                    len(normalized), len(candidate_text), 1
                )
                if length_ratio < threshold:
                    continue
                compared_pairs += 1
                candidate_shingles = shingle_sets[candidate]
                similarity = len(shingles & candidate_shingles) / max(
                    len(shingles | candidate_shingles), 1
                )
                if similarity >= threshold:
                    union_find.union(index, candidate)

        for signature in sketch:
            inverted_sketch[signature].append(index)

    root_to_members: Dict[int, List[int]] = defaultdict(list)
    for index in range(len(normalized_stories)):
        root_to_members[union_find.find(index)].append(index)

    node_to_cluster: Dict[int, str] = {}
    for members in root_to_members.values():
        representative_hash = min(exact_hashes[member] for member in members)
        cluster_id = f"{prefix}_{representative_hash[:16]}"
        for member in members:
            node_to_cluster[member] = cluster_id

    normalized_to_node = {value: index for index, value in enumerate(normalized_stories)}
    for normalized, row_indexes in normalized_to_rows.items():
        node = normalized_to_node[normalized]
        for row_index in row_indexes:
            records[row_index]["story_cluster_id"] = node_to_cluster[node]
            records[row_index]["story_exact_id"] = f"exact_{exact_hashes[node][:16]}"

    return {
        "rows": len(records),
        "exact_normalized_stories": len(normalized_stories),
        "heuristic_story_clusters": len(root_to_members),
        "fuzzy_merges": len(normalized_stories) - len(root_to_members),
        "candidate_pairs_compared": compared_pairs,
        "threshold": threshold,
        "shingle_size": shingle_size,
        "sketch_size": sketch_size,
    }


def _normalize_letters(value: Any) -> List[str]:
    if isinstance(value, str):
        candidates = re.findall(r"[A-Za-z]", value.upper())
    elif isinstance(value, (list, tuple, set)):
        candidates = [str(item).strip().upper() for item in value]
    else:
        return []
    return sorted({letter for letter in candidates if VALID_LETTER_RE.fullmatch(letter)})


def _strip_answer_markup(text: str) -> str:
    text = BOXED_RE.sub("", text)
    text = re.sub(r"</?think>", "", text, flags=re.IGNORECASE)
    text = text.replace("$$", " ").replace(r"\[", " ").replace(r"\]", " ")
    text = re.sub(r"(?m)^\s*[-*_]{3,}\s*$", "", text)
    text = re.sub(r"(?m)^\s*#{1,6}\s*", "", text)
    text = re.sub(r"(?m)^\s*(?:最终答案|答案为|最终选择)\s*[：:]?.*$", "", text)
    return text


def _compress_rationale(text: str, max_chars: int) -> str:
    text = _strip_answer_markup(text)
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    units = re.split(r"(?<=[。！？!?；;])\s*|\n+", text)
    kept: List[str] = []
    seen: Set[str] = set()
    filler = re.compile(
        r"^(?:嗯|好的|让我们|我来|这个问题(?:看起来)?|首先我(?:需要|得)|题目要求我们)"
    )
    for raw_unit in units:
        unit = raw_unit.strip(" \t-*#")
        if not unit or filler.search(unit):
            continue
        key = "".join(character for character in unit if character.isalnum()).lower()
        if not key or key in seen:
            continue
        seen.add(key)
        kept.append(unit)

    rationale = "\n".join(kept).strip()
    if max_chars and len(rationale) > max_chars:
        truncated = rationale[:max_chars]
        boundary = max(truncated.rfind("。"), truncated.rfind("！"), truncated.rfind("？"))
        if boundary >= max_chars // 2:
            truncated = truncated[: boundary + 1]
        rationale = truncated.rstrip() + "…"
    return rationale


def clean_assistant_answer(
    raw_answer: str,
    gold_letters: Sequence[str],
    *,
    max_cot_chars: int = 1800,
    cot_style: str = "compressed",
) -> str:
    """Return one canonical final box and, optionally, a compact rationale."""

    letters = _normalize_letters(gold_letters)
    if not letters:
        raise ValueError("gold_letters cannot be empty")
    final_box = r"\boxed{" + ",".join(letters) + "}"
    if cot_style == "answer_only" or max_cot_chars == 0:
        return final_box
    if cot_style != "compressed":
        raise ValueError("cot_style must be compressed or answer_only")

    think_parts = THINK_RE.findall(raw_answer)
    after_think = re.sub(THINK_RE, "", raw_answer)
    polished = _compress_rationale(after_think, max_cot_chars)
    internal = _compress_rationale("\n".join(think_parts), max_cot_chars)
    rationale = polished if len(polished) >= 80 else internal
    if not rationale:
        rationale = "根据故事中的信息可见性、人物信念与题目条件逐项核对。"
    return f"<think>\n{rationale}\n</think>\n\n{final_box}"


def _distribution(values: Iterable[Any]) -> Dict[str, int]:
    return dict(sorted((str(key), count) for key, count in Counter(values).items()))


def _length_summary(values: Sequence[int]) -> Dict[str, Any]:
    if not values:
        return {"count": 0}
    ordered = sorted(values)

    def percentile(fraction: float) -> int:
        index = min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)
        return ordered[max(index, 0)]

    return {
        "count": len(values),
        "min": ordered[0],
        "median": statistics.median(ordered),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "max": ordered[-1],
    }


def _duplicate_audit(values: Iterable[Any]) -> Dict[str, Any]:
    counter = Counter(value for value in values if value is not None)
    duplicates = {str(key): count for key, count in counter.items() if count > 1}
    return {
        "unique_non_null": len(counter),
        "duplicate_values": len(duplicates),
        "rows_in_duplicate_values": sum(duplicates.values()),
    }


def clean_training_records(
    raw_records: Sequence[Mapping[str, Any]],
    config: PipelineConfig,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """Validate and clean training chats, returning clean/rejected/audit."""

    config.validate()
    cleaned: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    rejection_reasons: Counter[str] = Counter()
    option_counts: Counter[int] = Counter()
    raw_answer_lengths: List[int] = []
    clean_answer_lengths: List[int] = []
    raw_box_counts: Counter[int] = Counter()
    qtypes_before: List[str] = []
    dims_before: List[str] = []
    gold_pred_mismatches = 0

    for ordinal, source in enumerate(raw_records, 1):
        record = copy.deepcopy(dict(source))
        source_line = record.pop("_source_line", ordinal)
        meta = record.get("meta") if isinstance(record.get("meta"), dict) else {}
        qtype = str(meta.get("qtype", ""))
        dimension = str(meta.get("dim", ""))
        qtypes_before.append(qtype or "missing")
        dims_before.append(dimension or "missing")

        user_prompt = _message_content(record, "user")
        raw_answer = _message_content(record, "assistant")
        story = extract_story_from_prompt(user_prompt)
        option_letters = extract_option_letters(user_prompt)
        option_counts[len(option_letters)] += 1
        gold_letters = _normalize_letters(record.get("gold_letters"))
        pred_letters = _normalize_letters(record.get("pred_letters"))
        reasons: List[str] = []

        if not isinstance(record.get("messages"), list) or not user_prompt or not raw_answer:
            reasons.append("invalid_or_missing_chat")
        if not story:
            reasons.append("story_extraction_failed")
        if not dimension:
            reasons.append("missing_dimension")
        if qtype == "Q2" and ("F" in option_letters or len(option_letters) > 5):
            reasons.append("legacy_q2_six_options")
        if not gold_letters:
            reasons.append("missing_gold_letters")
        elif option_letters and not set(gold_letters).issubset(option_letters):
            reasons.append("gold_outside_options")
        if pred_letters and pred_letters != gold_letters:
            gold_pred_mismatches += 1
            if not config.allow_pred_mismatch:
                reasons.append("gold_pred_mismatch")

        raw_answer_lengths.append(len(raw_answer))
        raw_box_counts[len(BOXED_RE.findall(raw_answer))] += 1
        if reasons:
            rejection_reasons.update(set(reasons))
            rejected.append(
                {
                    "source_line": source_line,
                    "sample_id": record.get("sample_id"),
                    "reasons": sorted(set(reasons)),
                    "record": record,
                }
            )
            continue

        canonical_answer = clean_assistant_answer(
            raw_answer,
            gold_letters,
            max_cot_chars=config.max_cot_chars,
            cot_style=config.cot_style,
        )
        messages = copy.deepcopy(record["messages"])
        for message in reversed(messages):
            if isinstance(message, dict) and message.get("role") == "assistant":
                message["content"] = canonical_answer
                break

        record["messages"] = messages
        record["story"] = story
        record["dimension"] = dimension
        record["question_type"] = qtype
        record["option_letters"] = option_letters
        record["gold_letters"] = gold_letters
        record["pred_letters"] = pred_letters
        record["_pipeline"] = {
            "schema_version": 1,
            "source_line": source_line,
            "teacher_answer_cleaned": True,
            "original_answer_chars": len(raw_answer),
            "clean_answer_chars": len(canonical_answer),
            "original_boxed_count": len(BOXED_RE.findall(raw_answer)),
        }
        clean_answer_lengths.append(len(canonical_answer))
        cleaned.append(record)

    cluster_audit = assign_story_clusters(
        cleaned,
        threshold=config.fuzzy_story_threshold,
        shingle_size=config.shingle_size,
        sketch_size=config.sketch_size,
    )
    audit = {
        "input_rows": len(raw_records),
        "accepted_rows": len(cleaned),
        "rejected_rows": len(rejected),
        "rejection_reasons": dict(sorted(rejection_reasons.items())),
        "qtype_before": _distribution(qtypes_before),
        "qtype_after": _distribution(record["question_type"] for record in cleaned),
        "dimensions_before": {
            "count": len(set(dims_before) - {"missing"}),
            "distribution": _distribution(dims_before),
        },
        "dimensions_after": {
            "count": len({record["dimension"] for record in cleaned}),
            "distribution": _distribution(record["dimension"] for record in cleaned),
        },
        "option_count_before": dict(
            sorted((str(key), value) for key, value in option_counts.items())
        ),
        "sample_id": _duplicate_audit(record.get("sample_id") for record in raw_records),
        "sample_index": _duplicate_audit(
            record.get("sample_index") for record in raw_records
        ),
        "gold_pred_mismatches": gold_pred_mismatches,
        "raw_boxed_count": dict(
            sorted((str(key), value) for key, value in raw_box_counts.items())
        ),
        "raw_answer_chars": _length_summary(raw_answer_lengths),
        "clean_answer_chars": _length_summary(clean_answer_lengths),
        "story_clustering": cluster_audit,
    }
    return cleaned, rejected, audit


def grouped_train_dev_split(
    records: Sequence[Mapping[str, Any]],
    *,
    val_fraction: float = 0.10,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """Create a dimension-aware split while keeping each story cluster intact."""

    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")
    groups: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for index, record in enumerate(records):
        cluster = str(record.get("story_cluster_id") or f"missing_cluster_{index}")
        groups[cluster].append(record)

    target_rows = max(1, round(len(records) * val_fraction)) if records else 0
    total_dims = Counter(str(record.get("dimension", "missing")) for record in records)
    target_dims = {dimension: count * val_fraction for dimension, count in total_dims.items()}
    group_dims = {
        cluster: Counter(str(record.get("dimension", "missing")) for record in rows)
        for cluster, rows in groups.items()
    }
    remaining = set(groups)
    dev_clusters: Set[str] = set()
    dev_dims: Counter[str] = Counter()
    dev_rows = 0

    def tie_breaker(cluster: str) -> int:
        return int.from_bytes(
            hashlib.blake2b(f"{seed}:{cluster}".encode(), digest_size=8).digest(),
            "big",
        )

    def select_cluster(cluster: str) -> None:
        nonlocal dev_rows
        dev_clusters.add(cluster)
        remaining.remove(cluster)
        dev_rows += len(groups[cluster])
        dev_dims.update(group_dims[cluster])

    # First solve a constrained set-cover problem so the dev diagnostics include
    # every fine-grained dimension that can be represented without removing that
    # dimension entirely from train.  The later fill phase approaches the row
    # target while preserving these coverage guarantees.
    remaining_cluster_counts: Counter[str] = Counter()
    for dimensions in group_dims.values():
        remaining_cluster_counts.update(dimensions.keys())
    uncovered_dimensions = set(total_dims)
    impossible_dev_dimensions: Set[str] = set()
    while uncovered_dimensions:
        candidates: List[Tuple[float, int, int, int, str]] = []
        for cluster in remaining:
            covered = uncovered_dimensions & set(group_dims[cluster])
            if not covered:
                continue
            # Every dimension touched by this cluster must retain at least one
            # distinct story cluster on the train side.
            if any(
                remaining_cluster_counts[dimension] <= 1
                for dimension in group_dims[cluster]
            ):
                continue
            row_count = len(groups[cluster])
            candidates.append(
                (
                    len(covered) / max(math.sqrt(row_count), 1.0),
                    len(covered),
                    -row_count,
                    -tie_breaker(cluster),
                    cluster,
                )
            )
        if not candidates:
            impossible_dev_dimensions = set(uncovered_dimensions)
            break
        chosen = max(candidates)[-1]
        newly_covered = uncovered_dimensions & set(group_dims[chosen])
        select_cluster(chosen)
        uncovered_dimensions -= newly_covered
        for dimension in group_dims[chosen]:
            remaining_cluster_counts[dimension] -= 1

    while remaining and dev_rows < target_rows:
        best_cluster: Optional[str] = None
        best_score: Optional[Tuple[float, float, int]] = None
        for cluster in remaining:
            row_count = len(groups[cluster])
            benefit = 0.0
            oversupply = 0.0
            for dimension, count in group_dims[cluster].items():
                target = max(target_dims[dimension], 1e-9)
                need = max(target_dims[dimension] - dev_dims[dimension], 0.0)
                benefit += min(float(count), need) / target
                oversupply += max(float(count) - need, 0.0) / target
            row_overshoot = max(dev_rows + row_count - target_rows, 0) / max(target_rows, 1)
            score = (
                benefit - 0.20 * oversupply - 2.0 * row_overshoot,
                -abs(target_rows - (dev_rows + row_count)),
                -tie_breaker(cluster),
            )
            if best_score is None or score > best_score:
                best_score = score
                best_cluster = cluster
        assert best_cluster is not None
        prospective = dev_rows + len(groups[best_cluster])
        if dev_rows > 0 and abs(target_rows - dev_rows) <= abs(target_rows - prospective):
            break
        select_cluster(best_cluster)

    # A non-empty dev set is required even when a single large cluster crosses
    # the requested target.
    if records and not dev_clusters:
        chosen = min(groups, key=lambda cluster: (len(groups[cluster]), tie_breaker(cluster)))
        dev_clusters.add(chosen)

    train: List[Dict[str, Any]] = []
    dev: List[Dict[str, Any]] = []
    for record in records:
        destination = dev if record.get("story_cluster_id") in dev_clusters else train
        destination.append(copy.deepcopy(dict(record)))

    random.Random(seed).shuffle(train)
    random.Random(seed + 1).shuffle(dev)
    train_clusters = {str(record.get("story_cluster_id")) for record in train}
    actual_dev_clusters = {str(record.get("story_cluster_id")) for record in dev}
    overlap = train_clusters & actual_dev_clusters
    audit = {
        "requested_dev_fraction": val_fraction,
        "actual_dev_fraction": len(dev) / len(records) if records else 0.0,
        "train_rows": len(train),
        "dev_rows": len(dev),
        "train_clusters": len(train_clusters),
        "dev_clusters": len(actual_dev_clusters),
        "cluster_overlap": sorted(overlap),
        "train_dimensions": _distribution(record.get("dimension") for record in train),
        "dev_dimensions": _distribution(record.get("dimension") for record in dev),
        "missing_train_dimensions": sorted(set(total_dims) - {
            str(record.get("dimension", "missing")) for record in train
        }),
        "missing_dev_dimensions": sorted(set(total_dims) - {
            str(record.get("dimension", "missing")) for record in dev
        }),
        "dev_dimensions_impossible_without_emptying_train": sorted(
            impossible_dev_dimensions
        ),
    }
    if overlap:
        raise AssertionError("story clusters leaked across train/dev")
    return train, dev, audit


def _parse_balance_target(target: str, counts: Sequence[int]) -> Optional[int]:
    normalized = str(target).strip().lower()
    if normalized in {"none", "off", "disabled"}:
        return None
    if not counts:
        return 0
    if normalized == "median":
        return max(1, round(statistics.median(counts)))
    if normalized == "max":
        return max(counts)
    if normalized == "min":
        return min(counts)
    try:
        numeric = int(normalized)
    except ValueError as exc:
        raise ValueError("balance_target must be none, min, median, max, or a positive integer") from exc
    if numeric <= 0:
        raise ValueError("numeric balance_target must be positive")
    return numeric


def balance_by_dimension(
    records: Sequence[Mapping[str, Any]],
    *,
    target: str = "median",
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Deterministically under/over-sample each present fine-grained dimension."""

    by_dimension: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        by_dimension[str(record.get("dimension", "missing"))].append(record)
    counts_before = {dimension: len(rows) for dimension, rows in by_dimension.items()}
    target_count = _parse_balance_target(target, list(counts_before.values()))
    if target_count is None:
        untouched = [copy.deepcopy(dict(record)) for record in records]
        return untouched, {
            "strategy": "none",
            "target_per_dimension": None,
            "rows_before": len(records),
            "rows_after": len(untouched),
            "counts_before": dict(sorted(counts_before.items())),
            "counts_after": dict(sorted(counts_before.items())),
        }

    balanced: List[Dict[str, Any]] = []
    for dimension in sorted(by_dimension):
        source_rows = list(by_dimension[dimension])
        rng = random.Random(f"{seed}:{dimension}")
        rng.shuffle(source_rows)
        selected: List[Mapping[str, Any]] = []
        cycle = 0
        while len(selected) < target_count:
            if cycle:
                rng.shuffle(source_rows)
            selected.extend(source_rows[: target_count - len(selected)])
            cycle += 1
        occurrence_by_source: Counter[str] = Counter()
        for row in selected:
            copied = copy.deepcopy(dict(row))
            source_key = str(
                copied.get("sample_id")
                or copied.get("_pipeline", {}).get("source_line")
                or len(balanced)
            )
            occurrence = occurrence_by_source[source_key]
            occurrence_by_source[source_key] += 1
            pipeline_meta = copied.setdefault("_pipeline", {})
            pipeline_meta["sampling_occurrence"] = occurrence
            pipeline_meta["sample_instance_id"] = hashlib.sha256(
                f"{dimension}:{source_key}:{occurrence}".encode()
            ).hexdigest()[:20]
            balanced.append(copied)

    random.Random(seed + 2).shuffle(balanced)
    counts_after = Counter(str(record.get("dimension")) for record in balanced)
    audit = {
        "strategy": str(target),
        "target_per_dimension": target_count,
        "rows_before": len(records),
        "rows_after": len(balanced),
        "counts_before": dict(sorted(counts_before.items())),
        "counts_after": dict(sorted(counts_after.items())),
    }
    return balanced, audit


def normalize_answer_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(text)).lower()
    return "".join(character for character in normalized if character.isalnum())


def _map_answer_texts_to_letters(
    options: Mapping[str, Any], answers: Sequence[Any]
) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
    normalized_options: Dict[str, List[str]] = defaultdict(list)
    for letter, option_text in options.items():
        normalized_options[normalize_answer_text(str(option_text))].append(str(letter).upper())

    mapped: List[str] = []
    unmapped: List[str] = []
    ambiguous: List[Dict[str, Any]] = []
    for answer in answers:
        answer_text = str(answer)
        candidates = sorted(set(normalized_options.get(normalize_answer_text(answer_text), [])))
        if len(candidates) == 1:
            mapped.append(candidates[0])
        elif not candidates:
            unmapped.append(answer_text)
        else:
            ambiguous.append({"answer": answer_text, "candidate_letters": candidates})
    return sorted(set(mapped)), unmapped, ambiguous


def audit_public_records(
    raw_records: Sequence[Mapping[str, Any]],
    config: PipelineConfig,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """Reconcile public-test labels and build a conservative safe dev set."""

    config.validate()
    annotated: List[Dict[str, Any]] = []
    for ordinal, source in enumerate(raw_records, 1):
        record = copy.deepcopy(dict(source))
        source_line = record.pop("_source_line", ordinal)
        story = record.get("story") if isinstance(record.get("story"), str) else ""
        options = record.get("options") if isinstance(record.get("options"), dict) else {}
        answers = record.get("correct_answers")
        answers = answers if isinstance(answers, list) else []
        declared = _normalize_letters(record.get("correct_letters"))
        mapped, unmapped, ambiguous = _map_answer_texts_to_letters(options, answers)

        issues: List[str] = []
        if not options or not answers or not declared:
            issues.append("missing_label_fields")
        if unmapped:
            issues.append("answer_text_unmapped")
        if ambiguous:
            issues.append("answer_text_ambiguous")
        if not unmapped and not ambiguous and mapped != declared:
            issues.append("letters_answers_conflict")
        task_format = str(record.get("task_format", ""))
        if task_format == "mcq_single" and max(len(declared), len(mapped)) > 1:
            issues.append("single_format_with_multiple_answers")

        if "missing_label_fields" in issues:
            label_status = "invalid"
        elif "answer_text_unmapped" in issues:
            label_status = "unmapped"
        elif "answer_text_ambiguous" in issues:
            label_status = "ambiguous"
        elif "letters_answers_conflict" in issues:
            label_status = "conflict"
        else:
            label_status = "agree"

        record["story"] = story
        record["_label_audit"] = {
            "source_line": source_line,
            "status": label_status,
            "issues": issues,
            "declared_letters": declared,
            "answer_text_mapped_letters": mapped,
            "unmapped_answers": unmapped,
            "ambiguous_answers": ambiguous,
            "manual_letters": None,
        }
        annotated.append(record)

    cluster_audit = assign_story_clusters(
        annotated,
        threshold=config.fuzzy_story_threshold,
        shingle_size=config.shingle_size,
        sketch_size=config.sketch_size,
        prefix="public_story",
    )

    safe_candidates: List[Dict[str, Any]] = []
    conflicts: List[Dict[str, Any]] = []
    for record in annotated:
        label_audit = record["_label_audit"]
        status = label_audit["status"]
        format_safe = "single_format_with_multiple_answers" not in label_audit["issues"]
        if config.safe_dev_policy == "agreement_only":
            label_safe = status == "agree"
            resolved = label_audit["declared_letters"]
            source_name = "declared_and_answer_text_agree"
        else:
            label_safe = status in {"agree", "conflict"}
            resolved = label_audit["answer_text_mapped_letters"]
            source_name = "exact_answer_text_mapping"

        if label_safe and format_safe:
            safe_record = copy.deepcopy(record)
            safe_record["original_correct_letters"] = label_audit["declared_letters"]
            safe_record["correct_letters"] = list(resolved)
            safe_record["evaluation_letters"] = list(resolved)
            safe_record["evaluation_label_source"] = source_name
            safe_candidates.append(safe_record)
        else:
            conflict_record = copy.deepcopy(record)
            conflict_record["requires_manual_review"] = True
            conflicts.append(conflict_record)

    if config.safe_dev_require_complete_story:
        source_counts = Counter(record["story_cluster_id"] for record in annotated)
        safe_counts = Counter(record["story_cluster_id"] for record in safe_candidates)
        complete_clusters = {
            cluster
            for cluster, count in source_counts.items()
            if safe_counts.get(cluster, 0) == count
        }
        retained: List[Dict[str, Any]] = []
        for record in safe_candidates:
            if record["story_cluster_id"] in complete_clusters:
                retained.append(record)
            else:
                rejected = copy.deepcopy(record)
                rejected["_label_audit"]["issues"].append("incomplete_safe_story_bundle")
                rejected["requires_manual_review"] = True
                conflicts.append(rejected)
        safe_candidates = retained

    statuses = Counter(record["_label_audit"]["status"] for record in annotated)
    issue_counts = Counter(
        issue for record in annotated for issue in record["_label_audit"]["issues"]
    )
    audit = {
        "input_rows": len(raw_records),
        "safe_dev_rows": len(safe_candidates),
        "manual_review_rows": len(conflicts),
        "safe_dev_policy": config.safe_dev_policy,
        "require_complete_story": config.safe_dev_require_complete_story,
        "label_status": dict(sorted(statuses.items())),
        "issues": dict(sorted(issue_counts.items())),
        "question_types": _distribution(record.get("question_type") for record in annotated),
        "task_formats": _distribution(record.get("task_format") for record in annotated),
        "story_clustering": cluster_audit,
    }
    return safe_candidates, conflicts, audit


def _config_dict(config: PipelineConfig) -> Dict[str, Any]:
    return asdict(config)


def run_pipeline(
    train_path: PathLike,
    public_test_path: PathLike,
    output_dir: PathLike,
    config: Optional[PipelineConfig] = None,
    *,
    force: bool = False,
) -> Dict[str, Any]:
    """Run the entire pipeline and atomically write all declared outputs."""

    config = config or PipelineConfig()
    config.validate()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    output_names = {
        "clean_all": "train_clean_all.jsonl",
        "train": "train.jsonl",
        "dev": "dev.jsonl",
        "balanced_train": "train_balanced.jsonl",
        "rejected_train": "train_rejected.jsonl",
        "public_safe_dev": "public_safe_dev.jsonl",
        "public_conflicts": "public_label_conflicts.jsonl",
        "audit": "audit.json",
    }
    existing = [output / name for name in output_names.values() if (output / name).exists()]
    if existing and not force:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(f"refusing to overwrite existing outputs: {names}; pass force=True")

    raw_train = read_jsonl(train_path)
    raw_public = read_jsonl(public_test_path)
    clean, rejected, training_audit = clean_training_records(raw_train, config)
    train, dev, split_audit = grouped_train_dev_split(
        clean,
        val_fraction=config.val_fraction,
        seed=config.seed,
    )
    balanced_train, balance_audit = balance_by_dimension(
        train,
        target=config.balance_target,
        seed=config.seed,
    )
    public_safe, public_conflicts, public_audit = audit_public_records(raw_public, config)

    audit: Dict[str, Any] = {
        "schema_version": 1,
        "config": _config_dict(config),
        "inputs": {
            "train": {
                "path": str(Path(train_path).resolve()),
                "sha256": file_sha256(train_path),
            },
            "public_test": {
                "path": str(Path(public_test_path).resolve()),
                "sha256": file_sha256(public_test_path),
            },
        },
        "training": training_audit,
        "split": split_audit,
        "balancing": balance_audit,
        "public_test": public_audit,
        "outputs": output_names,
        "invariants": {
            "train_dev_cluster_overlap": len(split_audit["cluster_overlap"]),
            "clean_answers_with_noncanonical_box_count": sum(
                len(BOXED_RE.findall(_message_content(record, "assistant"))) != 1
                for record in clean
            ),
            "balanced_dimension_count": len(balance_audit["counts_after"]),
        },
    }

    write_jsonl(output / output_names["clean_all"], clean)
    write_jsonl(output / output_names["train"], train)
    write_jsonl(output / output_names["dev"], dev)
    write_jsonl(output / output_names["balanced_train"], balanced_train)
    write_jsonl(output / output_names["rejected_train"], rejected)
    write_jsonl(output / output_names["public_safe_dev"], public_safe)
    write_jsonl(output / output_names["public_conflicts"], public_conflicts)
    write_json(output / output_names["audit"], audit)
    return audit
