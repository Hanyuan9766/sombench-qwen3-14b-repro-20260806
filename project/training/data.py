"""Chat JSONL loading and assistant-only tokenization."""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

LOGGER = logging.getLogger(__name__)
IGNORE_INDEX = -100


class DataError(ValueError):
    """Raised when a training record cannot be represented safely."""


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield non-empty JSON objects, reporting an exact line on failure."""

    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DataError(f"{source}:{line_number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(value, dict):
                raise DataError(f"{source}:{line_number}: expected JSON object")
            yield value


def _message_content(message: Mapping[str, Any]) -> str:
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise DataError("each chat message must contain non-empty string content")
    return content


def validate_messages(record: Mapping[str, Any]) -> list[dict[str, str]]:
    """Validate and normalize the OpenAI-style messages in one record."""

    raw_messages = record.get("messages")
    if not isinstance(raw_messages, list) or not raw_messages:
        raise DataError("record.messages must be a non-empty list")
    messages: list[dict[str, str]] = []
    assistant_count = 0
    for index, message in enumerate(raw_messages):
        if not isinstance(message, Mapping):
            raise DataError(f"messages[{index}] must be an object")
        role = message.get("role")
        if role not in {"system", "user", "assistant"}:
            raise DataError(f"messages[{index}].role={role!r} is not supported for SFT")
        if role == "assistant":
            assistant_count += 1
        messages.append({"role": str(role), "content": _message_content(message)})
    if assistant_count == 0:
        raise DataError("record contains no assistant response to train on")
    if messages[-1]["role"] != "assistant":
        raise DataError("the final message must be an assistant response")
    return messages


def _as_ids(value: Any) -> list[int]:
    if isinstance(value, Mapping):
        value = value.get("input_ids")
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], list):
        if len(value) != 1:
            raise DataError("unexpected batched tokenization result")
        value = value[0]
    if not isinstance(value, list) or not all(isinstance(item, int) for item in value):
        raise DataError("chat template did not return a flat token id list")
    return value


def _render_ids(tokenizer: Any, messages: Sequence[Mapping[str, str]], *, generation_prompt: bool) -> list[int]:
    return _as_ids(
        tokenizer.apply_chat_template(
            list(messages),
            tokenize=True,
            add_generation_prompt=generation_prompt,
            return_tensors=None,
        )
    )


def _is_prefix(prefix: Sequence[int], full: Sequence[int]) -> bool:
    return len(prefix) <= len(full) and list(prefix) == list(full[: len(prefix)])


def assistant_only_labels(tokenizer: Any, messages: Sequence[Mapping[str, str]]) -> tuple[list[int], list[int]]:
    """Render a chat and mask every non-assistant token.

    For each assistant turn, the chat prefix rendered with
    ``add_generation_prompt=True`` identifies the first response token.  The
    prefix through that assistant turn identifies its end.  This relies on the
    causal-prefix property of the Qwen3 chat template and fails closed if a
    custom template violates it.
    """

    input_ids = _render_ids(tokenizer, messages, generation_prompt=False)
    labels = [IGNORE_INDEX] * len(input_ids)
    for index, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        response_start_ids = _render_ids(tokenizer, messages[:index], generation_prompt=True)
        response_end_ids = _render_ids(tokenizer, messages[: index + 1], generation_prompt=False)
        if not _is_prefix(response_start_ids, input_ids):
            raise DataError(
                "chat template is not prefix-stable before the assistant response; "
                "use the unmodified Qwen3 tokenizer/chat_template"
            )
        if not _is_prefix(response_end_ids, input_ids):
            raise DataError(
                "chat template is not prefix-stable after the assistant response; "
                "multi-turn assistant-only masking would be ambiguous"
            )
        start, end = len(response_start_ids), len(response_end_ids)
        if end <= start:
            raise DataError("assistant response produced no trainable tokens")
        labels[start:end] = input_ids[start:end]
    if not any(label != IGNORE_INDEX for label in labels):
        raise DataError("assistant-only mask contains no trainable token")
    return input_ids, labels


def truncate_example(
    input_ids: Sequence[int],
    labels: Sequence[int],
    *,
    max_length: int,
    strategy: str,
) -> tuple[list[int], list[int]] | None:
    """Apply an explicit truncation policy while preserving label alignment."""

    if max_length <= 0:
        raise DataError("max_length must be positive")
    if len(input_ids) != len(labels):
        raise DataError("input_ids and labels must have identical lengths")
    if len(input_ids) <= max_length:
        return list(input_ids), list(labels)
    normalized = strategy.lower().replace("keep_", "")
    if normalized == "drop":
        return None
    if normalized == "error":
        raise DataError(f"example has {len(input_ids)} tokens, above max_length={max_length}")
    if normalized == "left":
        kept_ids = list(input_ids[-max_length:])
        kept_labels = list(labels[-max_length:])
    elif normalized == "right":
        kept_ids = list(input_ids[:max_length])
        kept_labels = list(labels[:max_length])
    else:
        raise DataError("truncation strategy must be one of: drop, error, left/keep_left, right/keep_right")
    if not any(label != IGNORE_INDEX for label in kept_labels):
        raise DataError(
            f"{strategy} truncation removed the entire assistant response; use keep_left or increase max_length"
        )
    return kept_ids, kept_labels


def record_metadata(record: Mapping[str, Any], row_number: int) -> tuple[str, str, str]:
    """Extract stable sample id, qtype and fine-grained dimension labels."""

    meta = record.get("meta") if isinstance(record.get("meta"), Mapping) else {}
    sample_id = str(record.get("sample_id") or meta.get("id") or f"row-{row_number:06d}")
    qtype = str(meta.get("qtype") or record.get("question_type") or "unknown").upper()
    dimension = str(meta.get("dim") or record.get("dimension") or "unknown")
    return sample_id, qtype, dimension


@dataclass(frozen=True)
class DatasetStats:
    total_records: int
    kept_records: int
    dropped_overlength: int
    max_observed_tokens: int
    qtypes: dict[str, int]
    dimensions: dict[str, int]


class ChatJsonlDataset:
    """Small eager dataset suitable for a few-thousand-example SFT run."""

    def __init__(
        self,
        path: str | Path,
        tokenizer: Any,
        *,
        max_length: int,
        truncation: str = "left",
    ) -> None:
        self.path = Path(path)
        self.examples: list[dict[str, Any]] = []
        qtypes: Counter[str] = Counter()
        dimensions: Counter[str] = Counter()
        total = dropped = max_observed = 0
        for row_number, record in enumerate(iter_jsonl(self.path), start=1):
            total += 1
            try:
                messages = validate_messages(record)
                input_ids, labels = assistant_only_labels(tokenizer, messages)
                max_observed = max(max_observed, len(input_ids))
                truncated = truncate_example(
                    input_ids,
                    labels,
                    max_length=max_length,
                    strategy=truncation,
                )
                if truncated is None:
                    dropped += 1
                    continue
                input_ids, labels = truncated
                sample_id, qtype, dimension = record_metadata(record, row_number)
                self.examples.append(
                    {
                        "input_ids": input_ids,
                        "attention_mask": [1] * len(input_ids),
                        "labels": labels,
                        "_sample_id": sample_id,
                        "_qtype": qtype,
                        "_dimension": dimension,
                    }
                )
                qtypes[qtype] += 1
                dimensions[dimension] += 1
            except DataError as exc:
                raise DataError(f"{self.path}:{row_number} ({record.get('sample_id', 'unknown')}): {exc}") from exc
        if not self.examples:
            raise DataError(f"no usable examples in {self.path}")
        self.stats = DatasetStats(
            total_records=total,
            kept_records=len(self.examples),
            dropped_overlength=dropped,
            max_observed_tokens=max_observed,
            qtypes=dict(sorted(qtypes.items())),
            dimensions=dict(sorted(dimensions.items())),
        )
        LOGGER.info("loaded %s: %s", self.path, self.stats)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.examples[index]


class AssistantOnlyDataCollator:
    """Right-pad tokenized examples and retain metadata for metric logging."""

    def __init__(self, tokenizer: Any, *, pad_to_multiple_of: int = 8) -> None:
        if tokenizer.pad_token_id is None:
            raise DataError("tokenizer.pad_token_id must be configured before constructing the collator")
        self.pad_token_id = int(tokenizer.pad_token_id)
        self.pad_to_multiple_of = max(1, int(pad_to_multiple_of))

    def __call__(self, features: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        import torch

        if not features:
            raise DataError("cannot collate an empty batch")
        longest = max(len(feature["input_ids"]) for feature in features)
        padded_length = ((longest + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of) * self.pad_to_multiple_of

        def pad(values: Sequence[int], fill: int) -> list[int]:
            return list(values) + [fill] * (padded_length - len(values))

        return {
            "input_ids": torch.tensor([pad(feature["input_ids"], self.pad_token_id) for feature in features]),
            "attention_mask": torch.tensor([pad(feature["attention_mask"], 0) for feature in features]),
            "labels": torch.tensor([pad(feature["labels"], IGNORE_INDEX) for feature in features]),
            "_sample_ids": [str(feature["_sample_id"]) for feature in features],
            "_qtypes": [str(feature["_qtype"]) for feature in features],
            "_dimensions": [str(feature["_dimension"]) for feature in features],
        }
