"""A Trainer variant that emits teacher-forced loss by qtype/dimension."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence


def masked_cross_entropy_by_example(logits: Any, labels: Any, ignore_index: int = -100) -> tuple[Any, Any]:
    """Return summed causal-LM NLL and token counts for every batch row."""

    import torch.nn.functional as F

    if logits.ndim != 3 or labels.ndim != 2 or logits.shape[:2] != labels.shape:
        raise ValueError("expected logits [batch, seq, vocab] and aligned labels [batch, seq]")
    shifted_logits = logits[:, :-1, :]
    shifted_labels = labels[:, 1:]
    mask = shifted_labels.ne(ignore_index)
    safe_labels = shifted_labels.masked_fill(~mask, 0)
    # Calculating row by row avoids materializing a [batch, seq] fp32 loss plus
    # another large flattened copy.  Typical QLoRA micro-batches are size one.
    sums = []
    counts = []
    for row_logits, row_labels, row_mask in zip(shifted_logits, safe_labels, mask):
        token_losses = F.cross_entropy(row_logits, row_labels, reduction="none")
        sums.append((token_losses * row_mask).sum())
        counts.append(row_mask.sum())
    import torch

    return torch.stack(sums), torch.stack(counts)


def metric_key(value: str) -> str:
    """Map arbitrary metadata to a stable, tracker-friendly metric suffix."""

    cleaned = re.sub(r"[^0-9A-Za-z_.-]+", "_", value.strip())
    return cleaned.strip("_") or "unknown"


class GroupLossAccumulator:
    """Accumulate token-weighted negative log likelihood by string group."""

    def __init__(self, qtypes: Iterable[str], dimensions: Iterable[str]) -> None:
        self.vocab = {
            "qtype": sorted(set(qtypes) | {"unknown"}),
            "dimension": sorted(set(dimensions) | {"unknown"}),
        }
        self._state: dict[str, dict[str, list[float]]] = {
            "train": defaultdict(lambda: [0.0, 0.0]),
            "eval": defaultdict(lambda: [0.0, 0.0]),
        }

    def add(
        self,
        phase: str,
        qtypes: Sequence[str],
        dimensions: Sequence[str],
        loss_sums: Sequence[float],
        token_counts: Sequence[float],
    ) -> None:
        if phase not in self._state:
            raise ValueError(f"unknown phase: {phase}")
        if not (len(qtypes) == len(dimensions) == len(loss_sums) == len(token_counts)):
            raise ValueError("group metadata and per-example losses must have equal lengths")
        state = self._state[phase]
        for qtype, dimension, loss_sum, token_count in zip(qtypes, dimensions, loss_sums, token_counts):
            for kind, value in (("qtype", qtype), ("dimension", dimension)):
                group = value if value in self.vocab[kind] else "unknown"
                bucket = state[f"{kind}\0{group}"]
                bucket[0] += float(loss_sum)
                bucket[1] += float(token_count)

    def flush(self, phase: str, *, device: Any | None = None) -> dict[str, float]:
        """Return averages and reset *phase*, summing workers when distributed."""

        import torch
        import torch.distributed as dist

        state = self._state[phase]
        output: dict[str, float] = {}
        prefix = "eval_loss" if phase == "eval" else "loss"
        for kind in ("qtype", "dimension"):
            group_averages: list[float] = []
            for group in self.vocab[kind]:
                values = state.get(f"{kind}\0{group}", [0.0, 0.0])
                tensor = torch.tensor(values, dtype=torch.float64, device=device or "cpu")
                if dist.is_available() and dist.is_initialized():
                    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
                total, count = tensor.tolist()
                if count > 0:
                    average = total / count
                    output[f"{prefix}_by_{kind}/{metric_key(group)}"] = average
                    group_averages.append(average)
            if group_averages:
                # The hidden A-set contains the same number of questions for
                # every one of the 71 fine dimensions.  A macro loss prevents
                # the naturally imbalanced dev rows from choosing a checkpoint
                # that over-optimizes high-frequency dimensions.
                output[f"{prefix}_macro_{kind}"] = sum(group_averages) / len(group_averages)
        self._state[phase] = defaultdict(lambda: [0.0, 0.0])
        return output


def build_group_vocab(*datasets: Any) -> tuple[list[str], list[str]]:
    qtypes: set[str] = set()
    dimensions: set[str] = set()
    for dataset in datasets:
        if dataset is None:
            continue
        qtypes.update(getattr(dataset, "stats").qtypes)
        dimensions.update(getattr(dataset, "stats").dimensions)
    return sorted(qtypes), sorted(dimensions)


def make_dimension_trainer_class() -> type:
    """Import transformers lazily and construct the concrete Trainer class."""

    from transformers import Trainer

    class DimensionLoggingTrainer(Trainer):
        def __init__(self, *args: Any, group_qtypes: Sequence[str], group_dimensions: Sequence[str], **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.group_losses = GroupLossAccumulator(group_qtypes, group_dimensions)

        def compute_loss(
            self,
            model: Any,
            inputs: dict[str, Any],
            return_outputs: bool = False,
            num_items_in_batch: Any | None = None,
        ) -> Any:
            qtypes = inputs.pop("_qtypes", [])
            dimensions = inputs.pop("_dimensions", [])
            inputs.pop("_sample_ids", None)
            outputs = model(**inputs)
            loss = outputs["loss"] if isinstance(outputs, Mapping) else outputs.loss
            logits = outputs["logits"] if isinstance(outputs, Mapping) else outputs.logits
            if qtypes and dimensions:
                with __import__("torch").no_grad():
                    sums, counts = masked_cross_entropy_by_example(logits.detach(), inputs["labels"])
                self.group_losses.add(
                    "train" if model.training else "eval",
                    qtypes,
                    dimensions,
                    sums.detach().cpu().tolist(),
                    counts.detach().cpu().tolist(),
                )
            return (loss, outputs) if return_outputs else loss

        def log(self, logs: dict[str, float], *args: Any, **kwargs: Any) -> None:
            device = getattr(self.args, "device", None)
            if "loss" in logs or "train_loss" in logs:
                logs.update(self.group_losses.flush("train", device=device))
            if "eval_loss" in logs:
                # Mutate the metrics object passed by Trainer.evaluate so the
                # macro metric is also available to best-checkpoint selection,
                # not only to TensorBoard/log output.
                logs.update(self.group_losses.flush("eval", device=device))
            super().log(logs, *args, **kwargs)

    return DimensionLoggingTrainer
