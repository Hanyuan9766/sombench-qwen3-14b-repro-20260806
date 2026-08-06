# SoMBench data pipeline

This standard-library-only pipeline validates and prepares the reference data
before GPU training.  It performs the following operations:

- extracts stories and question options from the training chats;
- rejects legacy Q2 rows with six options and other structurally unsafe rows;
- replaces verbose/multiple answer boxes with a compact rationale and exactly
  one final `\boxed{...}`;
- assigns normalized and high-threshold fuzzy story cluster IDs;
- creates a dimension-aware train/dev split with no story-cluster leakage;
- deterministically balances all present fine-grained dimensions;
- reconciles public `correct_letters` against exact `correct_answers` to option
  text mappings, excluding conflicts, unmapped answers, and single-choice
  format violations from the conservative safe dev set;
- writes a complete `audit.json` with input hashes, distributions, filters, and
  invariant checks.

Run from the project root:

```bash
python -m data_pipeline \
  --train /path/to/SoMBench_reference_training_without_Q4.jsonl \
  --public-test /path/to/sombench-public-test-v1.jsonl \
  --output-dir artifacts/data/v1
```

The default public policy is `agreement_only`: answer text must map exactly to
one or more options and its mapped letters must agree with `correct_letters`.
Use `--safe-dev-policy mapped_answer_text` only after deciding that exact answer
text is authoritative.  `public_label_conflicts.jsonl` always keeps both label
representations plus a nullable `manual_letters` field for adjudication.

Known outputs are replaced only with `--force`.  Each file is written through a
temporary file followed by an atomic rename.

