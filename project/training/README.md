# Qwen3-14B QLoRA, merge, and evaluation

Run commands from the repository root so `python -m training...` resolves the
package. Keep the original base snapshot immutable.

## Environment

Use Python 3.10 and the server's CUDA-compatible PyTorch, then install:

```bash
pip install -r training/requirements.txt
pip install flash-attn==2.7.4.post1 --no-build-isolation
python -c "import transformers,peft,bitsandbytes; print(transformers.__version__, peft.__version__)"
```

The required versions are Transformers 4.51.3 and PEFT 0.15.2. If Flash
Attention cannot be built, set `model.attn_implementation=sdpa`; do not change
the model or tokenizer revision between training and merge.

## Train and resume

The input is OpenAI-style JSONL with a final assistant message and `meta.qtype`
and `meta.dim`. The tokenizer's own chat template identifies assistant spans;
system/user tokens receive label `-100`. The configured `truncation: error`
ensures an upstream cleaning mistake cannot silently cut the story or answer.

```bash
python -m training.train --config training/configs/qwen3_14b_qlora_stage1.yaml
python -m training.train --config training/configs/qwen3_14b_qlora.yaml
```

Stage 1 sees every accepted training row once. Stage 2 verifies and loads the
stage-1 Adapter, then calibrates it on the 71-dimension-balanced set at half
the learning rate. The second stage refuses an Adapter whose base fingerprint
or LoRA shape differs from the current verified configuration.

Checkpoints are discovered under `output_dir` when
`resume_from_checkpoint: auto`. `trainer_state.json` contains token-weighted
`loss_by_qtype/*`, `loss_by_dimension/*`, and their validation equivalents.
Best-checkpoint selection uses `eval_loss_macro_dimension`, so every one of the
71 hidden-task dimensions has equal influence despite the imbalanced dev rows.
The final artifact at `final_adapter/` is not submit-ready.

## Merge to a complete ModelScope repository

```bash
python -m training.merge --config training/configs/merge_qwen3_14b.yaml
python -m training.repo_check /root/autodl-tmp/sombench/merged_model
```

Merge loads the original base in BF16, applies the adapter, writes sharded
Safetensors plus tokenizer/chat template and ModelScope metadata, then rejects
residual LoRA/4-bit files or an implausibly small repository. Upload only the
merged directory, never `final_adapter/`.

## Reproduce the public-test protocol

Serve the merged directory in the separate official-version environment:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model /root/autodl-tmp/sombench/merged_model \
  --served-model-name sombench-qwen3-14b \
  --dtype bfloat16 --tensor-parallel-size 1 --max-model-len 16384 \
  --host 127.0.0.1 --port 8000
python -m training.evaluate --config training/configs/eval_vllm.yaml
```

The evaluator refuses settings other than `cot / 0.6 / 0.95 / 8192 / n=1`,
uses exactly three seeds, selects the original reference-data single/multi
system prompt from `task_format`, checkpoints every response, resumes failed
runs, and extracts the final balanced `\\boxed{}` only. It writes:

- `predictions.jsonl`: append-only API attempts, including failures;
- `predictions_canonical.jsonl`: exactly one successful row per seed/sample key;
- `sample_stability.jsonl`: every sample's three predictions, majority, tie and
  consistency status;
- `metrics.json`: separately named reported-letter, 136-row safe-agreement and
  answer-text-mapped diagnostics, split by Q1/Q2/Q3 and all available dimensions.

For the corrupted public labels, the recommended model diagnostic is the
136-row agreement-safe subset. The all-213 reported-letter score is retained as
a machine-label reference, and the mapped-answer score is an explicitly
separate alternative; conflicts never enter the safe score. Public raw rows do
not contain dimension metadata, so their report states coverage `0/213` and
does not invent a 71-task assignment. `eval_dev_vllm.yaml` uses the genuine dev
metadata to report all three primary and 71 fine-grained dimensions.

Create a compact, read-only public/dev handoff after both runs finish:

```bash
python -m training.summarize_evaluation \
  --public-dir /root/autodl-tmp/sombench/outputs/eval-qwen3-14b-socialmind-v1 \
  --dev-dir /root/autodl-tmp/sombench/outputs/eval-dev-qwen3-14b-socialmind-v1 \
  --json-out /root/autodl-tmp/sombench/outputs/evaluation-summary.json \
  --markdown-out /root/autodl-tmp/sombench/outputs/evaluation-summary.md
```

The summarizer performs no network/model calls and does not modify either input
directory. It checks the current metrics schema, cross-checks the stability
artifact, and independently validates that `predictions_canonical.jsonl` is a
unique, complete `(seed, sample_id)` grid. Its integrity table reports finish
reasons, length truncations, empty parsed predictions, parse rates, majority
ties, and three-seed consistency before presenting all three label policies,
Q1/Q2/Q3, and each seed.

## Deliberate hard stops

- Training fails without CUDA/BF16 support, a usable chat template, or any
  example over the configured length when `truncation: error` is active.
- Evaluation fails at the end if any API request remains missing, but the next
  invocation resumes only those failed keys.
- Repository validation expects `model_type=qwen3`, BF16 config, fast tokenizer,
  chat template, Safetensors shards, and at least 20 GB of weights.
