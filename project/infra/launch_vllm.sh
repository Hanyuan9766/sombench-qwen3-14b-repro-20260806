#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

model_dir="${1:-${SOMBENCH_RELEASE_DIR}}"
port="${SOMBENCH_VLLM_PORT:-8000}"
served_name="${SOMBENCH_SERVED_MODEL_NAME:-sombench-qwen3-14b}"

if [[ ! -x "${SOMBENCH_EVAL_ENV}/bin/vllm" ]]; then
  echo "ERROR: vLLM environment is missing; run setup_eval_env.sh first." >&2
  exit 2
fi

"${SOMBENCH_EVAL_ENV}/bin/python" "${SCRIPT_DIR}/validate_hf_repo.py" "${model_dir}"

exec "${SOMBENCH_EVAL_ENV}/bin/vllm" serve "${model_dir}" \
  --served-model-name "${served_name}" \
  --host 127.0.0.1 \
  --port "${port}" \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.90 \
  --generation-config vllm \
  --disable-log-requests

