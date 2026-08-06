#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

repo_id="${SOMBENCH_BASE_REPO_ID:-Qwen/Qwen3-14B}"
revision="${SOMBENCH_BASE_REVISION:-master}"
max_workers="${SOMBENCH_DOWNLOAD_WORKERS:-8}"
if [[ ! "${max_workers}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: SOMBENCH_DOWNLOAD_WORKERS must be a positive integer." >&2
  exit 2
fi
if [[ -x "${SOMBENCH_HUB_ENV}/bin/ms-hub" ]]; then
  download_env="${SOMBENCH_HUB_ENV}"
elif [[ -x "${SOMBENCH_EVAL_ENV}/bin/ms-hub" ]]; then
  download_env="${SOMBENCH_EVAL_ENV}"
elif [[ -x "${SOMBENCH_TRAIN_ENV}/bin/ms-hub" ]]; then
  download_env="${SOMBENCH_TRAIN_ENV}"
else
  echo "ERROR: modelscope-hub is missing; run setup_hub_env.sh first." >&2
  exit 2
fi
download_python="${download_env}/bin/python"

mkdir -p "${SOMBENCH_MODEL_DIR}"
"${download_python}" "${SCRIPT_DIR}/resolve_model_revision.py" \
  --repo-id "${repo_id}" \
  --revision "${revision}" \
  --output "${SOMBENCH_MODEL_DIR}.source_revision.json"
"${download_env}/bin/ms-hub" download "${repo_id}" \
  --repo-type model \
  --revision "${revision}" \
  --max-workers "${max_workers}" \
  --local-dir "${SOMBENCH_MODEL_DIR}"

"${download_python}" "${SCRIPT_DIR}/verify_source_snapshot.py" \
  "${SOMBENCH_MODEL_DIR}.source_revision.json" \
  "${SOMBENCH_MODEL_DIR}" \
  --output "${SOMBENCH_MODEL_DIR}.source_verification.json"

validator_python=""
for candidate in "${SOMBENCH_TRAIN_ENV}" "${SOMBENCH_EVAL_ENV}" "${download_env}"; do
  if [[ -x "${candidate}/bin/python" ]] && \
    "${candidate}/bin/python" -c 'import safetensors, torch, transformers' >/dev/null 2>&1; then
    validator_python="${candidate}/bin/python"
    break
  fi
done
if [[ -z "${validator_python}" ]]; then
  echo "ERROR: model downloaded, but no torch/transformers/safetensors validator environment is ready." >&2
  exit 3
fi

"${validator_python}" "${SCRIPT_DIR}/validate_hf_repo.py" "${SOMBENCH_MODEL_DIR}" \
  --output "${SOMBENCH_MODEL_DIR}.validation.json"
