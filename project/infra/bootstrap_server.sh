#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

required_free_gib="${SOMBENCH_REQUIRED_FREE_GIB:-180}"
mkdir -p \
  "${SOMBENCH_PROJECT_DIR}" \
  "${SOMBENCH_DATA_DIR}/raw" \
  "${SOMBENCH_DATA_DIR}/processed" \
  "${SOMBENCH_MODEL_DIR}" \
  "${SOMBENCH_OUTPUT_DIR}" \
  "${SOMBENCH_MERGED_DIR}" \
  "${SOMBENCH_RELEASE_DIR}" \
  "${SOMBENCH_LOG_DIR}" \
  "${SOMBENCH_CACHE_DIR}/huggingface" \
  "${SOMBENCH_CACHE_DIR}/modelscope" \
  "${SOMBENCH_CACHE_DIR}/modelscope-home" \
  "${SOMBENCH_CACHE_DIR}/pip" \
  "${SOMBENCH_CACHE_DIR}/torch" \
  "${SOMBENCH_TMP_DIR}" \
  "${SOMBENCH_WHEELHOUSE}" \
  "${SOMBENCH_ROOT}/envs"

free_kib="$(df -Pk "${SOMBENCH_ROOT}" | awk 'NR == 2 {print $4}')"
required_kib="$((required_free_gib * 1024 * 1024))"
if (( free_kib < required_kib )); then
  echo "ERROR: ${SOMBENCH_ROOT} has less than ${required_free_gib} GiB free." >&2
  df -h "${SOMBENCH_ROOT}" >&2
  exit 2
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi is unavailable." >&2
  exit 3
fi

gpu_count="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')"
if [[ "${gpu_count}" != "1" ]]; then
  echo "WARNING: expected one visible GPU, found ${gpu_count}." >&2
fi

echo "Workspace: ${SOMBENCH_ROOT}"
df -h "${SOMBENCH_ROOT}"
nvidia-smi --query-gpu=index,name,memory.total,memory.free,driver_version --format=csv,noheader
echo "Server workspace initialized."
