#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

python_bin="${SOMBENCH_TRAIN_ENV}/bin/python"
if [[ ! -x "${python_bin}" ]]; then
  echo "ERROR: training environment is missing; run setup_train_env.sh first." >&2
  exit 2
fi

export MAX_JOBS="${SOMBENCH_FLASH_ATTN_BUILD_JOBS:-8}"
export FLASH_ATTENTION_FORCE_BUILD=TRUE
export FLASH_ATTN_CUDA_ARCHS="${SOMBENCH_FLASH_ATTN_ARCHS:-80}"
export PATH="${SOMBENCH_TRAIN_ENV}/bin:${PATH}"
"${python_bin}" -m pip install ninja==1.11.1.4 psutil==7.0.0 packaging==24.2
"${python_bin}" -m pip install flash-attn==2.7.4.post1 --no-build-isolation
"${python_bin}" -c 'import flash_attn; print("flash_attn", flash_attn.__version__)'
