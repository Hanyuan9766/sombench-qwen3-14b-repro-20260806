#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

python_bin="${SOMBENCH_SYSTEM_PYTHON:-python3}"
index_url="${SOMBENCH_PYPI_INDEX_URL:-https://mirrors.aliyun.com/pypi/simple}"

if [[ ! -x "${SOMBENCH_HUB_ENV}/bin/python" ]]; then
  "${python_bin}" -m venv "${SOMBENCH_HUB_ENV}"
fi

"${SOMBENCH_HUB_ENV}/bin/python" -m pip install --upgrade pip
"${SOMBENCH_HUB_ENV}/bin/python" -m pip install \
  --index-url "${index_url}" \
  modelscope-hub==0.2.0
"${SOMBENCH_HUB_ENV}/bin/python" -m pip check
mkdir -p "${SOMBENCH_LOG_DIR}"
"${SOMBENCH_HUB_ENV}/bin/python" -m pip freeze --all | LC_ALL=C sort >"${SOMBENCH_LOG_DIR}/hub-environment.freeze.txt"

"${SOMBENCH_HUB_ENV}/bin/python" -c \
  'from modelscope_hub import HubApi; print("modelscope-hub import ok", HubApi)'
