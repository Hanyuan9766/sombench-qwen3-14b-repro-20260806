#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

CONDA_SH="${SOMBENCH_CONDA_SH:-/root/miniconda3/etc/profile.d/conda.sh}"
if [[ ! -f "${CONDA_SH}" ]]; then
  echo "ERROR: conda initialization script not found: ${CONDA_SH}" >&2
  exit 2
fi
# shellcheck disable=SC1090
source "${CONDA_SH}"

if [[ ! -x "${SOMBENCH_EVAL_ENV}/bin/python" ]]; then
  conda create --prefix "${SOMBENCH_EVAL_ENV}" --yes python=3.10 pip
fi

python_bin="${SOMBENCH_EVAL_ENV}/bin/python"
"${python_bin}" -m pip install --upgrade pip==25.1.1 setuptools==80.3.1 wheel==0.45.1
bash "${SCRIPT_DIR}/install_torch_stack.sh" "${python_bin}"
"${python_bin}" -m pip install -r "${SCRIPT_DIR}/requirements-eval.txt"
"${python_bin}" -m pip check
mkdir -p "${SOMBENCH_LOG_DIR}"
"${python_bin}" -m pip freeze --all | LC_ALL=C sort >"${SOMBENCH_LOG_DIR}/eval-environment.freeze.txt"

"${python_bin}" "${SCRIPT_DIR}/verify_environment.py" --mode eval
