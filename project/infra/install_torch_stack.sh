#!/usr/bin/env bash
set -euo pipefail

if (( $# != 1 )); then
  echo "Usage: $0 /absolute/path/to/python" >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

python_bin="$1"
if [[ ! -x "${python_bin}" ]]; then
  echo "ERROR: Python executable not found: ${python_bin}" >&2
  exit 2
fi

# This exact Linux/Python 3.10 artifact was checked against the official PyPI
# SHA-256.  Never pick an arbitrary glob result: a stale CPU, CUDA, Python, or
# platform wheel in the shared wheelhouse must fail visibly rather than change
# the runtime selected for the official environment.
expected_torch_name="torch-2.6.0-cp310-cp310-manylinux1_x86_64.whl"
shopt -s nullglob
torch_candidates=("${SOMBENCH_WHEELHOUSE}"/torch-2.6.0-*.whl)
shopt -u nullglob
if (( ${#torch_candidates[@]} > 1 )); then
  echo "ERROR: multiple torch 2.6.0 wheels found in ${SOMBENCH_WHEELHOUSE}:" >&2
  printf '  %s\n' "${torch_candidates[@]}" >&2
  exit 3
fi

torch_wheel=""
if (( ${#torch_candidates[@]} == 1 )); then
  candidate_name="$(basename -- "${torch_candidates[0]}")"
  if [[ "${candidate_name}" != "${expected_torch_name}" ]]; then
    echo "ERROR: incompatible local torch wheel for the Python 3.10 Linux environment: ${candidate_name}" >&2
    echo "Expected exactly: ${expected_torch_name}" >&2
    exit 4
  fi
  torch_wheel="${torch_candidates[0]}"
fi

if [[ -n "${torch_wheel}" ]]; then
  cuda_requirements=(
    "nvidia-cublas-cu12==12.4.5.8"
    "nvidia-cuda-cupti-cu12==12.4.127"
    "nvidia-cuda-nvrtc-cu12==12.4.127"
    "nvidia-cuda-runtime-cu12==12.4.127"
    "nvidia-cudnn-cu12==9.1.0.70"
    "nvidia-cufft-cu12==11.2.1.3"
    "nvidia-curand-cu12==10.3.5.147"
    "nvidia-cusolver-cu12==11.6.1.9"
    "nvidia-cusparse-cu12==12.3.1.170"
    "nvidia-nccl-cu12==2.21.5"
    "nvidia-nvjitlink-cu12==12.4.127"
    "nvidia-nvtx-cu12==12.4.127"
  )
  "${python_bin}" -m pip install \
    --no-index \
    --no-deps \
    --find-links "${SOMBENCH_WHEELHOUSE}" \
    "${cuda_requirements[@]}"
  "${python_bin}" -m pip install \
    --find-links "${SOMBENCH_WHEELHOUSE}" \
    "${torch_wheel}"
else
  "${python_bin}" -m pip install \
    --index-url https://download.pytorch.org/whl/cu124 \
    torch==2.6.0
fi
