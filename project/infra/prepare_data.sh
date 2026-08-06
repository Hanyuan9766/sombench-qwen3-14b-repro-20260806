#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

train_file="${SOMBENCH_TRAIN_RAW:-${SOMBENCH_DATA_DIR}/raw/SoMBench_reference_training_without_Q4.jsonl}"
public_file="${SOMBENCH_PUBLIC_RAW:-${SOMBENCH_DATA_DIR}/raw/sombench-public-test-v1.jsonl}"
output_dir="${SOMBENCH_PROCESSED_V1:-${SOMBENCH_DATA_DIR}/processed/v1}"
expected_train_sha="b47a950649f74ac37ad507e12e36045775ad23af743f6a1d92e60fb7b80fef68"
expected_public_sha="c50ffee7cc11bd9a3d2bedb5148c8cc854f4279e063111ecb54545563423bd6e"

for required in "${train_file}" "${public_file}"; do
  if [[ ! -f "${required}" ]]; then
    echo "ERROR: dataset is missing: ${required}" >&2
    exit 2
  fi
done

actual_train_sha="$(sha256sum "${train_file}" | awk '{print $1}')"
actual_public_sha="$(sha256sum "${public_file}" | awk '{print $1}')"
if [[ "${actual_train_sha}" != "${expected_train_sha}" ]]; then
  echo "ERROR: unexpected training data SHA-256: ${actual_train_sha}" >&2
  exit 3
fi
if [[ "${actual_public_sha}" != "${expected_public_sha}" ]]; then
  echo "ERROR: unexpected public data SHA-256: ${actual_public_sha}" >&2
  exit 4
fi

if [[ ! -x "${SOMBENCH_TRAIN_ENV}/bin/python" ]]; then
  echo "ERROR: training environment is missing." >&2
  exit 5
fi

cd "${SOMBENCH_PROJECT_DIR}"
"${SOMBENCH_TRAIN_ENV}/bin/python" -m data_pipeline \
  --train "${train_file}" \
  --public-test "${public_file}" \
  --output-dir "${output_dir}" \
  "$@"

