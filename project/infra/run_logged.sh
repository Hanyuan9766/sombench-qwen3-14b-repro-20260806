#!/usr/bin/env bash
set -euo pipefail

if (( $# < 2 )); then
  echo "Usage: $0 LOG_NAME COMMAND [ARG ...]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

log_name="$1"
shift
if [[ ! "${log_name}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "ERROR: LOG_NAME contains unsafe characters." >&2
  exit 3
fi

mkdir -p "${SOMBENCH_LOG_DIR}"
log_path="${SOMBENCH_LOG_DIR}/${log_name}.log"
pid_path="${SOMBENCH_LOG_DIR}/${log_name}.pid"

if [[ -f "${pid_path}" ]]; then
  old_pid="$(<"${pid_path}")"
  if [[ "${old_pid}" =~ ^[0-9]+$ ]] && kill -0 "${old_pid}" 2>/dev/null; then
    echo "ERROR: ${log_name} is already running as PID ${old_pid}." >&2
    exit 4
  fi
fi

nohup "$@" >"${log_path}" 2>&1 &
pid="$!"
printf '%s\n' "${pid}" >"${pid_path}"
echo "Started PID ${pid}; log=${log_path}; pid_file=${pid_path}"

