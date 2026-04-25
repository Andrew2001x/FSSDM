#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

DELAY_MINUTES="${1:-360}"
HOST="${2:-127.0.0.1}"
PORT="${3:-42069}"

echo "[INFO] $(date '+%F %T') client script armed; waiting ${DELAY_MINUTES} minutes."
sleep "$((DELAY_MINUTES * 60))"
echo "[INFO] $(date '+%F %T') starting client benchmark."

mkdir -p logs

exec env \
  SHARK_KEYBUF_IO_MB=512 \
  OMP_PROC_BIND=close \
  OMP_PLACES=cores \
  OMP_NUM_THREADS=4 \
  ./build/benchmark-ddpm 1 "${HOST}" "${PORT}" --seed 20260304 --config benchmarks/ddpm_bench_config.json \
  > logs/client_online.log 2>&1
