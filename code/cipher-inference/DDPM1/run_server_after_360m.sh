#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

DELAY_MINUTES="${1:-360}"
HOST="${2:-127.0.0.1}"
PORT="${3:-42069}"
SEED="${4:-20260304}"

mkdir -p logs

echo "[INFO] $(date '+%F %T') server ddpm script armed; waiting ${DELAY_MINUTES} minutes."
sleep "$((DELAY_MINUTES * 60))"
echo "[INFO] $(date '+%F %T') starting server benchmark-ddpm."

exec env \
  SHARK_KEYBUF_IO_MB=512 \
  OMP_PROC_BIND=close \
  OMP_PLACES=cores \
  OMP_NUM_THREADS=4 \
  ./build/benchmark-ddpm 0 "${HOST}" "${PORT}" --seed "${SEED}" \
  > "logs/server_online.log" 2>&1
