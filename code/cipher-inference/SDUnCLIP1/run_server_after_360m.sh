#!/usr/bin/env bash
set -euo pipefail

DELAY_MINUTES="${1:-600}"
HOST="${2:-127.0.0.1}"
PORT="${3:-42069}"

mkdir -p logs

echo "[INFO] $(date '+%F %T') server script armed; waiting ${DELAY_MINUTES} minutes."
sleep "$((DELAY_MINUTES * 60))"

echo "[INFO] $(date '+%F %T') starting server benchmark."
exec taskset -c 0-15 env \
  SHARK_KEYBUF_IO_MB=512 \
  OMP_NUM_THREADS=16 \
  OMP_DYNAMIC=FALSE \
  OMP_PROC_BIND=close \
  OMP_PLACES=cores \
  OMP_WAIT_POLICY=PASSIVE \
  GOMP_CPU_AFFINITY="0-15" \
  UNCLIP_SEED=20260304 \
  UNCLIP_ESTIMATE_REPEAT=1 \
  ./build/benchmark-unclip_img2imgsmall 0 "${HOST}" "${PORT}" > logs/server_online.log 2>&1
