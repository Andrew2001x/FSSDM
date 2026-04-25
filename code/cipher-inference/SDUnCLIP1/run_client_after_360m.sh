#!/usr/bin/env bash
set -euo pipefail

DELAY_MINUTES="${1:-600}"
HOST="${2:-127.0.0.1}"
PORT="${3:-42069}"

mkdir -p logs

echo "[INFO] $(date '+%F %T') client script armed; waiting ${DELAY_MINUTES} minutes."
sleep "$((DELAY_MINUTES * 60))"

while [ ! -s client.dat ]; do
  echo "[INFO] $(date '+%F %T') waiting for dealer output client.dat."
  sleep 5
done

echo "[INFO] $(date '+%F %T') starting client benchmark."
exec taskset -c 16-31 env \
  SHARK_KEYBUF_IO_MB=512 \
  OMP_NUM_THREADS=16 \
  OMP_DYNAMIC=FALSE \
  OMP_PROC_BIND=close \
  OMP_PLACES=cores \
  GOMP_SPINCOUNT=0 \
  OMP_WAIT_POLICY=PASSIVE \
  GOMP_CPU_AFFINITY="16-31" \
  UNCLIP_SEED=20260304 \
  UNCLIP_ESTIMATE_REPEAT=1 \
  ./build/benchmark-unclip_img2imgsmall 1 "${HOST}" "${PORT}" > logs/client_online.log 2>&1
