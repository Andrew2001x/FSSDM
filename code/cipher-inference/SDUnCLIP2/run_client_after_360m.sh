#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

DELAY_MINUTES="${1:-360}"
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
exec env \
  UNCLIP_USE_AUTOGEN_SPEC=1 \
  UNCLIP_ESTIMATE_REPEAT=1 \
  UNCLIP_DEBUG_PROTOCOL_FP=1 \
  UNCLIP_PROFILE_SCHEMA=1 \
  SHARK_DEBUG_LIGHT_PROGRESS=1 \
  GOMP_CPU_AFFINITY="4 5 6 7" \
  OMP_PROC_BIND=true \
  OMP_WAIT_POLICY=PASSIVE \
  OMP_NUM_THREADS=4 \
  UNCLIP_SEED=20260304 \
  ./build/benchmark-unclip_img2imgsmall 1 "${HOST}" "${PORT}" > logs/unclip_client.log 2>&1
