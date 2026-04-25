#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

DELAY_MINUTES="${1:-360}"
HOST="${2:-127.0.0.1}"
PORT="${3:-42069}"

mkdir -p logs

echo "[INFO] $(date '+%F %T') server script armed; waiting ${DELAY_MINUTES} minutes."
sleep "$((DELAY_MINUTES * 60))"

echo "[INFO] $(date '+%F %T') starting dealer benchmark."
env \
  UNCLIP_USE_AUTOGEN_SPEC=1 \
  UNCLIP_ESTIMATE_REPEAT=1 \
  UNCLIP_DEBUG_PROTOCOL_FP=1 \
  UNCLIP_PROFILE_SCHEMA=1 \
  SHARK_DEBUG_LIGHT_PROGRESS=1 \
  OMP_PROC_BIND=true \
  OMP_WAIT_POLICY=PASSIVE \
  OMP_NUM_THREADS=4 \
  UNCLIP_SEED=20260304 \
  ./build/benchmark-unclip_img2imgsmall 2 > logs/unclip_dealer.log 2>&1

echo "[INFO] $(date '+%F %T') starting server benchmark."
exec env \
  UNCLIP_USE_AUTOGEN_SPEC=1 \
  UNCLIP_ESTIMATE_REPEAT=1 \
  UNCLIP_DEBUG_PROTOCOL_FP=1 \
  UNCLIP_PROFILE_SCHEMA=1 \
  SHARK_DEBUG_LIGHT_PROGRESS=1 \
  GOMP_CPU_AFFINITY="0 1 2 3" \
  OMP_PROC_BIND=true \
  OMP_WAIT_POLICY=PASSIVE \
  OMP_NUM_THREADS=4 \
  UNCLIP_SEED=20260304 \
  ./build/benchmark-unclip_img2imgsmall 0 "${HOST}" "${PORT}" > logs/unclip_server.log 2>&1
