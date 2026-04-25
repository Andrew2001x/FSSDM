#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

DELAY_SECONDS=$((4 * 60 * 60))

mkdir -p logs

echo "[INFO] $(date '+%F %T') server script armed; waiting 4 hours."
sleep "${DELAY_SECONDS}"
echo "[INFO] $(date '+%F %T') starting server benchmark."

exec taskset -c 0-15 env \
  SHARK_KEYBUF_IO_MB=512 \
  OMP_NUM_THREADS=16 \
  OMP_DYNAMIC=FALSE \
  OMP_PROC_BIND=close \
  OMP_PLACES=cores \
  OMP_WAIT_POLICY=ACTIVE \
  GOMP_CPU_AFFINITY="0-15" \
  UNCLIP_SEED=20260304 \
  UNCLIP_ESTIMATE_REPEAT=1 \
  ./build/benchmark-unclip_img2imgsmall 0 127.0.0.1 42069 \
  > logs/server_online.log 2>&1
