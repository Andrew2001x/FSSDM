#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

DELAY_MINUTES="${1:-300}"

echo "[INFO] $(date '+%F %T') dealer script armed; waiting ${DELAY_MINUTES} minutes."
sleep "$((DELAY_MINUTES * 60))"
echo "[INFO] $(date '+%F %T') starting dealer benchmark."

exec env \
  UNCLIP_ESTIMATE_REPEAT=1 \
  UNCLIP_DEBUG_PROTOCOL_FP=1 \
  UNCLIP_PROFILE_SCHEMA=1 \
  SHARK_KEYBUF_IO_MB=512 \
  OMP_PROC_BIND=close \
  OMP_PLACES=cores \
  OMP_NUM_THREADS=32 \
  UNCLIP_SEED=20260304 \
  ./build/benchmark-unclip_img2imgsmall 2
