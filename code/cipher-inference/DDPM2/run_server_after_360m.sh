#!/usr/bin/env bash
set -euo pipefail

echo "[INFO] $(date '+%F %T') server script armed; waiting 420 minutes."
sleep $((7 * 60 * 60))
echo "[INFO] $(date '+%F %T') starting server benchmark."

mkdir -p logs

exec taskset -c 0-15 env \
  SHARK_KEYBUF_IO_MB=512 \
  OMP_NUM_THREADS=16 \
  OMP_DYNAMIC=FALSE \
  OMP_PROC_BIND=close \
  OMP_PLACES=cores \
  GOMP_SPINCOUNT=0 \
  OMP_WAIT_POLICY=PASSIVE \
  GOMP_CPU_AFFINITY="0-15" \
  ./build/benchmark-ddpm 0 127.0.0.1 42069 --seed 20260304 --steps 5 --config benchmarks/ddpm_bench_config.json \
  > logs/server_online4.log 2>&1
