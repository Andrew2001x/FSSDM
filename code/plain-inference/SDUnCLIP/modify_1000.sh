#!/usr/bin/env bash
set -euo pipefail

ts() { date +"%Y-%m-%d %H:%M:%S"; }

# =========================
# English note: localized comment removed.
MODEL="sd2-community/stable-diffusion-2-1-unclip-small"
SEED=1234

COCO_LIMIT=1000
FLICKR_LIMIT=1000
CC3M_LIMIT=1000

# English note: localized comment removed.
FLICKR_K=3
CC3M_K=3

STEPS=40
GUIDANCE=7.0

# English note: localized comment removed.
VARIATION=0.35
NOISE_SCALE=50
NOISE_LEVEL=-1

# English note: localized comment removed.
CHUNK_SIZE=10

# English note: localized comment removed.
AGG="best_tradeoff"
TRADEOFF_ALPHA=1.0

# =========================
# English note: localized comment removed.
# =========================
BATCH_INPUTS=1
PREFETCH=32

SAVE_FORMAT="jpg"     # "png" / "jpg"
PNG_COMPRESS=1
JPG_QUALITY=95

INCEPTION_BATCH=64

ENABLE_TF32=1
ENABLE_CHANNELS_LAST=1
ENABLE_COMPILE_UNET=0

# English note: localized comment removed.
BETWEEN_DATASET_CLEANUP=1
BETWEEN_CHUNK_CLEANUP=1

# =========================
# English note: localized comment removed.
# =========================
OUTROOT="./output00modify2-layer-all-1000??fp32-safe-chunk10"
LOGDIR="${OUTROOT}/logs"
mkdir -p "${OUTROOT}" "${LOGDIR}"

TIMEFILE="${OUTROOT}/time.txt"
START_EPOCH=$(date +%s)
START_HUMAN=$(ts)
echo "start_time: ${START_HUMAN}" > "${TIMEFILE}"
echo "start_epoch: ${START_EPOCH}" >> "${TIMEFILE}"

on_exit () {
  rc=$?
  END_EPOCH=$(date +%s)
  END_HUMAN=$(ts)
  ELAPSED=$((END_EPOCH-START_EPOCH))
  HH=$((ELAPSED/3600))
  MM=$(((ELAPSED%3600)/60))
  SS=$((ELAPSED%60))
  printf "end_time: %s\n" "${END_HUMAN}" >> "${TIMEFILE}"
  printf "end_epoch: %s\n" "${END_EPOCH}" >> "${TIMEFILE}"
  printf "elapsed_seconds: %s\n" "${ELAPSED}" >> "${TIMEFILE}"
  printf "elapsed_hms: %02d:%02d:%02d\n" "${HH}" "${MM}" "${SS}" >> "${TIMEFILE}"
  printf "exit_code: %s\n" "${rc}" >> "${TIMEFILE}"
}
trap on_exit EXIT

COCO_DIR="${OUTROOT}/coco_unclip_variation_v2"
FLICKR_DIR="${OUTROOT}/flickr_unclip_variation_v2"
CC3M_DIR="${OUTROOT}/cc3m_unclip_variation_v2"

EVAL_COCO_OUT="${LOGDIR}/eval_coco_v2.txt"
EVAL_FLICKR_OUT="${LOGDIR}/eval_flickr_v2.txt"
EVAL_CC3M_OUT="${LOGDIR}/eval_cc3m_v2.txt"

GEN_ALL_OUT="${LOGDIR}/gen_all_oncepipe.txt"

run_cmd () {
  local name="$1"; shift
  echo "[$(ts)] ===== RUN: ${name} ====="
  echo "[$(ts)] CMD: $*"
  "$@"
  echo "[$(ts)] ===== DONE: ${name} ====="
  echo
}

# English note: localized comment removed.
EXTRA_FLAGS=""
if [[ "${ENABLE_TF32}" == "1" ]]; then EXTRA_FLAGS+=" --tf32"; fi
if [[ "${ENABLE_CHANNELS_LAST}" == "1" ]]; then EXTRA_FLAGS+=" --channels_last"; fi
if [[ "${ENABLE_COMPILE_UNET}" == "1" ]]; then EXTRA_FLAGS+=" --compile_unet"; fi

EXTRA_FLAGS+=" --disable_progress"
EXTRA_FLAGS+=" --batch_inputs ${BATCH_INPUTS} --prefetch ${PREFETCH}"
EXTRA_FLAGS+=" --save_format ${SAVE_FORMAT} --png_compress_level ${PNG_COMPRESS} --jpg_quality ${JPG_QUALITY}"
EXTRA_FLAGS+=" --disable_safety_checker"
EXTRA_FLAGS+=" --chunk_size ${CHUNK_SIZE}"

if [[ "${BETWEEN_DATASET_CLEANUP}" == "1" ]]; then
  EXTRA_FLAGS+=" --between_dataset_cleanup"
fi
if [[ "${BETWEEN_CHUNK_CLEANUP}" == "1" ]]; then
  EXTRA_FLAGS+=" --between_chunk_cleanup"
fi

echo "[$(ts)] START all jobs (single pipeline + chunked generation)"
echo "[$(ts)] All outputs under: ${OUTROOT}"
echo "[$(ts)] Logs under: ${LOGDIR}"
echo "[$(ts)] Perf: batch_inputs=${BATCH_INPUTS}, K=${CC3M_K}/${FLICKR_K}/${COCO_K}, prefetch=${PREFETCH}, chunk_size=${CHUNK_SIZE}, inception_batch=${INCEPTION_BATCH}, save_format=${SAVE_FORMAT}"

# ============================================================
# English note: localized comment removed.
# ============================================================
run_cmd "ALL generate (single pipe, chunked)" bash -lc "
python run_all_unclip_once_fp32_no_xformers_no_safety_1000.py \
  --model ${MODEL} \
  --dtype fp32 \
  --seed ${SEED} \
  --steps ${STEPS} \
  --guidance_scale ${GUIDANCE} \
  --variation_strength ${VARIATION} \
  --noise_scale ${NOISE_SCALE} \
  --noise_level ${NOISE_LEVEL} \
  --cc3m_root /home/guoyu/zw/stable/data/cc3m \
  --flickr_root /home/guoyu/zw/stable/data/flickr30k \
  --coco_root /home/guoyu/zw/stable/data/coco2017 \
  --cc3m_outdir ${CC3M_DIR} \
  --flickr_outdir ${FLICKR_DIR} \
  --coco_outdir ${COCO_DIR} \
  --cc3m_limit ${CC3M_LIMIT} \
  --flickr_limit ${FLICKR_LIMIT} \
  --coco_limit ${COCO_LIMIT} \
  --cc3m_k ${CC3M_K} \
  --flickr_k ${FLICKR_K} \
  --coco_k ${COCO_K} \
  --order cc3m,flickr,coco \
  ${EXTRA_FLAGS}
" |& tee "${GEN_ALL_OUT}"

# ============================================================
# (B) Eval
# ============================================================
run_cmd "CC3M eval" bash -lc "
python eval_cc3m_unclip_fast.py \
  --run_dir ${CC3M_DIR} \
  --limit_records ${CC3M_LIMIT} \
  --metrics clip_it,clip_ii,div_clip,fid \
  --aggregate ${AGG} \
  --tradeoff_alpha ${TRADEOFF_ALPHA} \
  --device cuda \
  --inception_batch ${INCEPTION_BATCH}
" |& tee "${EVAL_CC3M_OUT}"

run_cmd "Flickr30k eval" bash -lc "
python eval_flickr30k_unclip_fast.py \
  --run_dir ${FLICKR_DIR} \
  --limit_records ${FLICKR_LIMIT} \
  --metrics clip_it,clip_ii,div_clip,fid \
  --aggregate ${AGG} \
  --tradeoff_alpha ${TRADEOFF_ALPHA} \
  --device cuda \
  --inception_batch ${INCEPTION_BATCH}
" |& tee "${EVAL_FLICKR_OUT}"

run_cmd "COCO eval" bash -lc "
python eval_coco_unclip_fast.py \
  --run_dir ${COCO_DIR} \
  --limit_records ${COCO_LIMIT} \
  --metrics clip_it,clip_ii,div_clip,fid \
  --aggregate ${AGG} \
  --tradeoff_alpha ${TRADEOFF_ALPHA} \
  --device cuda \
  --inception_batch ${INCEPTION_BATCH}
" |& tee "${EVAL_COCO_OUT}"

echo "[$(ts)] ALL DONE"
echo "[$(ts)] Eval results saved to:"
echo "  - ${EVAL_CC3M_OUT}"
echo "  - ${EVAL_FLICKR_OUT}"
echo "  - ${EVAL_COCO_OUT}"
echo "[$(ts)] Total time saved to: ${TIMEFILE}"

