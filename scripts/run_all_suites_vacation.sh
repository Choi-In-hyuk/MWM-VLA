#!/bin/bash
# 휴가용 마스터 러너: 4 suite를 각각 따로 학습하고, 각 모델을 자기 도메인에서만 평가.
#
# Phase 1 (suite 순차, 4회): StreamingMamba 학습(predictor 30k -> stage2 30k)
#                            -> 그 suite 원본 LIBERO 50 trials 평가
# Phase 2 (suite 순차, 4회): 각 모델을 자기 suite의 LIBERO-plus 7축 평가
#
# 모델: VLA_DINO_StreamingMamba (2-frame streaming Mamba-2 WM + Qwen-LoRA + flow head)
# 단일 GPU (RTX A6000 48GB). 백본 = 원조 VLA-JEPA LIBERO ckpt.
#
# 검증된 세팅:
#  - 학습/원본LIBERO: vla_jepa env, LIBERO_HOME=/home/choi/LIBERO-PRO
#  - plus 평가       : vla_plus  env, LIBERO_HOME=/home/choi/LIBERO-plus (오늘 7축 평가와 동일)
#
# Usage: bash scripts/run_all_suites_vacation.sh [train|eval_libero|eval_plus|all]
set -uo pipefail
cd "$(dirname "$0")/.."

WHICH=${1:-all}

CKPT=results/vla_jepa_libero_orig/checkpoints/VLA-JEPA-LIBERO.pt
DATA=/home/choi/data/datasets/LIBERO
SUITES=(libero_spatial libero_object libero_goal libero_10)

STAGE1_STEPS=30000
STAGE2_STEPS=30000
WARMUP_STEPS=500
PER_GPU_BS=16
EVAL_TRIALS=50

PY_TRAIN=/home/choi/miniconda3/envs/vla_jepa/bin/python
PY_PLUS=/home/choi/miniconda3/envs/vla_plus/bin/python

# streaming Mamba-2 hyperparameters (MWM libero10 recipe)
STREAM="--stream_state_dim 1024 --stream_depth 12 --stream_d_state 64 \
  --stream_d_conv 1 --stream_headdim 64 --stream_chunk_size 64"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

# ------------------------------------------------------------------ Phase 1a: train
train_suite () {
  local MIX=$1
  local OUT=results/stream_${MIX}
  mkdir -p ${OUT}/pred ${OUT}/stage2
  local common="--framework VLA_DINO_StreamingMamba --dino_backbone dinov2_vitb14 \
    --qwen_lora --lora_r 16 --lora_alpha 32 \
    --backbone_ckpt ${CKPT} --data_root ${DATA} --data_mix ${MIX} \
    --warmup_steps ${WARMUP_STEPS} --log_every 50 --num_workers 8 ${STREAM}"

  # predictor: bs16 OK (48GB). 이미 완주(final ckpt 존재)면 스킵.
  if [[ -f ${OUT}/pred/checkpoints/mamba_wm_final.pt ]]; then
    echo "############ [${MIX}] Stage1 predictor -- SKIP (final ckpt 존재) ############"
  else
    echo "############ [${MIX}] Stage1 predictor ${STAGE1_STEPS} (bs${PER_GPU_BS}) ############"
    PYTHONPATH=$(pwd) ${PY_TRAIN} scripts/train_mamba_wm.py --stage predictor ${common} \
      --max_steps ${STAGE1_STEPS} --save_every 5000 --state_save_every 5000 \
      --batch_size ${PER_GPU_BS} --output_dir ${OUT}/pred > ${OUT}/pred/train.log 2>&1
    local rc=$?
    if [[ $rc -ne 0 ]]; then echo "!!! [${MIX}] predictor FAILED rc=$rc — 중단"; tail -20 ${OUT}/pred/train.log; exit 1; fi
  fi

  # stage2: +flow head라 무거움 -> bs8 x grad_accum2 = effective bs16 (48GB에 맞춤).
  if [[ -f ${OUT}/stage2/checkpoints/mamba_wm_final.pt ]]; then
    echo "############ [${MIX}] Stage2 -- SKIP (final ckpt 존재) ############"
  else
    echo "############ [${MIX}] Stage2 +head ${STAGE2_STEPS} (bs8 x ga2) ############"
    PYTHONPATH=$(pwd) ${PY_TRAIN} scripts/train_mamba_wm.py --stage stage2 ${common} \
      --max_steps ${STAGE2_STEPS} --save_every 5000 --state_save_every 5000 \
      --batch_size 8 --grad_accum 2 \
      --resume_ckpt ${OUT}/pred/checkpoints/mamba_wm_final.pt \
      --output_dir ${OUT}/stage2 > ${OUT}/stage2/train.log 2>&1
    local rc=$?
    if [[ $rc -ne 0 ]]; then echo "!!! [${MIX}] stage2 FAILED rc=$rc — 중단"; tail -20 ${OUT}/stage2/train.log; exit 1; fi
  fi
}

# ------------------------------------------------------------------ Phase 1b: LIBERO eval
eval_libero_suite () {
  local MIX=$1
  local CK=results/stream_${MIX}/stage2/checkpoints/mamba_wm_final.pt
  local PORT=$2
  local OUT=results/eval/${MIX}_stream
  local SLOG=/tmp/vac_libero_${PORT}.log
  mkdir -p "${OUT}"
  export LIBERO_HOME=/home/choi/LIBERO-PRO
  export LIBERO_CONFIG_PATH=${LIBERO_HOME}/libero
  export MUJOCO_GL=egl

  echo "############ [${MIX}] LIBERO eval (${EVAL_TRIALS} trials) ############"
  PYTHONPATH="$(pwd)" ${PY_TRAIN} deployment/model_server/server_policy.py \
      --ckpt_path ${CK} --port ${PORT} --cuda 0 > "${SLOG}" 2>&1 &
  local SPID=$!
  for i in $(seq 1 120); do
    grep -q "server listening" "${SLOG}" 2>/dev/null && { echo "server up."; break; }
    kill -0 ${SPID} 2>/dev/null || { echo "SERVER DIED:"; tail -20 "${SLOG}"; return 1; }
    sleep 2
  done
  PYTHONPATH="${LIBERO_HOME}:$(pwd)" ${PY_TRAIN} examples/LIBERO/eval_libero.py \
      --args.pretrained-path ${CK} --args.host 127.0.0.1 --args.port ${PORT} \
      --args.task-suite-name "${MIX}" --args.num-trials-per-task ${EVAL_TRIALS} \
      --args.video-out-path "${OUT}" --args.with_state "true" \
      --args.action-chunk-size 0 --args.seed 7 2>&1 | tee "${OUT}/eval.log"
  kill ${SPID} 2>/dev/null || true; wait ${SPID} 2>/dev/null || true
}

# ------------------------------------------------------------------ Phase 2: plus eval (자기 suite 7축)
eval_plus_suite () {
  local MIX=$1
  local CK=results/stream_${MIX}/stage2/checkpoints/mamba_wm_final.pt
  local PORT=$2
  local OUT=results/eval_plus/${MIX}_stream
  local SLOG=/tmp/vac_plus_${PORT}.log
  mkdir -p "${OUT}"
  export LIBERO_HOME=/home/choi/LIBERO-plus
  export LIBERO_CONFIG_PATH=${LIBERO_HOME}/libero
  export MUJOCO_GL=egl

  echo "############ [${MIX}] LIBERO-plus 7축 (자기 suite) ############"
  PYTHONPATH="$(pwd)" ${PY_TRAIN} deployment/model_server/server_policy.py \
      --ckpt_path ${CK} --port ${PORT} --cuda 0 > "${SLOG}" 2>&1 &
  local SPID=$!
  for i in $(seq 1 120); do
    grep -q "server listening" "${SLOG}" 2>/dev/null && { echo "server up."; break; }
    kill -0 ${SPID} 2>/dev/null || { echo "SERVER DIED:"; tail -20 "${SLOG}"; return 1; }
    sleep 2
  done
  # eval_libero_plus.py: suite 단위로 전체 돌고 category(7축)로 자동 집계
  PYTHONPATH="${LIBERO_HOME}:$(pwd)" ${PY_PLUS} examples/LIBERO/eval_libero_plus.py \
      --args.pretrained-path ${CK} --args.host 127.0.0.1 --args.port ${PORT} \
      --args.task-suite-name "${MIX}" --args.out-dir "${OUT}" \
      --args.num-trials-per-task 1 --args.with-state "true" \
      --args.action-chunk-size 0 --args.seed 7 2>&1 | tee "${OUT}/eval.log"
  kill ${SPID} 2>/dev/null || true; wait ${SPID} 2>/dev/null || true
}

# ================================================================== run
port=18200
if [[ "$WHICH" == "all" || "$WHICH" == "train" ]]; then
  for MIX in "${SUITES[@]}"; do train_suite "$MIX"; done
fi
if [[ "$WHICH" == "all" || "$WHICH" == "eval_libero" ]]; then
  for MIX in "${SUITES[@]}"; do eval_libero_suite "$MIX" $((port++)); done
fi
if [[ "$WHICH" == "all" || "$WHICH" == "eval_plus" ]]; then
  for MIX in "${SUITES[@]}"; do eval_plus_suite "$MIX" $((port++)); done
fi

echo "=================================================================="
echo "[ALL DONE] 4 suite 학습+평가 완료."
echo "  학습 ckpt:     results/stream_<suite>/stage2/checkpoints/mamba_wm_final.pt"
echo "  원본 LIBERO:   results/eval/<suite>_stream/eval.log"
echo "  LIBERO-plus:   results/eval_plus/<suite>_stream/eval.log (7축 집계)"
echo "=================================================================="
