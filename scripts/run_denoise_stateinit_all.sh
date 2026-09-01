#!/bin/bash
# denoise + libero_all(4 suite 섞어서 한 번에 학습) 러너.
#
# 우리 denoise(시점 고려: 입력 프레임에 가짜 카메라각/시각노이즈 증강, L_pred target 은
# CLEAN 정면 미래 latent) 를 libero_all mix (object+goal+spatial+10 합침) 하나로
# 학습 -> 통합 모델 1개 -> 4개 suite 각각에서 평가.
#
# 단일 GPU (RTX A6000 48GB).
#   - 학습/원본LIBERO: vla_jepa env, LIBERO_HOME=/home/choi/LIBERO-PRO
#   - plus 평가       : vla_plus  env, LIBERO_HOME=/home/choi/LIBERO-plus
#
# Usage: bash scripts/run_denoise_libero_all.sh [train|eval_libero|eval_plus|all]
set -uo pipefail
cd "$(dirname "$0")/.."

WHICH=${1:-all}

CKPT=results/vla_jepa_libero_orig/checkpoints/VLA-JEPA-LIBERO.pt
DATA=/home/choi/data/datasets/LIBERO
MIX=libero_all                       # 4 suite 섞은 통합 mix (mixtures.py 에 정의됨)
EVAL_SUITES=(libero_spatial libero_object libero_goal libero_10)  # 통합 모델을 각 도메인서 평가

FRAMEWORK=VLA_DINO_StreamingMamba_FutureOnly_Denoise_StateInit
RUNTAG=denoise_stateinit
OUT=results/${RUNTAG}

STAGE1_STEPS=30000
STAGE2_STEPS=30000
WARMUP_STEPS=500
PER_GPU_BS=16
EVAL_TRIALS=50

PY_TRAIN=/home/choi/miniconda3/envs/vla_jepa/bin/python
PY_PLUS=/home/choi/miniconda3/envs/vla_plus/bin/python

STREAM="--stream_state_dim 1024 --stream_depth 12 --stream_d_state 64 \
  --stream_d_conv 1 --stream_headdim 64 --stream_chunk_size 64"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

# ------------------------------------------------------------------ Phase 1: train (libero_all)
train_all () {
  mkdir -p ${OUT}/pred ${OUT}/stage2
  local common="--framework ${FRAMEWORK} --dino_backbone dinov2_vitb14 \
    --qwen_lora --lora_r 16 --lora_alpha 32 \
    --backbone_ckpt ${CKPT} --data_root ${DATA} --data_mix ${MIX} \
    --warmup_steps ${WARMUP_STEPS} --log_every 50 --num_workers 8 ${STREAM}"

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

# ------------------------------------------------------------------ Phase 2: LIBERO eval (통합 모델을 각 suite서)
eval_libero_suite () {
  local SUITE=$1
  local CK=${OUT}/stage2/checkpoints/mamba_wm_final.pt
  local PORT=$2
  local EOUT=results/eval/${SUITE}_${RUNTAG}
  local SLOG=/tmp/${RUNTAG}_libero_${PORT}.log
  mkdir -p "${EOUT}"
  export LIBERO_HOME=/home/choi/LIBERO-PRO
  export LIBERO_CONFIG_PATH=${LIBERO_HOME}/libero
  export MUJOCO_GL=egl

  echo "############ [${SUITE}] LIBERO eval (${EVAL_TRIALS} trials, 통합모델) ############"
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
      --args.task-suite-name "${SUITE}" --args.num-trials-per-task ${EVAL_TRIALS} \
      --args.video-out-path "${EOUT}" --args.with_state "true" \
      --args.action-chunk-size 0 --args.seed 7 2>&1 | tee "${EOUT}/eval.log"
  kill ${SPID} 2>/dev/null || true; wait ${SPID} 2>/dev/null || true
}

# ------------------------------------------------------------------ Phase 3: plus eval (통합 모델을 각 suite 7축)
eval_plus_suite () {
  local SUITE=$1
  local CK=${OUT}/stage2/checkpoints/mamba_wm_final.pt
  local PORT=$2
  local EOUT=results/eval_plus/${SUITE}_${RUNTAG}
  local SLOG=/tmp/${RUNTAG}_plus_${PORT}.log
  mkdir -p "${EOUT}"
  export LIBERO_HOME=/home/choi/LIBERO-plus
  export LIBERO_CONFIG_PATH=${LIBERO_HOME}/libero
  export MUJOCO_GL=egl

  echo "############ [${SUITE}] LIBERO-plus 7축 (통합모델) ############"
  PYTHONPATH="$(pwd)" ${PY_TRAIN} deployment/model_server/server_policy.py \
      --ckpt_path ${CK} --port ${PORT} --cuda 0 > "${SLOG}" 2>&1 &
  local SPID=$!
  for i in $(seq 1 120); do
    grep -q "server listening" "${SLOG}" 2>/dev/null && { echo "server up."; break; }
    kill -0 ${SPID} 2>/dev/null || { echo "SERVER DIED:"; tail -20 "${SLOG}"; return 1; }
    sleep 2
  done
  PYTHONPATH="${LIBERO_HOME}:$(pwd)" ${PY_PLUS} examples/LIBERO/eval_libero_plus.py \
      --args.pretrained-path ${CK} --args.host 127.0.0.1 --args.port ${PORT} \
      --args.task-suite-name "${SUITE}" --args.out-dir "${EOUT}" \
      --args.num-trials-per-task 1 --args.with-state "true" \
      --args.action-chunk-size 0 --args.seed 7 2>&1 | tee "${EOUT}/eval.log"
  kill ${SPID} 2>/dev/null || true; wait ${SPID} 2>/dev/null || true
}

# ================================================================== run
port=18400
if [[ "$WHICH" == "all" || "$WHICH" == "train" ]]; then
  train_all
fi
if [[ "$WHICH" == "all" || "$WHICH" == "eval_libero" ]]; then
  for S in "${EVAL_SUITES[@]}"; do eval_libero_suite "$S" $((port++)); done
fi
if [[ "$WHICH" == "all" || "$WHICH" == "eval_plus" ]]; then
  for S in "${EVAL_SUITES[@]}"; do eval_plus_suite "$S" $((port++)); done
fi

echo "=================================================================="
echo "[ALL DONE] denoise+StateInit libero_all 통합 학습+평가 완료."
echo "  학습 ckpt:     ${OUT}/stage2/checkpoints/mamba_wm_final.pt"
echo "  원본 LIBERO:   results/eval/<suite>_${RUNTAG}/eval.log"
echo "  LIBERO-plus:   results/eval_plus/<suite>_${RUNTAG}/eval.log"
echo "=================================================================="
