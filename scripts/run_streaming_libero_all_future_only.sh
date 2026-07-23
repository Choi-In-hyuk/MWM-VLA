#!/bin/bash
# Streaming v2 (2-frame WM) with FUTURE-ONLY action-head conditioning.
#
# Same recipe as scripts/run_streaming_libero_all.sh, but:
#   - framework = VLA_DINO_StreamingMamba_FutureOnly
#   - action head sees cond = cond_proj(s_target_pred) only (no s_present)
#   - stage1: 30k, stage2: 30k (shorter than baseline 50k+50k, per request)
#
# No SSv2/Droid pretraining. libero_all only.
#
# Usage:
#   bash scripts/run_streaming_libero_all_future_only.sh [stage1|stage2|eval|all]
set -eo pipefail
cd "$(dirname "$0")/.."

CKPT=/mnt/4TB_2/jamvla/models/VLA-JEPA-LIBERO/LIBERO/checkpoints/VLA-JEPA-LIBERO.pt
DATA=/mnt/4TB_2/jamvla/datasets/LIBERO
MIX=libero_all
TAG=stream_libero_all_future_only
OUT=results/${TAG}

STAGE1_STEPS=30000
STAGE2_STEPS=30000
WARMUP_STEPS=3000
PER_GPU_BS=16
NPROC=2
EVAL_TRIALS=50

# streaming Mamba-2 hyperparameters (same as baseline)
STREAM_STATE_DIM=1024
STREAM_DEPTH=12
STREAM_D_STATE=64
STREAM_D_CONV=1
STREAM_HEADDIM=64
STREAM_CHUNK_SIZE=64

WHICH=${1:-all}

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export PYTHONPATH=$(pwd)
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29511}

common="--framework VLA_DINO_StreamingMamba_FutureOnly --dino_backbone dinov2_vitb14 \
  --qwen_lora --lora_r 16 --lora_alpha 32 \
  --backbone_ckpt ${CKPT} --data_root ${DATA} --data_mix ${MIX} \
  --warmup_steps ${WARMUP_STEPS} --log_every 50 --num_workers 8 \
  --stream_state_dim ${STREAM_STATE_DIM} --stream_depth ${STREAM_DEPTH} \
  --stream_d_state ${STREAM_D_STATE} --stream_d_conv ${STREAM_D_CONV} \
  --stream_headdim ${STREAM_HEADDIM} --stream_chunk_size ${STREAM_CHUNK_SIZE}"

mkdir -p ${OUT}/pred ${OUT}/stage2

if [[ "$WHICH" == "all" || "$WHICH" == "stage1" ]]; then
  echo "=== Stage 1 (DDP x${NPROC}): predictor, per-gpu bs=${PER_GPU_BS}, ${STAGE1_STEPS} steps ==="
  torchrun --standalone --nproc_per_node=${NPROC} scripts/train_mamba_wm.py \
    --stage predictor ${common} \
    --max_steps ${STAGE1_STEPS} --save_every 5000 --state_save_every 5000 \
    --batch_size ${PER_GPU_BS} --output_dir ${OUT}/pred 2>&1 | tee ${OUT}/pred/train.log
fi

if [[ "$WHICH" == "all" || "$WHICH" == "stage2" ]]; then
  echo "=== Stage 2 (DDP x${NPROC}): +DiT head, per-gpu bs=${PER_GPU_BS}, ${STAGE2_STEPS} steps ==="
  torchrun --standalone --nproc_per_node=${NPROC} scripts/train_mamba_wm.py \
    --stage stage2 ${common} \
    --max_steps ${STAGE2_STEPS} --save_every 5000 --state_save_every 5000 \
    --batch_size ${PER_GPU_BS} \
    --resume_ckpt ${OUT}/pred/checkpoints/mamba_wm_final.pt \
    --output_dir ${OUT}/stage2 2>&1 | tee ${OUT}/stage2/train.log
fi

if [[ "$WHICH" == "all" || "$WHICH" == "eval" ]]; then
  EVAL_CKPT=${OUT}/stage2/checkpoints/mamba_wm_final.pt
  echo "=== Eval 4-suite (${EVAL_TRIALS} trials/task) on ${EVAL_CKPT} ==="
  bash scripts/eval_libero_all.sh ${EVAL_CKPT} ${TAG} ${EVAL_TRIALS}

  echo "=== Eval LIBERO-Plus (7 categories, 2-GPU split) on ${EVAL_CKPT} ==="
  bash scripts/eval_libero_plus_2gpu.sh ${EVAL_CKPT} ${TAG} 1
fi

echo "=== done. final ckpt: ${OUT}/stage2/checkpoints/mamba_wm_final.pt ==="
