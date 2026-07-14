#!/bin/bash
# V2 + Qwen-LoRA + change-aware context masking (training only).
#
# Stage 1 / 2 mirror run_2stage_dino_lora_ddp.sh, with two env vars enabling
# the new mask in the framework's forward path:
#   CHANGE_MASK_RATIO=0.5   # fraction of s_0 tokens hidden during training
#   CHANGE_MASK_TEMP=1.0    # sampling temperature: low=hard top-K, high=random
#
# After stage 2 finishes, LIBERO_10 eval runs with 50 trials/task.
#
# Usage:
#   bash scripts/run_changemask_v2lora.sh [stage1|stage2|eval|all]
set -eo pipefail
cd "$(dirname "$0")/.."

CKPT=/mnt/4TB_2/jamvla/models/VLA-JEPA-LIBERO/LIBERO/checkpoints/VLA-JEPA-LIBERO.pt
DATA=/mnt/4TB_2/jamvla/datasets/LIBERO
MIX=libero_10
TAG=changemask_v2lora
OUT=results/${TAG}_${MIX}

STAGE1_STEPS=30000
STAGE2_STEPS=30000
PER_GPU_BS=16
NPROC=2
EVAL_TRIALS=50
EVAL_PORT=18040

# change-mask hyperparameters (stage1 only; stage2 turns it OFF):
#   STAGE1_MASK_RATIO = base mask ratio at the start of stage1
#   STAGE1_MASK_RAMP  = linear ease-out fraction at the END of stage1 (mask -> 0)
#   CHANGE_MASK_TEMP  = soft top-K temperature (higher = more random)
# Stage2 explicitly sets CHANGE_MASK_RATIO=0 so the action head trains against the
# inference (no-mask) predictor output -- no train/test mismatch in stage2.
STAGE1_MASK_RATIO=${STAGE1_MASK_RATIO:-0.5}
STAGE1_MASK_RAMP=${STAGE1_MASK_RAMP:-0.3}
export CHANGE_MASK_TEMP=${CHANGE_MASK_TEMP:-1.0}

WHICH=${1:-all}

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export PYTHONPATH=$(pwd)
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29502}

common="--framework VLA_DINO_Mamba_Diff --dino_backbone dinov2_vitb14 \
  --qwen_lora --lora_r 16 --lora_alpha 32 \
  --backbone_ckpt ${CKPT} --data_root ${DATA} --data_mix ${MIX} \
  --warmup_steps 500 --log_every 20 --num_workers 8"

mkdir -p ${OUT}/pred ${OUT}/stage2 results/eval/libero_10_${TAG}_${EVAL_TRIALS}

echo "[run] stage1: MASK_RATIO=${STAGE1_MASK_RATIO} RAMP=${STAGE1_MASK_RAMP} TEMP=${CHANGE_MASK_TEMP} | stage2: MASK_RATIO=0 (off)"

if [[ "$WHICH" == "all" || "$WHICH" == "stage1" ]]; then
  echo "=== Stage 1 (DDP x${NPROC}): predictor (+LoRA, change-mask), bs=${PER_GPU_BS}, ${STAGE1_STEPS} steps ==="
  CHANGE_MASK_RATIO=${STAGE1_MASK_RATIO} CHANGE_MASK_RAMP=${STAGE1_MASK_RAMP} \
  torchrun --standalone --nproc_per_node=${NPROC} scripts/train_mamba_wm.py \
    --stage predictor ${common} \
    --max_steps ${STAGE1_STEPS} --save_every 3000 --state_save_every 3000 \
    --batch_size ${PER_GPU_BS} --output_dir ${OUT}/pred 2>&1 | tee ${OUT}/pred/train.log
fi

if [[ "$WHICH" == "all" || "$WHICH" == "stage2" ]]; then
  echo "=== Stage 2 (DDP x${NPROC}): +DiT head (no mask), bs=${PER_GPU_BS}, ${STAGE2_STEPS} steps ==="
  CHANGE_MASK_RATIO=0.0 CHANGE_MASK_RAMP=0.0 \
  torchrun --standalone --nproc_per_node=${NPROC} scripts/train_mamba_wm.py \
    --stage stage2 ${common} \
    --max_steps ${STAGE2_STEPS} --save_every 3000 --state_save_every 3000 \
    --batch_size ${PER_GPU_BS} \
    --resume_ckpt ${OUT}/pred/checkpoints/mamba_wm_final.pt \
    --output_dir ${OUT}/stage2 2>&1 | tee ${OUT}/stage2/train.log
fi

if [[ "$WHICH" == "all" || "$WHICH" == "eval" ]]; then
  EVAL_CKPT=${OUT}/stage2/checkpoints/mamba_wm_final.pt
  echo "=== Eval LIBERO-10 (${EVAL_TRIALS} trials/task) on ${EVAL_CKPT} ==="
  # Disable mask at eval time (paranoia: the framework already gates on training=True,
  # but make it explicit so the same shell run does not accidentally mask the eval).
  CHANGE_MASK_RATIO=0.0 bash scripts/eval_libero_dino.sh \
    libero_10 ${EVAL_TRIALS} ${EVAL_PORT} ${EVAL_CKPT} ${TAG}_${EVAL_TRIALS} 0 \
    2>&1 | tee results/eval/libero_10_${TAG}_${EVAL_TRIALS}/run.log
fi

echo "=== done. final ckpt: ${OUT}/stage2/checkpoints/mamba_wm_final.pt ==="
echo "=== eval log:        results/eval/libero_10_${TAG}_${EVAL_TRIALS}/eval.log ==="
