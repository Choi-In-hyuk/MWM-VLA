#!/bin/bash
# Dual-Mamba (V2+LoRA world model + Mamba action head) two-stage chain, DDP 2-GPU.
#
# Stage 1: same recipe as run_2stage_dino_lora.sh (predictor + Qwen LoRA, L_pred).
#          15k steps -- predictor is fine-tuning a pretrained latent geometry, not scratch.
# Stage 2: predictor fine-tune + Mamba action head (scratch) + Qwen LoRA,
#          L = a*L_pred + b*L_action.
#          The Mamba action head is trained from scratch, so stage 2 runs for
#          300k steps (~9.6M sample views at effective batch 32 -- roughly 10%
#          of MambaVLA's training volume, much more than V2+LoRA's 15k).
#
# Resumable: every --state_save_every steps a full training_state_latest.pt is
# written. To restart from the most recent state, pass --resume_state to the
# stage 2 invocation (uncomment block below).
#
# Usage: bash scripts/run_2stage_dualmamba.sh [stage1|stage2|all]
#        (default: all)
set -eo pipefail
cd "$(dirname "$0")/.."

CKPT=/mnt/4TB_2/jamvla/models/VLA-JEPA-LIBERO/LIBERO/checkpoints/VLA-JEPA-LIBERO.pt
DATA=/mnt/4TB_2/jamvla/datasets/LIBERO
MIX=libero_10
OUT=results/dual_mamba_${MIX}

STAGE1_STEPS=15000
STAGE2_STEPS=300000
PER_GPU_BS=16          # 2 GPUs -> effective batch 32
NPROC=2

WHICH=${1:-all}

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export PYTHONPATH=$(pwd)
# rendezvous on a free local port; change if 29500 is in use.
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29500}

common="--framework VLA_DINO_DualMamba --dino_backbone dinov2_vitb14 \
  --qwen_lora --lora_r 16 --lora_alpha 32 \
  --action_mamba_layers 5 --action_embed_dim 256 --action_inference_steps 10 \
  --backbone_ckpt ${CKPT} --data_root ${DATA} --data_mix ${MIX} \
  --warmup_steps 2000 --log_every 20 --num_workers 8"

mkdir -p ${OUT}/pred ${OUT}/stage2

if [[ "$WHICH" == "all" || "$WHICH" == "stage1" ]]; then
  echo "=== Stage 1 (DDP x${NPROC}): predictor (+LoRA), per-gpu bs=${PER_GPU_BS}, ${STAGE1_STEPS} steps ==="
  torchrun --standalone --nproc_per_node=${NPROC} scripts/train_mamba_wm.py \
    --stage predictor ${common} \
    --max_steps ${STAGE1_STEPS} --save_every 3000 --state_save_every 3000 \
    --batch_size ${PER_GPU_BS} --output_dir ${OUT}/pred 2>&1 | tee ${OUT}/pred/train.log
fi

if [[ "$WHICH" == "all" || "$WHICH" == "stage2" ]]; then
  echo "=== Stage 2 (DDP x${NPROC}): +Mamba head scratch, per-gpu bs=${PER_GPU_BS}, ${STAGE2_STEPS} steps ==="
  # To resume an interrupted stage 2, swap --resume_ckpt for:
  #   --resume_state ${OUT}/stage2/checkpoints/training_state_latest.pt
  torchrun --standalone --nproc_per_node=${NPROC} scripts/train_mamba_wm.py \
    --stage stage2 ${common} \
    --max_steps ${STAGE2_STEPS} --save_every 20000 --state_save_every 5000 \
    --batch_size ${PER_GPU_BS} \
    --resume_ckpt ${OUT}/pred/checkpoints/mamba_wm_final.pt \
    --output_dir ${OUT}/stage2 2>&1 | tee ${OUT}/stage2/train.log
fi

echo "=== done. final ckpt: ${OUT}/stage2/checkpoints/mamba_wm_final.pt ==="
