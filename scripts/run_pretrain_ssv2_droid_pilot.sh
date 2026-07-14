#!/bin/bash
# Phase 1 PILOT: Streaming Mamba predictor pretraining on SSv2 + Droid.
#
# Purpose: cheaply verify that SSv2+Droid pretraining actually helps LIBERO
# fine-tune before committing to the full 50k-step run. Same recipe as the
# full launcher (bs=64 per GPU, LR 2e-4/2e-5) but 30% of the steps.
#
# Success criterion (measured AFTER LIBERO fine-tune with these weights):
#   * LIBERO 4-suite Avg > 93.7% (baseline stream_libero_all no-pretrain)
#   -> GO to Phase 2 (full 50k run)
#   * LIBERO 4-suite Avg <= 93.7%
#   -> STOP. Reconsider recipe (mix ratio, LoRA on/off, predictor size, etc.)
#
# Usage:
#   bash scripts/run_pretrain_ssv2_droid_pilot.sh
set -eo pipefail
cd "$(dirname "$0")/.."

BACKBONE=/mnt/4TB_2/jamvla/models/VLA-JEPA-LIBERO/LIBERO/checkpoints/VLA-JEPA-LIBERO.pt
SSV2_ROOT=/mnt/4TB_2/jamvla/datasets/ssv2
DROID_ROOT=/mnt/4TB_2/jamvla/datasets/DroidLerobot
TAG=pretrain_ssv2_droid_pilot
OUT=results/${TAG}

MAX_STEPS=15000        # pilot: 30% of the full 50k
WARMUP_STEPS=1500      # scaled proportionally
PER_GPU_BS=32          # OOM at 64 (Qwen attention + Mamba MLP too large); 32 fits
NPROC=2                # effective batch = 64 (2x VLA-JEPA-relative-to-2GPU baseline)
PRED_LR=1.4e-4         # sqrt(64/32) ~ 1.4x scaling from bs=32 base LR
LORA_LR=1.4e-5

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export PYTHONPATH=$(pwd)
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29511}

mkdir -p ${OUT}

echo "=== PILOT (DDP x${NPROC}): SSv2+Droid, bs=${PER_GPU_BS}, ${MAX_STEPS} steps ==="
torchrun --standalone --nproc_per_node=${NPROC} scripts/pretrain_ssv2_droid.py \
  --backbone_ckpt ${BACKBONE} \
  --ssv2_root ${SSV2_ROOT} \
  --droid_root ${DROID_ROOT} \
  --output_dir ${OUT} \
  --framework VLA_DINO_StreamingMamba \
  --dino_backbone dinov2_vitb14 \
  --qwen_lora --lora_r 16 --lora_alpha 32 \
  --batch_size ${PER_GPU_BS} --num_workers 8 \
  --max_steps ${MAX_STEPS} --warmup_steps ${WARMUP_STEPS} \
  --lr ${PRED_LR} --lora_lr ${LORA_LR} \
  --log_every 50 --save_every 3000 --state_save_every 3000 \
  --stream_state_dim 1024 --stream_depth 12 --stream_d_state 64 \
  --stream_d_conv 1 --stream_headdim 64 --stream_chunk_size 64 \
  2>&1 | tee ${OUT}/train.log

echo "=== PILOT done. ckpt: ${OUT}/checkpoints/mamba_wm_final.pt ==="
echo "=== Next: bash scripts/run_finetune_libero_from_pilot.sh ==="
