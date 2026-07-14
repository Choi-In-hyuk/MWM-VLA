#!/bin/bash
# V2 + Qwen-LoRA two-stage chain, DDP 2-GPU.
# Same recipe as run_2stage_dino_lora.sh but launched via torchrun and using
# the resumable trainer. Both stages 15k steps, per-gpu bs=16 (effective 32).
# Usage: bash scripts/run_2stage_dino_lora_ddp.sh [stage1|stage2|all]
set -eo pipefail
cd "$(dirname "$0")/.."

CKPT=/mnt/4TB_2/jamvla/models/VLA-JEPA-LIBERO/LIBERO/checkpoints/VLA-JEPA-LIBERO.pt
DATA=/mnt/4TB_2/jamvla/datasets/LIBERO
MIX=libero_10
OUT=results/dino_mamba_diff_lora_ddp_${MIX}

STAGE1_STEPS=15000
STAGE2_STEPS=15000
PER_GPU_BS=16
NPROC=2

WHICH=${1:-all}

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export PYTHONPATH=$(pwd)
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29501}

common="--framework VLA_DINO_Mamba_Diff --dino_backbone dinov2_vitb14 \
  --qwen_lora --lora_r 16 --lora_alpha 32 \
  --backbone_ckpt ${CKPT} --data_root ${DATA} --data_mix ${MIX} \
  --warmup_steps 500 --log_every 20 --num_workers 8"

mkdir -p ${OUT}/pred ${OUT}/stage2

if [[ "$WHICH" == "all" || "$WHICH" == "stage1" ]]; then
  echo "=== V2+LoRA Stage 1 (DDP x${NPROC}): predictor, per-gpu bs=${PER_GPU_BS}, ${STAGE1_STEPS} steps ==="
  torchrun --standalone --nproc_per_node=${NPROC} scripts/train_mamba_wm.py \
    --stage predictor ${common} \
    --max_steps ${STAGE1_STEPS} --save_every 3000 --state_save_every 3000 \
    --batch_size ${PER_GPU_BS} --output_dir ${OUT}/pred 2>&1 | tee ${OUT}/pred/train.log
fi

if [[ "$WHICH" == "all" || "$WHICH" == "stage2" ]]; then
  echo "=== V2+LoRA Stage 2 (DDP x${NPROC}): +DiT head, per-gpu bs=${PER_GPU_BS}, ${STAGE2_STEPS} steps ==="
  # Resume an interrupted stage 2 with:
  #   --resume_state ${OUT}/stage2/checkpoints/training_state_latest.pt
  torchrun --standalone --nproc_per_node=${NPROC} scripts/train_mamba_wm.py \
    --stage stage2 ${common} \
    --max_steps ${STAGE2_STEPS} --save_every 3000 --state_save_every 3000 \
    --batch_size ${PER_GPU_BS} \
    --resume_ckpt ${OUT}/pred/checkpoints/mamba_wm_final.pt \
    --output_dir ${OUT}/stage2 2>&1 | tee ${OUT}/stage2/train.log
fi

echo "=== done. final ckpt: ${OUT}/stage2/checkpoints/mamba_wm_final.pt ==="
