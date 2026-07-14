#!/bin/bash
# Streaming-Mamba world model (VLA_DINO_StreamingMamba) on libero_10.
#
# Architecture vs baseline (VLA_DINO_Mamba_Diff):
#   - predictor sees TWO frames (past=t-H, present=t) instead of one.
#   - predicts s_(t+H). robot_state is NOT fed to the predictor (vision-only);
#     it still goes to the action head, same as baseline.
#   - no SSM state carry at inference (states=None always). Cold start at t=0
#     is naturally handled by the dataloader's front-padding (frame 0 replicate).
#
# Recipe mirrors `run_2stage_dino_lora_ddp.sh` exactly (same data, same step
# count, same LoRA, same DDP layout) so the comparison is apples-to-apples.
#
# Usage:
#   bash scripts/run_streaming_libero10.sh [stage1|stage2|all]
set -eo pipefail
cd "$(dirname "$0")/.."

CKPT=/mnt/4TB_2/jamvla/models/VLA-JEPA-LIBERO/LIBERO/checkpoints/VLA-JEPA-LIBERO.pt
DATA=/mnt/4TB_2/jamvla/datasets/LIBERO
MIX=libero_10
TAG=stream_libero10
OUT=results/${TAG}

STAGE1_STEPS=15000
STAGE2_STEPS=15000
WARMUP_STEPS=500
PER_GPU_BS=16
NPROC=2

# streaming Mamba-2 hyperparameters
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
export MASTER_PORT=${MASTER_PORT:-29504}

common="--framework VLA_DINO_StreamingMamba --dino_backbone dinov2_vitb14 \
  --qwen_lora --lora_r 16 --lora_alpha 32 \
  --backbone_ckpt ${CKPT} --data_root ${DATA} --data_mix ${MIX} \
  --warmup_steps ${WARMUP_STEPS} --log_every 20 --num_workers 8 \
  --stream_state_dim ${STREAM_STATE_DIM} --stream_depth ${STREAM_DEPTH} \
  --stream_d_state ${STREAM_D_STATE} --stream_d_conv ${STREAM_D_CONV} \
  --stream_headdim ${STREAM_HEADDIM} --stream_chunk_size ${STREAM_CHUNK_SIZE}"

mkdir -p ${OUT}/pred ${OUT}/stage2

if [[ "$WHICH" == "all" || "$WHICH" == "stage1" ]]; then
  echo "=== Stage 1 (DDP x${NPROC}): predictor, bs=${PER_GPU_BS}/gpu, ${STAGE1_STEPS} steps ==="
  torchrun --standalone --nproc_per_node=${NPROC} scripts/train_mamba_wm.py \
    --stage predictor ${common} \
    --max_steps ${STAGE1_STEPS} --save_every 3000 --state_save_every 3000 \
    --batch_size ${PER_GPU_BS} --output_dir ${OUT}/pred 2>&1 | tee ${OUT}/pred/train.log
fi

if [[ "$WHICH" == "all" || "$WHICH" == "stage2" ]]; then
  echo "=== Stage 2 (DDP x${NPROC}): +DiT action head, bs=${PER_GPU_BS}/gpu, ${STAGE2_STEPS} steps ==="
  torchrun --standalone --nproc_per_node=${NPROC} scripts/train_mamba_wm.py \
    --stage stage2 ${common} \
    --max_steps ${STAGE2_STEPS} --save_every 3000 --state_save_every 3000 \
    --batch_size ${PER_GPU_BS} \
    --resume_ckpt ${OUT}/pred/checkpoints/mamba_wm_final.pt \
    --output_dir ${OUT}/stage2 2>&1 | tee ${OUT}/stage2/train.log
fi

echo "=== done. final ckpt: ${OUT}/stage2/checkpoints/mamba_wm_final.pt ==="
