#!/bin/bash
# Streaming-Mamba-2 Stage-2 training on libero_10 with TRUNCATED-BPTT
# episode streaming.
#
# Prerequisite: Stage-1 finished. Pass its `mamba_wm_final.pt`.
#
# Truncated-BPTT design (what makes this different from Stage 1):
#   - dataloader is an episode-streaming sampler (slot-based)
#   - each slot streams ONE episode's chunks in order
#   - per-slot SSM hidden state is carried ACROSS training steps (detached
#     each step → truncated BPTT)
#   - slots that just started a new episode get state zeroed
#   - training distribution matches the inference distribution where state
#     is carried chunk-by-chunk within an episode
#
# Usage:
#   bash scripts/run_streaming_libero10_stage2.sh
set -eo pipefail
cd "$(dirname "$0")/.."

CKPT=/mnt/4TB_2/jamvla/models/VLA-JEPA-LIBERO/LIBERO/checkpoints/VLA-JEPA-LIBERO.pt
DATA=/mnt/4TB_2/jamvla/datasets/LIBERO
MIX=libero_10

# Default to step-35000 ckpt (we stopped Stage 1 early). Override via env if different.
STAGE1_CKPT=${STAGE1_CKPT:-results/stream_libero10/pred/checkpoints/mamba_wm_step35000.pt}
OUT=results/stream_libero10/stage2

STAGE2_STEPS=50000
WARMUP_STEPS=5000
NUM_SLOTS=16          # per-rank batch (each slot = 1 episode streamed)
NPROC=2

# streaming Mamba-2 hyperparameters (must match Stage 1)
STREAM_STATE_DIM=1024
STREAM_DEPTH=12
STREAM_D_STATE=64
STREAM_D_CONV=1
STREAM_HEADDIM=64
STREAM_CHUNK_SIZE=64

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export PYTHONPATH=$(pwd)
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29505}

mkdir -p ${OUT}

echo "=== Stage 2 truncated-BPTT (DDP x${NPROC}): ${STAGE2_STEPS} steps, num_slots=${NUM_SLOTS}/rank ==="
torchrun --standalone --nproc_per_node=${NPROC} scripts/train_streaming_stage2.py \
    --backbone_ckpt ${CKPT} \
    --resume_ckpt ${STAGE1_CKPT} \
    --data_root ${DATA} --data_mix ${MIX} \
    --dino_backbone dinov2_vitb14 \
    --qwen_lora --lora_r 16 --lora_alpha 32 \
    --stream_state_dim ${STREAM_STATE_DIM} --stream_depth ${STREAM_DEPTH} \
    --stream_d_state ${STREAM_D_STATE} --stream_d_conv ${STREAM_D_CONV} \
    --stream_headdim ${STREAM_HEADDIM} --stream_chunk_size ${STREAM_CHUNK_SIZE} \
    --max_steps ${STAGE2_STEPS} --warmup_steps ${WARMUP_STEPS} \
    --num_slots ${NUM_SLOTS} --save_every 5000 --state_save_every 5000 \
    --log_every 50 --output_dir ${OUT} 2>&1 | tee ${OUT}/train.log

echo "=== done. final ckpt: ${OUT}/checkpoints/mamba_wm_final.pt ==="
