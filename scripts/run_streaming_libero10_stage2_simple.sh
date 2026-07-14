#!/bin/bash
# Streaming-Mamba-2 Stage 2 (SIMPLE — same recipe as Stage 1, just stage=stage2)
#
# After abandoning truncated-BPTT (it stalled both pred_cos and action_loss),
# we revert to the baseline V2 recipe:
#   - random-window dataloader (chunk pairs from any episode)
#   - within a sample, M=2 chunks share SSM state (Stage-1-style BPTT)
#   - across samples, no state carry (independent windows)
#   - train predictor + diffusion head + Qwen LoRA jointly (L_pred + L_action)
#
# Inference still carries SSM state across chunks within an episode (the
# WebsocketServer calls model.episode_reset() at each episode start).
# Train↔infer distribution mismatch is accepted; we rely on the SSM's
# implicit decay to keep that gap small.
#
# Prerequisite: Stage-1 finished. Loads its `mamba_wm_step35000.pt`.
set -eo pipefail
cd "$(dirname "$0")/.."

CKPT=/mnt/4TB_2/jamvla/models/VLA-JEPA-LIBERO/LIBERO/checkpoints/VLA-JEPA-LIBERO.pt
DATA=/mnt/4TB_2/jamvla/datasets/LIBERO
MIX=libero_10

STAGE1_CKPT=${STAGE1_CKPT:-results/stream_libero10/pred/checkpoints/mamba_wm_step35000.pt}
OUT=results/stream_libero10/stage2_simple

STAGE2_STEPS=50000
WARMUP_STEPS=5000
PER_GPU_BS=16
NPROC=2

SEQ_LEN_M=2
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
export MASTER_PORT=${MASTER_PORT:-29506}

mkdir -p ${OUT}

common="--framework VLA_DINO_StreamingMamba --dino_backbone dinov2_vitb14 \
  --qwen_lora --lora_r 16 --lora_alpha 32 \
  --backbone_ckpt ${CKPT} --data_root ${DATA} --data_mix ${MIX} \
  --warmup_steps ${WARMUP_STEPS} --log_every 50 --num_workers 8 \
  --stream_state_dim ${STREAM_STATE_DIM} --stream_depth ${STREAM_DEPTH} \
  --stream_d_state ${STREAM_D_STATE} --stream_d_conv ${STREAM_D_CONV} \
  --stream_headdim ${STREAM_HEADDIM} --stream_chunk_size ${STREAM_CHUNK_SIZE} \
  --seq_len_M ${SEQ_LEN_M}"

echo "=== Stage 2 (DDP x${NPROC}, M=${SEQ_LEN_M} BPTT random-window): ${STAGE2_STEPS} steps, bs=${PER_GPU_BS}/gpu ==="
echo "    resuming Stage-1 ckpt: ${STAGE1_CKPT}"
torchrun --standalone --nproc_per_node=${NPROC} scripts/train_mamba_wm.py \
    --stage stage2 ${common} \
    --max_steps ${STAGE2_STEPS} --save_every 5000 --state_save_every 5000 \
    --batch_size ${PER_GPU_BS} \
    --resume_ckpt ${STAGE1_CKPT} \
    --output_dir ${OUT} 2>&1 | tee ${OUT}/train.log

echo "=== done. final ckpt: ${OUT}/checkpoints/mamba_wm_final.pt ==="
