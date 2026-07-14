#!/bin/bash
# VRAM + throughput smoke test at bs=64 per GPU.
# Runs 500 steps to measure peak VRAM and iterations/sec, then exits.
#
# Watch VRAM in another shell:
#   watch -n 2 'nvidia-smi --query-gpu=index,memory.used --format=csv,noheader'
#
# If OOM: drop PER_GPU_BS to 48 or 32 in the launcher.
set -eo pipefail
cd "$(dirname "$0")/.."

BACKBONE=/mnt/4TB_2/jamvla/models/VLA-JEPA-LIBERO/LIBERO/checkpoints/VLA-JEPA-LIBERO.pt
SSV2_ROOT=/mnt/4TB_2/jamvla/datasets/ssv2
DROID_ROOT=/mnt/4TB_2/jamvla/datasets/DroidLerobot
OUT=/tmp/smoke_pretrain_bs64

PER_GPU_BS=64
NPROC=2

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export PYTHONPATH=$(pwd)
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29510}

mkdir -p ${OUT}

echo "=== Smoke test: bs=${PER_GPU_BS} x ${NPROC} GPU, 500 steps ==="
torchrun --standalone --nproc_per_node=${NPROC} scripts/pretrain_ssv2_droid.py \
  --backbone_ckpt ${BACKBONE} \
  --ssv2_root ${SSV2_ROOT} \
  --droid_root ${DROID_ROOT} \
  --output_dir ${OUT} \
  --framework VLA_DINO_StreamingMamba \
  --dino_backbone dinov2_vitb14 \
  --qwen_lora --lora_r 16 --lora_alpha 32 \
  --batch_size ${PER_GPU_BS} --num_workers 8 \
  --max_steps 500 --warmup_steps 100 \
  --lr 2e-4 --lora_lr 2e-5 \
  --log_every 20 --save_every 10000 --state_save_every 10000 \
  --stream_state_dim 1024 --stream_depth 12 --stream_d_state 64 \
  --stream_d_conv 1 --stream_headdim 64 --stream_chunk_size 64 \
  2>&1 | tee ${OUT}/smoke.log

echo "=== done. peak VRAM (see nvidia-smi log) ==="
