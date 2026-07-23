#!/bin/bash
# Fair-comparison variant of run_finetune_libero_from_pretrain.sh:
# stage2 for 100k steps (matches baseline stream_libero_all's total LIBERO steps).
#
# Same recipe otherwise (pretrain ckpt reused, DDP x2, bs=16 per GPU).
#
# Usage:
#   bash scripts/run_finetune_libero_from_pretrain_100k.sh [stage2|eval|all]
set -eo pipefail
cd "$(dirname "$0")/.."

CKPT=/mnt/4TB_2/jamvla/models/VLA-JEPA-LIBERO/LIBERO/checkpoints/VLA-JEPA-LIBERO.pt
DATA=/mnt/4TB_2/jamvla/datasets/LIBERO
MIX=libero_all
TAG=finetune_libero_from_pretrain_100k
OUT=results/${TAG}
PRETRAIN_CKPT=results/pretrain_ssv2_droid/checkpoints/mamba_wm_final.pt

STAGE2_STEPS=100000     # was 30000 → matches baseline 100k
WARMUP_STEPS=5000       # scale warmup up (was 2000)
PER_GPU_BS=16
NPROC=2
EVAL_TRIALS=50

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
export MASTER_PORT=${MASTER_PORT:-29510}

common="--framework VLA_DINO_StreamingMamba --dino_backbone dinov2_vitb14 \
  --qwen_lora --lora_r 16 --lora_alpha 32 \
  --backbone_ckpt ${CKPT} --data_root ${DATA} --data_mix ${MIX} \
  --warmup_steps ${WARMUP_STEPS} --log_every 50 --num_workers 8 \
  --stream_state_dim ${STREAM_STATE_DIM} --stream_depth ${STREAM_DEPTH} \
  --stream_d_state ${STREAM_D_STATE} --stream_d_conv ${STREAM_D_CONV} \
  --stream_headdim ${STREAM_HEADDIM} --stream_chunk_size ${STREAM_CHUNK_SIZE}"

mkdir -p ${OUT}/stage2

if [[ "$WHICH" == "all" || "$WHICH" == "stage2" ]]; then
  if [[ ! -f "${PRETRAIN_CKPT}" ]]; then
    echo "ERROR: pretrain checkpoint not found: ${PRETRAIN_CKPT}"
    exit 1
  fi
  echo "=== Stage 2 (DDP x${NPROC}) from PRETRAIN: +DiT head, bs=${PER_GPU_BS}, ${STAGE2_STEPS} steps ==="
  torchrun --standalone --nproc_per_node=${NPROC} scripts/train_mamba_wm.py \
    --stage stage2 ${common} \
    --max_steps ${STAGE2_STEPS} --save_every 5000 --state_save_every 5000 \
    --batch_size ${PER_GPU_BS} \
    --resume_ckpt ${PRETRAIN_CKPT} \
    --output_dir ${OUT}/stage2 2>&1 | tee ${OUT}/stage2/train.log
fi

if [[ "$WHICH" == "all" || "$WHICH" == "eval" ]]; then
  EVAL_CKPT=${OUT}/stage2/checkpoints/mamba_wm_final.pt
  echo "=== 4-suite eval on ${EVAL_CKPT} ==="
  bash scripts/eval_libero_all.sh ${EVAL_CKPT} ${TAG} ${EVAL_TRIALS}
  echo "=== LIBERO-Plus eval on ${EVAL_CKPT} ==="
  bash scripts/eval_libero_plus_2gpu.sh ${EVAL_CKPT} ${TAG} 1
fi

echo "=== done. final ckpt: ${OUT}/stage2/checkpoints/mamba_wm_final.pt ==="
