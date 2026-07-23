#!/bin/bash
# Train our streaming Mamba VLA on CALVIN D dataset (same-env baseline).
#
# For long-horizon (5-task chain) evaluation on CALVIN D.
# Uses --dataset_type calvin to route the LIBERO-style trainer to our
# CalvinDataset loader (npz per frame + auto_lang_ann.npy).
#
# Usage:
#   bash scripts/run_calvin_D.sh [stage2|eval|all]
set -eo pipefail
cd "$(dirname "$0")/.."

CKPT=/mnt/4TB_2/jamvla/models/VLA-JEPA-LIBERO/LIBERO/checkpoints/VLA-JEPA-LIBERO.pt
DATA=/mnt/4TB_2/jamvla/datasets/calvin/task_D_D
MIX=calvin_D
TAG=calvin_D
OUT=results/${TAG}

STAGE2_STEPS=50000
WARMUP_STEPS=2000
PER_GPU_BS=16
NPROC=2

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
export MASTER_PORT=${MASTER_PORT:-29520}

common="--framework VLA_DINO_StreamingMamba --dino_backbone dinov2_vitb14 \
  --qwen_lora --lora_r 16 --lora_alpha 32 \
  --backbone_ckpt ${CKPT} --data_root ${DATA} --data_mix ${MIX} \
  --dataset_type calvin --calvin_split training \
  --warmup_steps ${WARMUP_STEPS} --log_every 50 --num_workers 8 \
  --stream_state_dim ${STREAM_STATE_DIM} --stream_depth ${STREAM_DEPTH} \
  --stream_d_state ${STREAM_D_STATE} --stream_d_conv ${STREAM_D_CONV} \
  --stream_headdim ${STREAM_HEADDIM} --stream_chunk_size ${STREAM_CHUNK_SIZE}"

mkdir -p ${OUT}/stage2

if [[ "$WHICH" == "all" || "$WHICH" == "stage2" ]]; then
  echo "=== Stage 2 (DDP x${NPROC}) on CALVIN D, bs=${PER_GPU_BS}, ${STAGE2_STEPS} steps ==="
  torchrun --standalone --nproc_per_node=${NPROC} scripts/train_mamba_wm.py \
    --stage stage2 ${common} \
    --max_steps ${STAGE2_STEPS} --save_every 5000 --state_save_every 5000 \
    --batch_size ${PER_GPU_BS} \
    --output_dir ${OUT}/stage2 2>&1 | tee ${OUT}/stage2/train.log
fi

if [[ "$WHICH" == "all" || "$WHICH" == "eval" ]]; then
  EVAL_CKPT=${OUT}/stage2/checkpoints/mamba_wm_final.pt
  echo "=== CALVIN 5-task chain eval on ${EVAL_CKPT} ==="
  bash scripts/eval_calvin_LH.sh ${EVAL_CKPT} ${TAG} ${DATA} 1000
fi

echo "=== done. final ckpt: ${OUT}/stage2/checkpoints/mamba_wm_final.pt ==="
