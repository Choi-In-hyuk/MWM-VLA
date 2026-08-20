#!/bin/bash
# SMOKE test for VLA_DINO_StreamingMamba_FutureOnly_Denoise.
#
# Goal: verify the pipeline runs end-to-end (aug -> DINO -> Qwen -> predictor ->
# clean-target L_pred -> action head) WITHOUT crashing, and that pred_cos still
# trains under augmentation. NOT a real run — tiny step counts, single GPU,
# one suite (libero_object).
#
# Usage: bash scripts/run_streaming_denoise_smoke.sh
set -eo pipefail
cd "$(dirname "$0")/.."

source /home/choi/miniconda3/etc/profile.d/conda.sh
conda activate vla_jepa

CKPT=/home/choi/data/checkpoints/VLA-JEPA/LIBERO/checkpoints/VLA-JEPA-LIBERO.pt
DATA=/home/choi/data/datasets/LIBERO
MIX=libero_object
TAG=smoke_denoise
OUT=results/${TAG}

# tiny smoke settings
STAGE1_STEPS=60
STAGE2_STEPS=60
WARMUP_STEPS=10
PER_GPU_BS=4

# streaming Mamba-2 hyperparameters (same as baseline)
STREAM_STATE_DIM=1024
STREAM_DEPTH=12
STREAM_D_STATE=64
STREAM_D_CONV=1
STREAM_HEADDIM=64
STREAM_CHUNK_SIZE=64

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export PYTHONPATH=$(pwd)

common="--framework VLA_DINO_StreamingMamba_FutureOnly_Denoise --dino_backbone dinov2_vitb14 \
  --qwen_lora --lora_r 16 --lora_alpha 32 \
  --backbone_ckpt ${CKPT} --data_root ${DATA} --data_mix ${MIX} \
  --warmup_steps ${WARMUP_STEPS} --log_every 10 --num_workers 4 \
  --stream_state_dim ${STREAM_STATE_DIM} --stream_depth ${STREAM_DEPTH} \
  --stream_d_state ${STREAM_D_STATE} --stream_d_conv ${STREAM_D_CONV} \
  --stream_headdim ${STREAM_HEADDIM} --stream_chunk_size ${STREAM_CHUNK_SIZE}"

mkdir -p ${OUT}/pred ${OUT}/stage2

echo "=== [smoke] Stage 1: predictor, ${STAGE1_STEPS} steps (single GPU) ==="
python scripts/train_mamba_wm.py \
  --stage predictor ${common} \
  --max_steps ${STAGE1_STEPS} --save_every ${STAGE1_STEPS} --state_save_every ${STAGE1_STEPS} \
  --batch_size ${PER_GPU_BS} --output_dir ${OUT}/pred 2>&1 | tee ${OUT}/pred/train.log

echo "=== [smoke] Stage 2: +DiT head, ${STAGE2_STEPS} steps (single GPU) ==="
python scripts/train_mamba_wm.py \
  --stage stage2 ${common} \
  --max_steps ${STAGE2_STEPS} --save_every ${STAGE2_STEPS} --state_save_every ${STAGE2_STEPS} \
  --batch_size ${PER_GPU_BS} \
  --resume_ckpt ${OUT}/pred/checkpoints/mamba_wm_final.pt \
  --output_dir ${OUT}/stage2 2>&1 | tee ${OUT}/stage2/train.log

echo "=== [smoke] DONE. If both stages finished with finite loss, pipeline is OK. ==="
