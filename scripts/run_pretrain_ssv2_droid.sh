#!/bin/bash
# Streaming Mamba predictor pretraining on SSv2 + Droid (DDP x2).
#
# Video + language only; no state, no action supervision.
# Stage: predictor (mamba_predictor + Qwen LoRA trainable; L_pred only).
#
# Usage:
#   bash scripts/run_pretrain_ssv2_droid.sh
set -eo pipefail
cd "$(dirname "$0")/.."

BACKBONE=/mnt/4TB_2/jamvla/models/VLA-JEPA-LIBERO/LIBERO/checkpoints/VLA-JEPA-LIBERO.pt
SSV2_ROOT=/mnt/4TB_2/jamvla/datasets/ssv2
DROID_ROOT=/mnt/4TB_2/jamvla/datasets/DroidLerobot
TAG=pretrain_ssv2_droid
OUT=results/${TAG}

MAX_STEPS=50000
WARMUP_STEPS=5000
PER_GPU_BS=32          # bs=64 OOM'd (Qwen attention too heavy at 128 imgs/step);
                       # bs=32 * 2 GPU = effective 64 fits in ~87GB/GPU
NPROC=2
# LR sqrt-scaled from bs=32 base (LR=1e-4/1e-5) to effective bs=64:
#   sqrt(64/32) = 1.4  ->  predictor 1.4e-4, LoRA 1.4e-5
PRED_LR=1.4e-4
LORA_LR=1.4e-5

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export PYTHONPATH=$(pwd)
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29508}

# Raise per-process mmap slot ceiling (PyAV opens many mmaps per Droid mp4).
# No-op if already high or if we lack sudo — best effort only.
CURRENT_MAX_MAP=$(cat /proc/sys/vm/max_map_count 2>/dev/null || echo 0)
if [[ "${CURRENT_MAX_MAP}" -lt 262144 ]]; then
  sudo -n sysctl -w vm.max_map_count=262144 2>/dev/null \
    && echo "[launcher] raised vm.max_map_count to 262144" \
    || echo "[launcher] WARN: could not raise vm.max_map_count (was ${CURRENT_MAX_MAP}); run 'sudo sysctl -w vm.max_map_count=262144' manually if OOM recurs"
fi

mkdir -p ${OUT}

# Auto-resume: pretrain_ssv2_droid.py auto-detects ${OUT}/checkpoints/training_state_latest.pt.
LATEST=${OUT}/checkpoints/training_state_latest.pt
if [[ -f "${LATEST}" ]]; then
  echo "[launcher] found ${LATEST} — training will auto-resume from it"
else
  echo "[launcher] no prior state — training will start fresh"
fi

echo "=== Pretraining (DDP x${NPROC}): SSv2+Droid, per-gpu bs=${PER_GPU_BS}, ${MAX_STEPS} steps ==="
torchrun --standalone --nproc_per_node=${NPROC} scripts/pretrain_ssv2_droid.py \
  --backbone_ckpt ${BACKBONE} \
  --ssv2_root ${SSV2_ROOT} \
  --droid_root ${DROID_ROOT} \
  --output_dir ${OUT} \
  --framework VLA_DINO_StreamingMamba \
  --dino_backbone dinov2_vitb14 \
  --qwen_lora --lora_r 16 --lora_alpha 32 \
  --batch_size ${PER_GPU_BS} --num_workers 4 \
  --max_steps ${MAX_STEPS} --warmup_steps ${WARMUP_STEPS} \
  --lr ${PRED_LR} --lora_lr ${LORA_LR} \
  --log_every 50 --save_every 5000 --state_save_every 5000 \
  --stream_state_dim 1024 --stream_depth 12 --stream_d_state 64 \
  --stream_d_conv 1 --stream_headdim 64 --stream_chunk_size 64 \
  2>&1 | tee ${OUT}/train.log

echo "=== done. pretrained ckpt: ${OUT}/checkpoints/mamba_wm_final.pt ==="
