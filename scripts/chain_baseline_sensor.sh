#!/bin/bash
# Wait for our SSv2+Droid-pretrained model's Sensor Noise eval to finish, then
# automatically kick off the no-pretrain baseline's Sensor Noise eval on GPU 1.
set -uo pipefail
cd "$(dirname "$0")/.."

OURS_LOG=results/eval_libero_plus/finetune_libero_from_pretrain/Sensor_Noise/eval.log
BASELINE_CKPT=results/stream_libero_all/stage2/checkpoints/mamba_wm_final.pt

echo "[chain] $(date): waiting for ${OURS_LOG} to finish (Total success rate line)..."
until grep -q "Total success rate" "${OURS_LOG}" 2>/dev/null; do
    sleep 300
done

echo "[chain] $(date): our Sensor Noise done. Starting baseline Sensor Noise on GPU 1 in 30s..."
sleep 30

bash scripts/eval_libero_plus_all.sh \
    "${BASELINE_CKPT}" \
    stream_libero_all_plus 1 1 "7" > /tmp/baseline_gpu1_sensor.log 2>&1
echo "[chain] $(date): baseline Sensor Noise done."
