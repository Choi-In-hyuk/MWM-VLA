#!/bin/bash
# Wait for baseline LIBERO-Plus eval to fully finish (all 7 categories),
# then launch a 100k-step fair-comparison finetune of our pretrained model.
set -uo pipefail
cd "$(dirname "$0")/.."

BASE_DIR=results/eval_libero_plus/stream_libero_all_plus
CATS=("Background_Textures" "Camera_Viewpoints" "Language_Instructions" \
      "Light_Conditions" "Objects_Layout" "Robot_Initial_States" "Sensor_Noise")

all_done() {
  for c in "${CATS[@]}"; do
    grep -q "Total success rate" "${BASE_DIR}/${c}/eval.log" 2>/dev/null || return 1
  done
  return 0
}

echo "[chain100k] $(date): waiting for baseline eval to finish all 7 categories..."
until all_done; do
  sleep 600  # check every 10 min
done

echo "[chain100k] $(date): baseline eval done. Waiting 30s then starting 100k finetune..."
sleep 30

bash scripts/run_finetune_libero_from_pretrain_100k.sh all > /tmp/finetune_100k.log 2>&1
echo "[chain100k] $(date): 100k finetune + eval done."
