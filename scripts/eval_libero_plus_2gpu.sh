#!/bin/bash
# LIBERO-Plus eval across 7 perturbations split over 2 GPUs (in parallel).
#
# GPU 0 handles perturbations 1-4 (Background/Camera/Language/Light)
# GPU 1 handles perturbations 5-7 (Objects/Robot/Sensor)
# Both run in the background; this script waits for both to finish, then
# concatenates their summaries.
#
# Usage:
#   bash scripts/eval_libero_plus_2gpu.sh <ckpt> <tag> [num_trials]
set -uo pipefail
cd "$(dirname "$0")/.."

CKPT=${1:?usage: eval_libero_plus_2gpu.sh <ckpt> <tag> [num_trials]}
TAG=${2:?usage: eval_libero_plus_2gpu.sh <ckpt> <tag> [num_trials]}
NUM_TRIALS=${3:-1}

OUT_ROOT=results/eval_libero_plus/${TAG}
mkdir -p "${OUT_ROOT}"

echo "=== LIBERO-Plus split: GPU0 = 1,2,3,4 | GPU1 = 5,6,7 ==="

bash scripts/eval_libero_plus_all.sh "${CKPT}" "${TAG}" "${NUM_TRIALS}" 0 "1,2,3,4" \
    > "${OUT_ROOT}/gpu0.log" 2>&1 &
PID0=$!

bash scripts/eval_libero_plus_all.sh "${CKPT}" "${TAG}" "${NUM_TRIALS}" 1 "5,6,7" \
    > "${OUT_ROOT}/gpu1.log" 2>&1 &
PID1=$!

echo "GPU0 pid=${PID0}  GPU1 pid=${PID1}"
echo "Follow live: tail -f ${OUT_ROOT}/gpu0.log ${OUT_ROOT}/gpu1.log"

wait ${PID0}; RC0=$?
wait ${PID1}; RC1=$?

echo ""
echo "=== GPU0 exit=${RC0}, GPU1 exit=${RC1} ==="

# Merge per-GPU summaries into one
FINAL=${OUT_ROOT}/summary.txt
cat "${OUT_ROOT}"/summary_gpu*.txt 2>/dev/null > "${FINAL}" || true

echo "=== LIBERO-Plus final summary (${TAG}, ${NUM_TRIALS} trials/task) ==="
cat "${FINAL}"
echo ""
echo "Summary: ${FINAL}"
