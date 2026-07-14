#!/bin/bash
# Run LIBERO eval on all four suites for the same checkpoint.
#
# Each suite is evaluated by `eval_libero_dino.sh` (server_policy + rollout
# client). Suites are run sequentially because the server holds the GPU.
#
# Usage:
#   bash scripts/eval_libero_all.sh <ckpt> <tag> [trials_per_task]
# Outputs:
#   results/eval/<suite>_<tag>_<trials>/{eval.log, rollout_*.mp4, run.log}
set -uo pipefail
cd "$(dirname "$0")/.."

CKPT=${1:?usage: eval_libero_all.sh <ckpt> <tag> [trials]}
TAG=${2:?usage: eval_libero_all.sh <ckpt> <tag> [trials]}
TRIALS=${3:-50}

# distinct ports per suite so a stale server from a previous run can't collide
declare -A PORTS=(
  [libero_spatial]=18050
  [libero_object]=18051
  [libero_goal]=18052
  [libero_10]=18053
)

# run in this order: easiest -> hardest, so failures surface earlier
SUITES=(libero_spatial libero_object libero_goal libero_10)

mkdir -p results/eval
SUMMARY=results/eval/${TAG}_${TRIALS}_summary.txt
: > "${SUMMARY}"

for SUITE in "${SUITES[@]}"; do
  PORT=${PORTS[$SUITE]}
  EVAL_TAG=${TAG}_${TRIALS}
  OUT_DIR=results/eval/${SUITE}_${EVAL_TAG}
  mkdir -p "${OUT_DIR}"
  echo ""
  echo "============================================================"
  echo "  Suite: ${SUITE} | trials/task=${TRIALS} | port=${PORT}"
  echo "  Out:   ${OUT_DIR}"
  echo "============================================================"

  bash scripts/eval_libero_dino.sh \
      "${SUITE}" "${TRIALS}" "${PORT}" "${CKPT}" "${EVAL_TAG}" 0 \
      2>&1 | tee "${OUT_DIR}/run.log" || true

  # extract this suite's final total success rate from the eval log
  RATE=$(grep -oE "Total success rate: [0-9.]+" "${OUT_DIR}/eval.log" 2>/dev/null | tail -1 | awk '{print $4}')
  printf "%-18s  rate=%s\n" "${SUITE}" "${RATE:-N/A}" | tee -a "${SUMMARY}"
done

echo ""
echo "=== 4-suite summary (${TAG}, ${TRIALS} trials/task) ==="
cat "${SUMMARY}"
echo ""
echo "Logs:    results/eval/<suite>_${TAG}_${TRIALS}/eval.log"
echo "Summary: ${SUMMARY}"
