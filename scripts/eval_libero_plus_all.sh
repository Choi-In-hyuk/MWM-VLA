#!/bin/bash
# LIBERO-Plus eval: run a subset of 7 perturbations sequentially on one GPU.
#
# Adapted from examples/LIBERO-Plus/eval_libero_plus.sh for our 2-GPU server
# (original ran 7 perturbations in parallel on 7 GPUs).
#
# For each perturbation:
#   1. start server_policy (vla_jepa env) on port
#   2. run eval_libero.py from liberoplus_env (has LIBERO-Plus sim installed)
#   3. wait, extract success rate, log
#
# Usage:
#   bash scripts/eval_libero_plus_all.sh <ckpt> <tag> [num_trials] [cuda] [subset]
#
# `subset` is a comma-separated list of perturbation indices (1-7). Defaults to
# all 7. Use e.g. subset="1,2,3,4" on GPU 0 and subset="5,6,7" on GPU 1 to
# split the workload across 2 GPUs in parallel.
set -uo pipefail
cd "$(dirname "$0")/.."

CKPT=${1:?usage: eval_libero_plus_all.sh <ckpt> <tag> [num_trials] [cuda] [subset]}
TAG=${2:?usage: eval_libero_plus_all.sh <ckpt> <tag> [num_trials] [cuda] [subset]}
NUM_TRIALS=${3:-1}   # LIBERO-Plus paper uses 1 trial per perturbed task
CUDA=${4:-0}         # which GPU this run uses
SUBSET=${5:-"1,2,3,4,5,6,7"}   # perturbation indices to run (1-7)

# Env paths
LIBERO_PLUS_ROOT=/home/choi/LIBERO-plus
LIBEROPLUS_ENV_PY=/home/choi/miniconda3/envs/liberoplus_env/bin/python
SERVER_PY=/home/choi/miniconda3/envs/vla_jepa/bin/python

# Isolated libero config (points to LIBERO-plus assets, not overwriting user ~/.libero)
export LIBERO_CONFIG_PATH=/home/choi/VLA-WM/liberoplus_config
export LIBERO_PLUS_TASK_CLASSIFICATION=${LIBERO_PLUS_ROOT}/libero/libero/benchmark/task_classification.json
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1

# server-side (training env, no LIBERO sim needed)
SERVER_PYTHONPATH="$(pwd):${PYTHONPATH:-}"
# client-side (needs LIBERO-Plus sim + our repo code)
CLIENT_PYTHONPATH="${LIBERO_PLUS_ROOT}:$(pwd):$(pwd)/examples/LIBERO-Plus:${PYTHONPATH:-}"

PERTURBATIONS=(
  "Background Textures"
  "Camera Viewpoints"
  "Language Instructions"
  "Light Conditions"
  "Objects Layout"
  "Robot Initial States"
  "Sensor Noise"
)

BASE_PORT=$((19080 + CUDA * 100))   # keep ports per-GPU disjoint when running in parallel
OUT_ROOT=results/eval_libero_plus/${TAG}
mkdir -p "${OUT_ROOT}"
SUMMARY=${OUT_ROOT}/summary_gpu${CUDA}.txt
: > "${SUMMARY}"

# Convert subset "1,2,3" -> space list for iteration
SUBSET_LIST=${SUBSET//,/ }

INDEX=0
for PERT in "${PERTURBATIONS[@]}"; do
  INDEX=$((INDEX+1))
  # skip if not in this GPU's subset
  case " ${SUBSET_LIST} " in *" ${INDEX} "*) : ;; *) continue ;; esac

  PORT=$((BASE_PORT+INDEX))
  SLUG=${PERT// /_}
  OUT_DIR=${OUT_ROOT}/${SLUG}
  mkdir -p "${OUT_DIR}"

  echo ""
  echo "============================================================"
  echo "  [gpu ${CUDA} | ${INDEX}/7] Perturbation: ${PERT}"
  echo "  Port: ${PORT} | trials/task=${NUM_TRIALS}"
  echo "  Out:  ${OUT_DIR}"
  echo "============================================================"

  SERVER_LOG=${OUT_DIR}/server.log

  # start server (pinned to this GPU via CUDA_VISIBLE_DEVICES so PyTorch never
  # sees the other GPU — prevents device-0 buffers leaking into a device-1 model).
  CUDA_VISIBLE_DEVICES=${CUDA} PYTHONPATH="${SERVER_PYTHONPATH}" ${SERVER_PY} deployment/model_server/server_policy.py \
      --ckpt_path "${CKPT}" --port ${PORT} --cuda 0 > "${SERVER_LOG}" 2>&1 &
  SERVER_PID=$!
  trap "kill ${SERVER_PID} 2>/dev/null || true" EXIT

  # wait for server ready
  for i in $(seq 1 120); do
      grep -q "server listening" "${SERVER_LOG}" 2>/dev/null && { echo "server up."; break; }
      kill -0 ${SERVER_PID} 2>/dev/null || { echo "SERVER DIED:"; tail -20 "${SERVER_LOG}"; exit 1; }
      sleep 2
  done

  # run eval
  PYTHONPATH="${CLIENT_PYTHONPATH}" ${LIBEROPLUS_ENV_PY} examples/LIBERO/eval_libero.py \
      --args.pretrained-path "${CKPT}" \
      --args.host 127.0.0.1 --args.port ${PORT} \
      --args.task-suite-name libero_mix \
      --args.category_value "${PERT}" \
      --args.num-trials-per-task "${NUM_TRIALS}" \
      --args.video-out-path "${OUT_DIR}" \
      --args.with_state "true" \
      --args.action-chunk-size 0 \
      --args.seed 7 2>&1 | tee "${OUT_DIR}/eval.log" || true

  # kill server for this perturbation
  kill ${SERVER_PID} 2>/dev/null || true
  wait ${SERVER_PID} 2>/dev/null || true
  trap - EXIT

  RATE=$(grep -oE "Total success rate: [0-9.]+" "${OUT_DIR}/eval.log" 2>/dev/null | tail -1 | awk '{print $4}')
  printf "%-24s  rate=%s\n" "${SLUG}" "${RATE:-N/A}" | tee -a "${SUMMARY}"
done

echo ""
echo "=== LIBERO-Plus summary (${TAG}, ${NUM_TRIALS} trials/task) ==="
cat "${SUMMARY}"
echo ""
echo "Summary: ${SUMMARY}"
