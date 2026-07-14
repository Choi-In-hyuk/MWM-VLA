#!/bin/bash
# LIBERO eval for VLA_DINO_Mamba (server_policy + eval_libero.py), single vla_jepa env.
# Usage: bash scripts/eval_libero_dino.sh <suite> <num_trials> [port]
set -uo pipefail
cd "$(dirname "$0")/.."

SUITE=${1:-libero_10}
NUM_TRIALS=${2:-2}
PORT=${3:-18000}
CKPT=${4:-results/dino_mamba_libero_10/id/checkpoints/mamba_wm_final.pt}
TAG=${5:-dino}
CHUNK=${6:-0}   # 0=use config (full horizon); >0 = re-plan every N steps
PY=/home/choi/miniconda3/envs/vla_jepa/bin/python
OUT=results/eval/${SUITE}_${TAG}
SERVER_LOG=/tmp/dino_server_${PORT}.log

export LIBERO_HOME=${LIBERO_HOME:-/home/choi/MambaVLA/LIBERO}
export LIBERO_CONFIG_PATH=${LIBERO_CONFIG_PATH:-${LIBERO_HOME}/libero}
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
mkdir -p "${OUT}"

# server runs in the training env (vla_jepa) only — no LIBERO sim deps needed.
SERVER_PYTHONPATH="$(pwd):${PYTHONPATH:-}"
# client picks up LIBERO + sim deps from the standalone mambavla_env site-packages.
CLIENT_PYTHONPATH="${LIBERO_HOME}:/home/choi/MambaVLA/mambavla_env/lib/python3.10/site-packages:$(pwd):${PYTHONPATH:-}"

echo "=== starting model server (port ${PORT}) ==="
PYTHONPATH="${SERVER_PYTHONPATH}" ${PY} deployment/model_server/server_policy.py \
    --ckpt_path ${CKPT} --port ${PORT} --cuda 0 > "${SERVER_LOG}" 2>&1 &
SERVER_PID=$!
trap "kill ${SERVER_PID} 2>/dev/null || true" EXIT

# wait for the websocket server to be ready (or die)
for i in $(seq 1 120); do
    grep -q "server listening" "${SERVER_LOG}" 2>/dev/null && { echo "server up."; break; }
    kill -0 ${SERVER_PID} 2>/dev/null || { echo "SERVER DIED:"; tail -20 "${SERVER_LOG}"; exit 1; }
    sleep 2
done

echo "=== eval: ${SUITE}, ${NUM_TRIALS} trials/task ==="
PYTHONPATH="${CLIENT_PYTHONPATH}" ${PY} examples/LIBERO/eval_libero.py \
    --args.pretrained-path ${CKPT} \
    --args.host 127.0.0.1 --args.port ${PORT} \
    --args.task-suite-name "${SUITE}" \
    --args.num-trials-per-task ${NUM_TRIALS} \
    --args.video-out-path "${OUT}" \
    --args.with_state "true" \
    --args.action-chunk-size ${CHUNK} \
    --args.seed 7 2>&1 | tee "${OUT}/eval.log"

echo "=== done. result: ${OUT}/eval.log ==="
