#!/bin/bash
# CALVIN 5-task chain (long-horizon) evaluation.
#
# Starts our policy server (vla_jepa env) on <PORT> and runs the CALVIN
# evaluate_policy loop from the calvin_env conda environment (which has
# pybullet + hydra + the CALVIN sim).
#
# Usage:
#   bash scripts/eval_calvin_LH.sh <ckpt> <tag> <dataset_dir> [num_sequences]
#
# Example:
#   bash scripts/eval_calvin_LH.sh \
#       results/calvin_D/stage2/checkpoints/mamba_wm_final.pt \
#       calvin_D_v1 \
#       /mnt/4TB_2/jamvla/datasets/calvin/task_D_D \
#       1000
set -uo pipefail
cd "$(dirname "$0")/.."

CKPT=${1:?usage: eval_calvin_LH.sh <ckpt> <tag> <dataset_dir> [num_sequences]}
TAG=${2:?usage: eval_calvin_LH.sh <ckpt> <tag> <dataset_dir> [num_sequences]}
DATASET_DIR=${3:?usage: eval_calvin_LH.sh <ckpt> <tag> <dataset_dir> [num_sequences]}
NUM_SEQ=${4:-1000}

PORT=20080
OUT=results/eval_calvin/${TAG}
mkdir -p ${OUT}

SERVER_PY=/home/choi/miniconda3/envs/vla_jepa/bin/python
CALVIN_PY=/home/choi/miniconda3/envs/calvin_env/bin/python

export PYTHONUNBUFFERED=1
export MUJOCO_GL=egl

# server-side (training env)
SERVER_PYTHONPATH="$(pwd):${PYTHONPATH:-}"
# client-side (needs CALVIN sim + our repo code)
CLIENT_PYTHONPATH="/home/choi/calvin/calvin_models:/home/choi/calvin/calvin_env:$(pwd):${PYTHONPATH:-}"

echo "=== starting model server (port ${PORT}) ==="
SERVER_LOG=${OUT}/server.log
PYTHONPATH="${SERVER_PYTHONPATH}" ${SERVER_PY} deployment/model_server/server_policy.py \
    --ckpt_path "${CKPT}" --port ${PORT} --cuda 0 > "${SERVER_LOG}" 2>&1 &
SERVER_PID=$!
trap "kill ${SERVER_PID} 2>/dev/null || true" EXIT

# wait for server ready
for i in $(seq 1 120); do
    grep -q "server listening" "${SERVER_LOG}" 2>/dev/null && { echo "server up."; break; }
    kill -0 ${SERVER_PID} 2>/dev/null || { echo "SERVER DIED:"; tail -20 "${SERVER_LOG}"; exit 1; }
    sleep 2
done

echo "=== eval: CALVIN chain, ${NUM_SEQ} sequences ==="
PYTHONPATH="${CLIENT_PYTHONPATH}" ${CALVIN_PY} examples/CALVIN/eval_calvin.py \
    --dataset_path "${DATASET_DIR}" \
    --host 127.0.0.1 --port ${PORT} \
    --num_sequences ${NUM_SEQ} \
    --log_dir "${OUT}" 2>&1 | tee "${OUT}/eval.log"

echo "=== done. log: ${OUT}/eval.log ==="
