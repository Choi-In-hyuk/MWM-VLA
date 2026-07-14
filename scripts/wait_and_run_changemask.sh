#!/bin/bash
# Poll GPU memory; once BOTH GPUs have >= MIN_FREE_GB free, launch the full
# change-mask pipeline (stage1 -> stage2 -> eval).
#
# Designed to be safe to leave running unattended: never preempts other users,
# only starts when there's headroom for our bs=16/GPU run (~36 GB/GPU during
# stage2 + safety margin).
#
# Usage: nohup bash scripts/wait_and_run_changemask.sh > results/changemask_v2lora_libero_10/watcher.log 2>&1 &
set -eo pipefail
cd "$(dirname "$0")/.."

MIN_FREE_GB=${MIN_FREE_GB:-50}   # per-GPU free memory threshold
POLL_SEC=${POLL_SEC:-60}
NEED_BOTH_GPUS=${NEED_BOTH_GPUS:-1}

OUT=results/changemask_v2lora_libero_10
mkdir -p ${OUT}

echo "[watcher] start $(date -Iseconds); threshold ${MIN_FREE_GB} GB per GPU, poll ${POLL_SEC}s"

while true; do
    # nvidia-smi: free memory in MiB
    readarray -t FREE_MB < <(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
    G0=${FREE_MB[0]:-0}
    G1=${FREE_MB[1]:-0}
    G0_GB=$(( G0 / 1024 ))
    G1_GB=$(( G1 / 1024 ))
    NEED_MB=$(( MIN_FREE_GB * 1024 ))

    if [[ $G0 -ge $NEED_MB && $G1 -ge $NEED_MB ]]; then
        echo "[watcher] $(date -Iseconds): GPU0 ${G0_GB} GB free, GPU1 ${G1_GB} GB free -> launching"
        break
    fi
    echo "[watcher] $(date -Iseconds): GPU0 ${G0_GB} GB / GPU1 ${G1_GB} GB free (need ${MIN_FREE_GB} GB) -- waiting"
    sleep ${POLL_SEC}
done

echo "[watcher] launching full pipeline at $(date -Iseconds)"
bash scripts/run_changemask_v2lora.sh all > ${OUT}/run_all.log 2>&1
echo "[watcher] pipeline returned exit=$? at $(date -Iseconds)"
