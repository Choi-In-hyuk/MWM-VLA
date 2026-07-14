#!/bin/bash
# Wall-clock latency benchmark for predict_action.
#
# Reports mean / median / p90 / p99 inference time (seconds) of a SINGLE
# predict_action call (one re-plan = 7 actions).
#
# Usage:
#   bash scripts/bench_latency.sh <ckpt> [n_warmup] [n_iters] [cuda_idx]
# Defaults:
#   n_warmup=10  n_iters=100  cuda_idx=0
set -uo pipefail
cd "$(dirname "$0")/.."

CKPT=${1:?usage: bench_latency.sh <ckpt> [n_warmup] [n_iters] [cuda_idx]}
N_WARMUP=${2:-10}
N_ITERS=${3:-100}
CUDA_IDX=${4:-0}
PY=/home/choi/miniconda3/envs/vla_jepa/bin/python

export PYTHONUNBUFFERED=1
export PYTHONPATH=$(pwd)
export CUDA_VISIBLE_DEVICES=${CUDA_IDX}

${PY} - <<PYEOF
import time, statistics
import numpy as np
import torch
from PIL import Image
from starVLA.model.framework.base_framework import baseframework

CKPT = "${CKPT}"
N_WARMUP = ${N_WARMUP}
N_ITERS  = ${N_ITERS}

print(f"[bench] loading {CKPT}")
model = baseframework.from_pretrained(CKPT)
device = torch.device("cuda:0")
model = model.to(device).eval()
print(f"[bench] model on {device}")

# dummy batch (single sample, 2 views)
rng = np.random.RandomState(0)
def make_imgs():
    return [[Image.fromarray(rng.randint(0, 255, (256, 256, 3), dtype=np.uint8))
             for _ in range(2)]]
state = rng.randn(1, 8).astype(np.float32)
inst = ["pick up the block"]

# warmup
print(f"[bench] warmup {N_WARMUP} iters...")
if hasattr(model, "episode_reset"):
    model.episode_reset()
for _ in range(N_WARMUP):
    _ = model.predict_action(batch_images=make_imgs(), instructions=inst, state=state)
torch.cuda.synchronize()

# timed
print(f"[bench] timing {N_ITERS} iters...")
times = []
if hasattr(model, "episode_reset"):
    model.episode_reset()
for i in range(N_ITERS):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    _ = model.predict_action(batch_images=make_imgs(), instructions=inst, state=state)
    torch.cuda.synchronize()
    times.append(time.perf_counter() - t0)

times_ms = sorted(t * 1000.0 for t in times)
n = len(times_ms)
mean = statistics.mean(times_ms)
median = statistics.median(times_ms)
p90 = times_ms[int(0.9 * n)]
p99 = times_ms[min(n-1, int(0.99 * n))]
mn, mx = times_ms[0], times_ms[-1]

print()
print(f"=== predict_action latency (single re-plan = 7 actions) ===")
print(f"  n_iters = {n}")
print(f"  mean    = {mean:8.2f} ms   ({1000.0/mean:6.1f} Hz max re-plan rate)")
print(f"  median  = {median:8.2f} ms")
print(f"  p90     = {p90:8.2f} ms")
print(f"  p99     = {p99:8.2f} ms")
print(f"  min/max = {mn:.2f} / {mx:.2f} ms")
print()
# action_horizon=7. At 20Hz sim (50ms/step) we need re-plan within 7*50=350ms.
print(f"  budget at 20 Hz sim (7-step chunk) = 350 ms")
ok = (p99 <= 350.0)
print(f"  p99 {'<=' if ok else '>'} 350 ms  -> {'REAL-TIME OK' if ok else 'TOO SLOW'}")
PYEOF
