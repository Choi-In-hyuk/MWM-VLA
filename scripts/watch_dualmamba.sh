#!/bin/bash
# Convenience tail viewer for run_2stage_dualmamba.sh background runs.
#
# Usage:
#   bash scripts/watch_dualmamba.sh             # follow current active stage log
#   bash scripts/watch_dualmamba.sh status      # one-shot: latest line per stage + GPU
#   bash scripts/watch_dualmamba.sh stage1      # follow stage 1 log
#   bash scripts/watch_dualmamba.sh stage2      # follow stage 2 log
#   bash scripts/watch_dualmamba.sh run         # follow the top-level run.log
#
# Each training log line already contains:
#   [stage] step/max (pct%) ep<N> | <losses> | lr=... | <it/s> | ETA <h>h
# so a plain `tail -f` gives you step / percent / epoch / it/s / ETA at a glance.
set -eo pipefail
cd "$(dirname "$0")/.."

OUT=results/dual_mamba_libero_10
S1=${OUT}/pred/train.log
S2=${OUT}/stage2/train.log
RUN=${OUT}/run.log
MODE=${1:-auto}

last_line() { [ -f "$1" ] && tail -n 1 "$1" 2>/dev/null || echo "(no log yet)"; }

show_status() {
  echo "=== status @ $(date '+%F %T') ==="
  echo "[run.log]    $(last_line $RUN)"
  echo "[stage1]     $(last_line $S1)"
  echo "[stage2]     $(last_line $S2)"
  echo
  echo "--- GPUs ---"
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total \
    --format=csv,noheader 2>/dev/null | sed 's/^/  /'
  echo
  echo "--- last checkpoints ---"
  ls -lt ${OUT}/pred/checkpoints/ 2>/dev/null | head -4 | sed 's/^/  /'
  ls -lt ${OUT}/stage2/checkpoints/ 2>/dev/null | head -4 | sed 's/^/  /'
  echo
  echo "--- running procs ---"
  pgrep -af "train_mamba_wm|torchrun.*train_mamba" 2>/dev/null | head -5 || echo "  (none)"
}

case "$MODE" in
  status) show_status ;;
  stage1) tail -f $S1 ;;
  stage2) tail -f $S2 ;;
  run)    tail -f $RUN ;;
  auto)
    if [ -f $S2 ] && [ "$(wc -l <$S2 2>/dev/null || echo 0)" -gt 0 ]; then
      echo "[watch_dualmamba] following stage 2"
      tail -f $S2
    elif [ -f $S1 ]; then
      echo "[watch_dualmamba] following stage 1"
      tail -f $S1
    else
      echo "[watch_dualmamba] no train.log yet, following run.log"
      tail -f $RUN
    fi
    ;;
  *) echo "unknown mode: $MODE"; echo "use: status|stage1|stage2|run|auto"; exit 1 ;;
esac
