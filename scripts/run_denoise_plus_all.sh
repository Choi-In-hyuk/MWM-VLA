#!/bin/bash
# denoise 통합모델의 LIBERO-Plus 7축 평가 (4 suite 전부).
#
# 바깥 루프 = category(축), 안쪽 루프 = 4 suite.
# Sensor Noise 가 제일 오래 걸리므로 축 순서에서 맨 마지막에 둔다 -> 무거운 부분이
# 전부 뒤로 몰려, 앞쪽 6축 결과가 먼저 다 쌓인다.
#
# 결과: results/eval_plus/<suite>_denoise_all/results.jsonl (resume 지원: 축/suite
#       중간에 끊겨도 이미 끝난 task_index 는 건너뜀).
# 집계: 각 suite 폴더의 results.jsonl 을 category별로 합산 (스크립트 summarize).
#
# 단일 GPU. server_policy(vla_jepa, 모델) + eval_libero_plus.py(vla_plus, plus env).
# Usage: bash scripts/run_denoise_plus_all.sh
set -uo pipefail
cd "$(dirname "$0")/.."

CK=results/denoise_all/stage2/checkpoints/mamba_wm_final.pt
PY_TRAIN=/home/choi/miniconda3/envs/vla_jepa/bin/python
PY_PLUS=/home/choi/miniconda3/envs/vla_plus/bin/python

SUITES=(libero_spatial libero_object libero_goal libero_10)
# 축 순서: 가벼운/중간 6축 먼저, Sensor Noise 맨 마지막
CATEGORIES=(
  "Background Textures"
  "Camera Viewpoints"
  "Language Instructions"
  "Light Conditions"
  "Objects Layout"
  "Robot Initial States"
  "Sensor Noise"
)

export LIBERO_HOME=/home/choi/LIBERO-plus
export LIBERO_CONFIG_PATH=${LIBERO_HOME}/libero
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1

[[ -f "$CK" ]] || { echo "!!! ckpt 없음: $CK"; exit 1; }

port=18600
run_one () {   # $1=suite  $2=category
  local SUITE=$1 CAT=$2
  local EOUT=results/eval_plus/${SUITE}_denoise_all
  local SLOG=/tmp/denoise_plus_${port}.log
  mkdir -p "${EOUT}"

  echo ""
  echo "########## [${CAT}] ${SUITE} (port=${port}) ##########"
  PYTHONPATH="$(pwd)" ${PY_TRAIN} deployment/model_server/server_policy.py \
      --ckpt_path ${CK} --port ${port} --cuda 0 > "${SLOG}" 2>&1 &
  local SPID=$!
  local up=0
  for i in $(seq 1 120); do
    grep -q "server listening" "${SLOG}" 2>/dev/null && { up=1; break; }
    kill -0 ${SPID} 2>/dev/null || { echo "SERVER DIED:"; tail -20 "${SLOG}"; break; }
    sleep 2
  done
  if [[ $up -ne 1 ]]; then echo "!!! [${CAT}/${SUITE}] server 안 뜸 — skip"; kill ${SPID} 2>/dev/null; port=$((port+1)); return 1; fi

  PYTHONPATH="${LIBERO_HOME}:$(pwd)" ${PY_PLUS} examples/LIBERO/eval_libero_plus.py \
      --args.pretrained-path ${CK} --args.host 127.0.0.1 --args.port ${port} \
      --args.task-suite-name "${SUITE}" --args.out-dir "${EOUT}" \
      --args.only-category "${CAT}" \
      --args.num-trials-per-task 1 --args.with-state true \
      --args.action-chunk-size 0 --args.seed 7 \
      >> "${EOUT}/eval.log" 2>&1 || echo "!!! [${CAT}/${SUITE}] eval rc=$?"

  kill ${SPID} 2>/dev/null || true; wait ${SPID} 2>/dev/null || true
  port=$((port+1))
}

for CAT in "${CATEGORIES[@]}"; do
  for SUITE in "${SUITES[@]}"; do
    run_one "${SUITE}" "${CAT}"
  done
done

echo ""
echo "=================================================================="
echo "[PLUS DONE] denoise 통합모델 LIBERO-Plus 7축 x 4 suite 완료."
echo "  suite별 결과: results/eval_plus/<suite>_denoise_all/results.jsonl"
echo "  집계 보기:    각 폴더 eval.log 의 SUMMARY, 또는 아래 한 줄로 전체 합산"
echo "=================================================================="
# 전체 합산 (suite별 category rate + 전체)
${PY_TRAIN} - <<'PY'
import json, glob, collections
rows = collections.defaultdict(lambda: [0,0])  # (suite,cat)->[succ,tot]
for f in glob.glob("results/eval_plus/*_denoise_all/results.jsonl"):
    suite = f.split("/")[-2].replace("_denoise_all","")
    for line in open(f):
        try: r = json.loads(line)
        except: continue
        k = (suite, r["category"]); rows[k][1]+=1
        if r["success"]: rows[k][0]+=1
suites = sorted(set(k[0] for k in rows))
cats = sorted(set(k[1] for k in rows))
print(f"{'category':<24}" + "".join(f"{s.replace('libero_',''):>10}" for s in suites) + f"{'AVG':>10}")
for c in cats:
    line=f"{c:<24}"; ss=tt=0
    for s in suites:
        a,b=rows[(s,c)]; ss+=a; tt+=b
        line+=f"{(a/b*100 if b else 0):>9.1f}%"
    line+=f"{(ss/tt*100 if tt else 0):>9.1f}%"
    print(line)
tot=[sum(rows[k][0] for k in rows), sum(rows[k][1] for k in rows)]
print(f"{'-- TOTAL --':<24}{'':>{10*len(suites)}}{(tot[0]/tot[1]*100 if tot[1] else 0):>9.1f}%")
PY
