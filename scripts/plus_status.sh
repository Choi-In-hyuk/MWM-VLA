#!/bin/bash
# denoise LIBERO-Plus 평가 실시간 상태: 현재 조합 + category별/전체 누적 성공률.
# 실시간: watch -n5 bash scripts/plus_status.sh
cd "$(dirname "$0")/.."

echo "=== 현재 진행 조합 ==="
grep '##########' results/denoise_plus_run.log 2>/dev/null | tail -1 | sed 's/#//g'
pgrep -f eval_libero_plus.py >/dev/null && echo "  [eval 실행 중]" || echo "  [eval 프로세스 없음]"
echo ""

/home/choi/miniconda3/envs/vla_jepa/bin/python - <<'PY'
import json, glob, collections
rows = collections.defaultdict(lambda:[0,0])   # key -> [succ, total]
per_suite = collections.defaultdict(lambda:[0,0])
for f in glob.glob("results/eval_plus/*_denoise_all/results.jsonl"):
    suite = f.split("/")[-2].replace("_denoise_all","").replace("libero_","")
    for l in open(f):
        try: r = json.loads(l)
        except: continue
        c = r["category"]; ok = bool(r["success"])
        for k in [("ALL",c),(suite,c)]:
            rows[k][1]+=1
            if ok: rows[k][0]+=1
        per_suite[suite][1]+=1
        if ok: per_suite[suite][0]+=1

if not rows:
    print("아직 결과 없음.")
else:
    suites = sorted(per_suite)                       # 실제로 데이터가 있는 suite
    cats   = sorted({k[1] for k in rows if k[0]=="ALL"})
    W = 16
    hdr = f"{'category':<22}" + "".join(f"{s:>{W}}" for s in suites) + f"{'ALL':>{W}}"
    print(hdr); print("-"*len(hdr))
    for c in cats:
        line = f"{c:<22}"
        for s in suites:
            a,b = rows[(s,c)]
            line += (f"{a}/{b}={a/b*100:.0f}%" if b else "-").rjust(W)
        a,b = rows[("ALL",c)]
        line += (f"{a}/{b}={a/b*100:.1f}%").rjust(W)
        print(line)
    print("-"*len(hdr))
    line = f"{'TOTAL':<22}"
    for s in suites:
        a,b = per_suite[s]
        line += (f"{a}/{b}={a/b*100:.0f}%" if b else "-").rjust(W)
    tot = [sum(v[0] for v in per_suite.values()), sum(v[1] for v in per_suite.values())]
    line += (f"{tot[0]}/{tot[1]}={tot[0]/tot[1]*100:.1f}%").rjust(W)
    print(line)
PY
