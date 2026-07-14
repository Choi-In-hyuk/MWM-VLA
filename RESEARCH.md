# VLA-DINO-Mamba: VLM-조건 Latent 월드모델 + 액션 생성

> VLA-JEPA를 확장한 연구. 잘 학습된 **VLM(Qwen)의 언어+시각 이해(의도)** 를 latent 월드모델에 전달해
> **미래 latent를 예측**하고, 그걸로 **액션을 생성**한다. 월드모델을 학습 보조가 아니라
> **추론 시점의 액션 생성기**로 쓴다.

---

## 1. 동기 (Motivation)

- **VLA-JEPA의 한계**: V-JEPA 월드모델이 학습 시 보조 loss로만 쓰이고 **추론 땐 버려진다**.
  (논문: *"At inference, the world model is dropped entirely."*)
- **이미지 예측 월드모델 + IDM은 느리다** (UniPi/AVDC/SuSIE: 픽셀 생성). → **latent 예측이면 빠르다**(JEPA).
- **raw 센서/state가 아니라, 잘 학습된 VLM의 의미론적 이해를 조건으로** 미래를 예측하면 어떨까 —
  Qwen의 "이 행동을 하면 세상이 이렇게 변한다"는 goal/intent를 월드모델에 주입.

**핵심 아이디어**: `현재 프레임 → DINO latent` + `Qwen 의도 토큰` → **Mamba가 미래 latent 예측** →
그 (현재, 미래) 상태로 **액션 생성**.

---

## 2. Thesis & Novelty

> **"잘 학습된 VLM(Qwen)의 언어+시각 의도를 조건으로 latent 월드모델이 미래 상태를 예측하고,
> 추론 시점에 그 예측으로 액션을 생성한다."** — *의미론적 VLM-조건 월드모델*.

차별점: ① 조건이 **raw state가 아니라 VLM의 semantic intent**, ② **추론 시점**에 WM 사용
(VLA-JEPA는 버림), ③ 미래 예측이 **단일 DINO latent 공간**(DreamVLA의 멀티속성 분해와 대비).

| 연구 | 추론시 WM | 액션 생성 | 미래 표현 |
|---|---|---|---|
| VLA-JEPA | ❌ 버림 | diffusion 헤드 | (학습용) V-JEPA tube |
| DreamVLA | ✅ | diffusion (WM 조건) | **멀티속성**(동적/깊이/의미) |
| V-JEPA 2-AC | ✅ | MPC (~16s/action) | V-JEPA latent |
| **본 연구** | ✅ | diffusion (s_0,s_end 조건) | **단일 DINO latent** |

> **coupled risk**(WM 예측 흔들리면 액션도): 본 연구도 WM 기반이라 risk 존재하지만,
> ① 단순·집중 WM(단일 DINO 빈칸 예측) ② Qwen의 좋은 intent 조건 ③ 헤드를 WM의 *실제 출력*으로 학습
> → **risk를 낮춤**(없애는 게 아님).

---

## 3. 결과 (Results) — LIBERO-10 (최난도)

| 모델 | 구조 | 성공률 |
|---|---|---|
| V1 | per-frame s₁..s₇ + 학습된 IDM | **20%** (4/20) |
| V2 | **끝점 s_end + diffusion 헤드** | **84.8%** (424/500, 50 trials) |
| V2 + Qwen LoRA | + Qwen LoRA(attn, r16) | **88.4%** (442/500, 50 trials) |
| #2 (VLA-WM) | + DINO를 JEPA식 학습(online+EMA+마스킹) | **84.8%** (424/500, 50 trials) |
| V2+LoRA (libero_all 4-suite 학습) | 동일 구조, 데이터만 4-suite | **91.2%** (456/500, 50 trials) |
| **Streaming v2 — 2-frame WM** *(best)* | + past(t−7) frame, no SSM carry | **93.2%** (466/500, 50 trials) |
| (참고) base VLA-JEPA | 원본 (full) | 95.8% |

> **#2 (DINO-JEPA) 결과 — 개선 없음.** 인코더가 예측 task엔 잘 적응(pred_cos 0.98,
> s0_std ~1.4 안정 = collapse 없음)했으나, **단일 시드에서 액션 성공률은 84.8%로 V2+LoRA(88.4)에 못 미침.**
> "표현 목적이 좋아져도 정책이 안 좋아지는" 전형. 진짜 회귀인지 시드 노이즈인지는 multi-seed 필요.

### Streaming v2 (2-frame world model) — 현재 베스트

> 자세한 보고: [`STREAM_V2_REPORT.md`](STREAM_V2_REPORT.md), 설계 진화 과정: [`DESIGN_STREAMING_WM.md`](DESIGN_STREAMING_WM.md).

핵심 변경 (V2 + LoRA 대비):
- **Predictor 입력에 과거 frame 추가**: 단일 현재 frame → `[past(t−7), present(t)]` 두 장 DINO 임베딩.
  per-frame `time_emb`로 두 frame을 구분.
- **SSM carry 안 함**: `states=None` 고정. 학습/추론 분포 완전 일치.
- **robot_state는 predictor에서 제거** (action_head에만 유지) — baseline과 동일.
- **cold start는 dataloader front-padding으로 자연 처리**: `obs_indices=[-7, 0, 7]`,
  base<7인 sample은 frame 0 복제 → 추론 t=0의 분포 그대로 학습.

학습 metric (libero_10 단독, 15k+15k step, DDP×2, eff bs 32):
- stage1: pred_cos = 0.9124
- stage2: pred_cos = 0.9242, action_loss = 0.0104

평가: **libero_10 50 trial/task = 93.2%**. baseline `v2lora_libero_all` (libero_10 91.2%) 대비 +2.0%p,
**같은 데이터**(libero_10 단독)·**같은 학습 budget**(15k+15k)에서 +2.0%p는 **predictor에 과거 frame 1장 추가**한 단일 변경 효과.

#### 폐기된 시도 (왜 안 했는가)

| 방향 | 시도 결과 | 폐기 이유 |
|---|---|---|
| SSM hidden state를 episode 전체에 걸쳐 carry | 0% (3 trial) | 학습 시 carry가 sample 안 2 chunk만 발생 → 추론 분포(수십 chunk 누적)와 분포 불일치. carry 끄면(83%) toggle(67%)보다 잘 됨. |
| Truncated-BPTT episode streaming (Stage-2) | pred_cos 0.938 → 0.902로 회귀 | random-window보다 학습 다양성 부족 + 누적 state로 gradient 불안정 |
| Predictor에 robot_state token 추가 | shape/dtype 미스매치 디버깅 비용 | action head에 이미 robot_state 들어가 중복. predictor는 시각만 집중하는 게 단순. |
| ckpt 합치기 (학습 중단 후 final.pt 수동 재구성) | Qwen base weight이 원본과 다름 (98% mismatch) | strict=False 로딩 시 key prefix 미스매치를 silent하게 무시. 해결: 학습 끝까지 돌려서 `save("final", full=True)`가 정상 실행되게 함. |

### 진행 중 / 추가 실험

- **V3 (시간-버퍼 월드모델)** — `VLA_DINO_Mamba_Temporal`: 과거 7프레임 latent를 Mamba에 시간 시퀀스로
  넣어 동역학(모션) 인코딩, 끝점 s_{t+7} 예측 유지. 추론은 매 스텝 DINO 임베딩→연속 버퍼(cold-start는
  음수 delta 자동 첫프레임 패딩). **V3-a(frozen DINO), V3-b(JEPA-DINO)** 학습 완료했으나 **action loss가
  0.028/0.045로 best(0.0115)보다 2.4~3.9배 높아** 학습 loss 기준 기각(eval 생략). 실패 원인이 동역학이
  아니라 추론/진행상황(웹조사: LaRA/PALM)이라는 해석과 일치.
- **A (학습량↑, MambaVLA 통찰: 간단한 구조 + 많은 학습)** — `run_A_longtrain.sh` / `run_A_b_liberoall.sh`:
  - **A-a**: V2+LoRA 구조 동일, **step 15k→40k** (libero_10, DINO frozen). *진행 중* — stage1 pred_cos
    0.878→**0.905**, stage2 action loss **0.0087 < best 0.0115** (학습 loss로 best 갱신, eval 대기).
  - **A-b**: + DINO 인코더 학습(action-anchored + 타깃 detach → EMA 없이 collapse-free), **libero_all
    4-suite, 60k** (인코더 학습엔 데이터 다양성 필요). *큐 대기*.

> 베스트는 (eval 기준) 여전히 **V2+LoRA (88.4%)**. A-a가 학습 loss로 best를 갱신 → "학습량이 병목"
> 가설을 지지(eval로 확정 예정). 그리퍼는 **바이너리(0/1)**, 추론 시 0.5 임계로 이진화.

- **V1 20% → V2 84.8%**: per-frame "작은 변화" 문제(50ms DINO latent 거의 동일 → 액션 효과 묻힘)를
  **끝점 예측(큰 변화) + diffusion 헤드(표현력)** 로 해결. ← 핵심 도약.
- **V2 84.8% → +LoRA 88.4%** (+3.6%, borderline 유의 z≈1.7): action 토큰이 V-JEPA-튭 WM용으로
  학습됐던 걸 **Qwen LoRA로 Mamba/DINO에 재최적화** → modest 개선. 가설 지지.
- **88.4% = base(90%+)의 ~2% 이내** → **경량화 + 완전히 새로운 구조로 경쟁력 확보** (contribution의 핵심).

### 메트릭 주의 (교훈)
- `pred_cos`(예측 latent의 코사인)는 **정지 장면에 압도**되는 약한 지표 — frame0→끝점도 대부분 정지라
  cos가 높아도 "변화(액션 관련) 예측"을 못 잼. **판단은 action loss + LIBERO 성공률로.**

### 실패 모드 (V2/LoRA 공통)
- **멀티오브젝트/멀티스텝**에서 실패 집중 (예: "put **both** moka pots"). 첫 물체는 잘 잡고
  **두 번째 물체 grasp**(정밀도·시퀀싱)에서 실패. 정책은 합리적(이상한 액션·붕괴 없음).
- 실행 청크↓(K=3, 추론만)은 도움 안 됨(75%) → open-loop drift가 주범 아님 = grasp 정밀도/표현 한계.

---

## 4. V1 아키텍처 (baseline) + 결과 분석

```
현재프레임 → DINO(frozen) → s_0
언어+현재 → Qwen(frozen) → action 토큰
(s_0, action토큰) → Mamba 예측기 → s_1..s_7 (per-frame)
(s_k, s_{k+1}, state) → IDM 헤드 → a_k   (7 transitions = 7 actions)
```
- 학습: Stage1 예측기(L_pred) → Stage2 IDM(L_idm + L_consist).
- 결과: pred_cos 0.86, idm/consist ~0.02 (RMS ~0.14) → **LIBERO-10 20%**.
- **왜 20%인가**:
  - **(A) per-frame 작은 변화** → ID(s_t,s_{t+1}) 애매 → 액션 14% 오차 → 정밀 조작 실패.
  - **(B) DINO는 semantic이라 정밀 metric 액션을 덜 담음** (robot state로 보강 필요).
  - **(C) 간접경로 오차 누적** + open-loop 청크 drift.

---

## 5. V2 아키텍처 (메인 방향) ⭐

```
┌──────────────────────────────────────────────────────────────────┐
│                                                                    │
│   [현재 프레임] ── DINO (frozen) ─────────────────────→ s_0         │
│                                                          │         │
│   [언어 + 현재] ── Qwen-VL (frozen) ──→ action 토큰      │         │
│                    ("어떻게 움직여 latent가 어떻게 변하나") │         │
│                                          │               │         │
│                          ┌───────────────▼───────────────▼──┐      │
│   (s_0 + action 토큰) ──→│  Mamba 예측기 (학습)              │      │
│                          │  → s_end (현재+H프레임, 청크 끝)   │      │
│                          └───────────────┬──────────────────┘      │
│   [robot state(8D)] ──────────────────┐  │                         │
│                          ┌────────────▼──▼──────────────────┐      │
│   조건=(s_0, s_end, state) ─→│ diffusion 헤드 (학습)        │      │
│                          │  flow-matching → 액션 청크 (H개)  │      │
│                          └───────────────┬──────────────────┘      │
│                                  액션 a_0..a_{H-1} ◀┘               │
└──────────────────────────────────────────────────────────────────┘
```

### 추론 흐름 (open-loop 청킹)
```
청크0: frame 0 관측 → s_end=예측(frame H) → diffusion → 액션 H개 실행 → frame H 도달
청크1: frame H 관측 → s_end=예측(frame 2H) → ...
```
**s_end = 현재 + H프레임 = 그 청크 *하나*의 끝** (에피소드 끝 아님). 매 청크 새로 예측.

### 설계 결정 (확정)
| 결정 | 선택 | 근거 |
|---|---|---|
| 미래 예측 단위 | **끝점 s_end** (현재+H, Qwen 호출주기) | 변화가 커서 예측·생성 명확 ((A) 해소) |
| Qwen 조건 | **action 토큰** | "어떻게 변하나"를 담도록 학습된 토큰 |
| 헤드 조건 | **(s_0, s_end, robot state)** | 액션=현재→미래 *전이*라 s_0 필수; state는 metric 보강 ((B)) |
| robot state | 8D proprioception (관절+그리퍼), MLP→토큰, **조건 자리** | DINO가 못 담는 관절 kinematics 보강 |
| 헤드 구조 | 원본 `FlowmatchingActionHead` 재사용, 조건만 교체 | 검증된 코드 |

---

## 6. 학습 (V2, staged)

```
Stage 1: Mamba 예측기 학습
         L_pred = Mamba(s_0, action토큰) ≈ DINO(끝점 프레임)

Stage 2: 예측기 fine-tune(완전 frozen ❌) + diffusion 헤드 학습
         L = α·L_pred + β·L_action
         L_action = flow-matching, 조건 (s_0, *예측* s_end, state) → GT 액션
         (헤드는 예측 s_end로 학습 → 추론과 일치)
```
- **완전 frozen 안 함**: 예측기가 *액션에 유용한* latent로 적응하되, **L_pred로 DINO 실제 미래에 anchor**(drift/shortcut 방지).
- **비율(α,β)은 Stage 1 cos 보고 조정**: cos 높으면 L_pred 약하게(약한 fine-tune), 어중간하면 L_pred 강하게.

---

## 7. 평가

- **핵심**: LIBERO-10(최난도)에서 성공률. ("어려운 게 되면 쉬운 건 된다" — dev는 libero_10 집중, 논문 최종본엔 전 suite.)
- **비교군**: 원본 diffusion 헤드(VLA-JEPA) baseline.
- eval 비결정적(action 샘플링 unseeded) → 다중 시드.
- **eval 셋업(작동 확인)**: `scripts/eval_libero_dino.sh <suite> <trials> <port> <ckpt> <tag>`.
  env: `LIBERO_HOME=~/LIBERO-PRO`, `LIBERO_CONFIG_PATH=$LIBERO_HOME/libero`,
  `PYTHONPATH=$LIBERO_HOME:~/VLA-JEPA`, `MUJOCO_GL=egl`. (server_policy + eval_libero.py)

---

## 8. 구현 (파일)

| 파일 | 역할 |
|---|---|
| `starVLA/model/framework/VLA_DINO_Mamba.py` | V1 (DINO+Qwen+Mamba+IDM) — baseline |
| **`VLA_DINO_Mamba_Diff.py`** (예정) | **V2 (끝점 예측 + diffusion 헤드)** |
| `starVLA/model/modules/world_model/mamba_world_model.py` | MambaStatePredictor, InverseDynamicsHead |
| `starVLA/model/modules/action_model/GR00T_ActionHeader.py` | FlowmatchingActionHead (재사용) |
| `starVLA/model/modules/dino_model/dino.py` | DINOv2 wrapper |
| `scripts/train_mamba_wm.py` | 트레이너 (staged) |
| `scripts/precompute_qwen_cache.py` | Qwen 토큰 캐시 (학습 가속) |
| `scripts/eval_libero_dino.sh` | LIBERO eval |

- 백본 ckpt: `/home/choi/data/checkpoints/VLA-JEPA/LIBERO/checkpoints/VLA-JEPA-LIBERO.pt`
- 데이터: `/home/choi/data/datasets/LIBERO` (libero_10: 379 eps, 20Hz, 2뷰)
- Qwen 캐시: `results/cache/qwen_libero10` (~10GB)

---

## 9. 상태 & TODO

- [x] V1 구현·학습·검증 (LIBERO-10 **20%** — per-frame 한계 확인)
- [x] novelty 문헌 검증, eval 셋업 작동
- [x] **V2 구현·학습·검증** (끝점 + diffusion) → **84.8%** (50 trials)
- [x] **Qwen LoRA** (양 stage) → **88.4%** (+3.6%, borderline 유의)
- [x] **V2+LoRA, 4-suite (libero_all) 학습** → libero_10 **91.2%** (50 trials)
- [x] **#2 (VLA-WM, JEPA-DINO)** → 84.8% (개선 없음, 본문 §3 참조)
- [x] **Streaming v1 (SSM hidden carry)** — 시도 후 폐기 (carry 켜면 0% / 끄면 83%, RESEARCH §3 표 참조)
- [x] **Streaming v2 (2-frame WM, no carry)** → libero_10 **93.2%** (50 trials) — 현재 베스트
- [ ] Streaming v2를 4-suite (libero_all)로 학습 → 4-suite eval
- [ ] LIBERO-Plus eval (perturbation robustness) — assets 설치 완료, 패키지 설치 보류
- [ ] (옵션) multi-seed로 유의성 확정
- [ ] 논문 최종: 전 suite 평가 + 지연시간(latency)

### 산출물(스크립트)
- `scripts/train_mamba_wm.py` (`--stage predictor|stage2`, `--qwen_lora`, `--qwen_cache`)
- `scripts/run_2stage_dino.sh` / `run_2stage_dino_lora.sh` (2-stage 체인)
- `scripts/precompute_qwen_cache.py` (Qwen 토큰 캐시)
- `scripts/eval_libero_dino.sh` (`<suite> <trials> <port> <ckpt> <tag> <chunk>`)
- `scripts/eval_compare_50.sh` (non-LoRA vs LoRA 50-trial)

---

## 참고 문헌
- VLA-JEPA — arXiv:2602.10098 (베이스)
- V-JEPA 2 / 2-AC — arXiv:2506.09985
- DreamVLA — arXiv:2507.04447
- LAWM (Latent Action Pretraining) — arXiv:2509.18428
- RoboMamba — arXiv:2406.04339
- World Model for Robot Learning: A Survey — arXiv:2605.00080
