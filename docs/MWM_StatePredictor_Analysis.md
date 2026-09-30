# MWM State Predictor — 전체 실험 정리 및 분석

> 학위논문 작성용 정리 문서.
> 주제: **VLA 에서 World Model 로 예측한 미래 상태의 활용 방식 분석**.
> 벤치마크: LIBERO (4-suite × 500 rollout).

## 0. Executive Summary

Mamba World Model 이 예측하는 **미래 로봇 상태 (r_target)** 를 flow-matching action head 가 활용하는 방식을 네 가지 범주로 나누어 체계적으로 실험. 모든 변형이 baseline (94.0%) ± 0.5% 노이즈 범위에 머무름. 원인 분석과 향후 방향 제시.

**핵심 관찰**:
- 8-dim 로봇 상태 조건이 196×768-dim DINO latent 조건과 동등한 성능 (94.0% vs 94.2%).
- 조건 강화(cross-attn 추가), 학습 loss 강화(auxiliary/feedback), 추론 시 guidance (velocity residual, best-of-N rerank) 어느 방식도 유의미한 개선을 주지 못함.
- LIBERO 는 이미 saturation. 남은 실패 (~6%) 는 physics/data 노이즈 범주이며 conditioning 이 아닌 다른 축의 개선이 필요함.

---

## 1. Baseline 아키텍처

### 1.1 VLA-JEPA base
```
images + instruction ──► Qwen-VL ──► action_tokens ──► Flow-matching Head ──► actions
```
Qwen-VL 이 flow-matching head 에 직접 조건 제공. Predictor 없음. LIBERO base ckpt (`VLA-JEPA-LIBERO.pt`) 가 여기서 학습됨.

### 1.2 MWM-v2-FO (DINO 예측 계열)
```
Qwen action_tokens ─►┐
                     │
                     ├─► Mamba ─► DINO_target_pred  ─cond_proj─► head 조건
                     │            [B, 196, 768]
DINO(present) ──────►┘
```
Mamba 가 DINO latent 을 예측. Head 는 예측된 미래 시각 특징을 조건으로 봄.  
**LIBERO 평균 94.2%**.

### 1.3 MWM StatePredictor (본 실험의 baseline)
```
Qwen action_tokens ─►┐
                     │
                     ├─► Mamba ─► r_target_pred  ─cond_proj─► head 조건
                     │            [B, 8]                       [B, 8, qwen_dim]
[r_past, r_present]─►┘
```
DINO latent 대신 **8-dim 로봇 상태** 를 예측. Predictor per-dim tokenization (`tokens_per_frame=8, dino_dim=1`). Cond_proj MLP 로 `[r_present, r_target_pred, δ]` (24-dim) 를 K=8 토큰 × qwen_dim=2048 로 확장.

**LIBERO 평균 94.0%** — DINO 방식과 사실상 동등.

### 1.4 첫 번째 강한 발견

| 조건 | 파라미터 규모 | LIBERO 평균 |
|---|---|---|
| DINO latent 예측 | 196 tok × 768 = 150K numbers | 94.2% |
| **8-dim state 예측** | **8 numbers** | **94.0%** |

**Head 가 필요한 조건 정보는 8-dim state (EE pose + gripper) 로 충분**. 시각 특징 대부분은 이미 Qwen action_tokens → Mamba 경로에서 소비되고, head 에겐 "action-relevant subspace 의 goal 좌표" 만 있으면 됨.

---

## 2. 활용 방식 분류

예측된 `r_target_pred` 를 head 가 사용하는 방식을 네 가지 범주로 분류:

| 범주 | 어디에 반영 | 대표 변형 |
|---|---|---|
| A. Passive conditioning | Cross-attn 조건 토큰 | Baseline, QOnly |
| B. Auxiliary training loss | 학습 loss 항 | Aux, Feedback |
| C. Denoising-time steering | Denoising 스텝 velocity 편향 | ErrorDenoise, LyapDenoise, QLD |
| D. Test-time filtering | 후보 action 랭킹 | Rerank |

이 네 범주가 우리가 시도한 모든 방식을 커버.

---

## 3. 각 변형의 아키텍처와 결과

각 변형은 baseline StatePredictor 를 확장. **모두 baseline stage1 ckpt (predictor) 를 재사용하고 stage2 만 재학습** (30k step, 2×GPU DDP, LIBERO 4-suite 데이터).

### 3.1 [Baseline] `VLA_MWM_StatePredictor`
- 파일: [starVLA/model/framework/VLA_MWM_StatePredictor.py](../starVLA/model/framework/VLA_MWM_StatePredictor.py)
- Head 조건: state_cond 만 (K=8 토큰).

### 3.2 [범주 B] `VLA_MWM_StatePredictor_AuxHead`
- 파일: [starVLA/model/framework/VLA_MWM_StatePredictor_AuxHead.py](../starVLA/model/framework/VLA_MWM_StatePredictor_AuxHead.py)
- 추가: 1-step consistency 로 `a_hat = a_τ + (1-t)·v_pred` 재구성 → 작은 MLP 로 state delta 예측 → GT delta 로 supervise.
- Loss: `L_total = L_pred + L_action + λ_aux · L_aux` (λ=0.1)
- 의도: Head 가 뽑는 action 이 물리적으로 어디로 갈지 학습 pressure 부여.

### 3.3 [범주 B] `VLA_MWM_StatePredictor_Feedback`
- 파일: [starVLA/model/framework/VLA_MWM_StatePredictor_Feedback.py](../starVLA/model/framework/VLA_MWM_StatePredictor_Feedback.py)
- 추가: `r_hat = r_present + Linear(Σ a_hat)`, `L_feedback = MSE(r_hat, r_target_pred.detach())`.
- 의도: 예측된 goal 에 실제로 도달하는 action 을 뽑도록 학습 시 pressure.
- Aux 와 차이: target 이 GT 가 아닌 predictor 예측. Integrator = Linear(7→8, zero-init).

### 3.4 [범주 C] `VLA_MWM_StatePredictor_ErrorDenoise`
- 파일: [starVLA/model/framework/VLA_MWM_StatePredictor_ErrorDenoise.py](../starVLA/model/framework/VLA_MWM_StatePredictor_ErrorDenoise.py)
- 추가: 매 denoising 스텝에서 `error = r_target_pred − integrator(a_τ)` 계산 → error_encoder(MLP) 로 embed_dim 토큰 → `extra_self_tokens` 로 head 에 주입.
- 의도: Denoising 진행 중 동적 error 신호가 head 에 매 스텝 다르게 들어감.

### 3.5 [범주 C] `VLA_MWM_StatePredictor_LyapDenoise`
- 파일: [starVLA/model/framework/VLA_MWM_StatePredictor_LyapDenoise.py](../starVLA/model/framework/VLA_MWM_StatePredictor_LyapDenoise.py)
- 추가: Lyapunov gradient 로 velocity residual 계산.
  - `V(a) = ||r_target − r_present − W(Σa)||²`
  - `−∂V/∂a = α · w(τ) · (error @ W)`
- `velocity_residual` 훅 (E4) 으로 head 의 velocity 예측에 직접 잔차 더함.
- α = 1.0 고정 실험. τ-gate 학습 가능.

### 3.6 [범주 C] `VLA_MWM_StatePredictor_QLD` (Qwen-in-head + Lyap velocity)
- 파일: [starVLA/model/framework/VLA_MWM_StatePredictor_QLD.py](../starVLA/model/framework/VLA_MWM_StatePredictor_QLD.py)
- 두 축 결합:
  1. **Qwen-in-head**: `vl_embs = concat(state_cond, Qwen_action_tokens)` → head 가 Qwen 시각/언어 세부 특징을 직접 접근.
  2. **Lyapunov velocity residual**: α = 0.3 고정. LyapDenoise 축소 버전.
- 의도: VLA-JEPA 원본의 Qwen-직결 head 구조 + Mamba 예측 활용.

### 3.7 [범주 A] `VLA_MWM_StatePredictor_QOnly` (QLD ablation)
- 파일: [starVLA/model/framework/VLA_MWM_StatePredictor_QOnly.py](../starVLA/model/framework/VLA_MWM_StatePredictor_QOnly.py)
- QLD 에서 velocity residual 만 제거 (α=0). Qwen-in-head 만 유지.
- 의도: QLD 이득이 Qwen-in-head 때문인지 velocity residual 때문인지 분리.

### 3.8 [범주 D] `VLA_MWM_StatePredictor_Rerank` (test-time only)
- 파일: [starVLA/model/framework/VLA_MWM_StatePredictor_Rerank.py](../starVLA/model/framework/VLA_MWM_StatePredictor_Rerank.py)
- QOnly ckpt 재사용, 재학습 없음.
- Inference 매 step: head 를 N=8회 호출 → 각 후보 chunk 를 integrator 로 굴려 도달지 계산 → `r_target_pred` 와 가장 가까운 chunk 선택.
- 의도: Predictor 를 조건이 아닌 candidate filter 로 활용.

---

## 4. 결과 표 (500 rollout / suite / task)

### 4.1 최종 성공률

| Variant | 범주 | spatial | object | goal | 10 | **avg** | Δ vs baseline |
|---|---|---|---|---|---|---|---|
| DINO MWM-v2-FO (참고) | — | — | — | — | — | 94.2 | +0.2 |
| **Baseline StatePredictor** | — | **93.4** | **98.8** | **94.8** | **89.0** | **94.00** | 0 |
| ErrorDenoise (버그) | C | 94.8 | 98.8 | 93.6 | 90.2 | 94.35 | +0.35 (노이즈) |
| LyapDenoise α=1 (FixB) | C | 94.2 | 99.0 | 88.8 | 83.6 | 91.40 | −2.60 |
| QLD α=0.3 | C+A | 94.2 | **100.0** | 92.6 | 89.8 | 94.15 | +0.15 (노이즈) |
| QOnly α=0 | A | 94.4 | 99.8 | 92.0 | 90.6 | 94.20 | +0.20 (노이즈) |
| Rerank N=8 | D | 진행 중 | — | — | — | — | — |

### 4.2 학습 시 새 모듈 실제 학습 여부

초기 실험에서 발견한 버그: 부모 프레임워크 `set_stage("stage2")` 가 학습 대상 모듈을 하드코딩 (`mamba_predictor, cond_proj, action_model`) → 자식 클래스에서 새로 추가한 모듈이 자동으로 unfreeze 안 됨.

**영향 받은 변형**: Aux, Feedback, ErrorDenoise, LyapDenoise 초기 실행. 이들의 새 모듈 (`aux_state_head`, `integrator`, `error_encoder`) 이 학습되지 않은 채 30k step 을 돌아, 결과가 사실상 **baseline stage2 재학습의 seed noise** 였음.

**수정**: `set_stage` override 로 새 모듈을 명시적 unfreeze. LyapDenoise FixB, QLD, QOnly 는 수정 후 결과.

### 4.3 학습 후 가중치 검증

| Variant | integrator norm | 기타 |
|---|---|---|
| LyapDenoise (초기) | 0.00 (미학습) | 버그로 dead |
| LyapDenoise FixB (α=1) | 0.29 | 학습됨 |
| QLD (α=0.3) | 0.16 | 학습됨, tau_mid 0.50→0.54 |
| QOnly (α=0) | 0.05→ (초기값 유지) | integrator 사용 안 됨 |

---

## 5. 변형별 상세 분석

### 5.1 Baseline vs DINO — "State 로 충분하다" (강한 발견)

DINO latent (196 tok × 768) 을 8-dim state 로 대체해도 성능 유지 (94.2 → 94.0).

**해석**: Head 가 필요한 미래 정보는 **action-relevant subspace 의 goal 좌표**이며, 시각/언어 context 는 이미 Qwen action_tokens → Mamba 경로에서 처리됨. Head 의 조건은 "어디로 갈지" 만 있으면 충분.

**함의**: 이후 어떤 조건 강화 시도도 이 기저 위에서 이득을 내야 함.

### 5.2 Aux / Feedback (범주 B) — 학습 loss 만으론 부족

- Aux (GT-anchored, 92.75%)와 Feedback (predictor-anchored, 93.75%) 모두 baseline 하회.
- **원인**: (1) 초기 버그로 새 모듈이 실제 학습 안 됨 → 결과가 baseline noise 범위. (2) Loss 만 추가하는 것은 head 의 행동을 근본적으로 바꾸지 못함. Head 는 이미 flow-matching loss 로 데모 분포를 학습하고 있고, 추가 loss 는 미세한 편차만 유도.

### 5.3 ErrorDenoise (범주 C) — 동적 조건 주입, 미미한 개선

- 학습 시 매 배치에서 sampled τ 에 맞춰 `error = r_target_pred − integrator(a_τ)` 를 encode → self-attn extra token 으로 주입.
- 결과 94.35% (baseline +0.35, 노이즈 범위).
- **한계**: 새 모듈 학습 여부와 무관하게 (버그 있었음), 추가 token 이 이미 cross-attn cond 로 들어가던 정보와 근본 다르지 않음.

### 5.4 LyapDenoise (범주 C) — α 고정 velocity feedback

- α=1.0 (FixB): velocity residual 이 실제로 적용됨. **평균 91.40%, 큰 하락**.
  - Goal −6.0, libero_10 −5.4. Multi-modal / long-horizon 크게 타격.
  - Velocity 편향이 데모 분포 밖으로 밀어붙임 → head 가 이를 counter-learn 하면서 flow-matching fit 불안정.
- **결론**: 순수 Lyapunov gradient residual (α=1) 은 이 세팅에서 성능 저하 요인.

### 5.5 QLD (범주 C+A) — Qwen-in-head + moderate α

- 두 축 결합: Qwen 토큰을 head 조건에 concat + α=0.3 velocity residual.
- 결과 94.15% (baseline +0.15, 노이즈).
- **Object 100% 는 계열 최초** — 하지만 후속 QOnly 실험에서 이는 velocity residual 때문이 아니라 Qwen-in-head 때문임을 확인.

### 5.6 QOnly (범주 A ablation) — QLD 이득 분리

- QLD 에서 α=0 만 다르게. 평균 94.20% (QLD 94.15 와 사실상 동일).
- **결정적 판정**:
  - Velocity residual (α=0 vs 0.3) 은 성능에 **의미 있는 기여 없음** (spatial/10 은 α=0 이 더 나음, goal/object 는 α=0.3 이 미세 우세, 평균 무의미).
  - QLD 의 object 100% 는 Qwen-in-head 기여 (QOnly 99.8% 로도 거의 재현).
- Qwen-in-head 자체도 baseline 대비 +0.20 노이즈 수준.

### 5.7 Rerank (범주 D) — Test-time filtering (진행 중)

- 학습 없이 QOnly ckpt 로 inference 만 수정.
- Best-of-N (N=8) + integrator-based scoring.
- Spatial 부분 결과 94.1% — QOnly 대비 개선 없음.
- **잠정 해석**: N=8 후보가 대부분 서로 비슷하거나 (flow-matching sampling diversity 부족), integrator 가 candidate 를 유의미하게 구분 못함.

---

## 6. 왜 모든 시도가 plateau 인가 — 원인 분석

### 6.1 LIBERO 자체의 ceiling

- 94% 는 이 벤치마크 계열의 실질 상한. 남은 6% 는:
  - 초기 pose 노이즈로 물체 근처 못 감 (물리 실패)
  - 접촉 / 마찰 랜덤성
  - 데모에 없던 edge case
- 이런 실패는 **conditioning 개선 축과 직교**. 어떤 조건 강화도 해결 불가.

### 6.2 Head 가 이미 충분한 조건 정보를 받음

- 정보 흐름: `Qwen action_tokens → Mamba → r_target_pred (8-dim)` → head cond.
- Head 가 필요한 신호는 "어디로 갈지" 뿐이고, 이는 8-dim 로 표현 가능.
- 추가로 Qwen 토큰을 concat 해도 (QOnly), velocity residual 을 더해도 (QLD/LyapDenoise), auxiliary loss 를 붙여도 (Aux/Feedback), **head 가 이미 이용하는 정보 이상의 유효 신호가 없음**.

### 6.3 Flow-matching head 는 이미 표현력 충분

- DiT 기반 head 는 고차 표현력. Demo 분포 fit 이 병목이 아님.
- Conditioning 정보 부족이 병목이 아닌 상황에서 조건을 더 주는 것은 marginal.

### 6.4 Predictor 오차의 상한

- `r_target_pred` 자체가 완벽하지 않음 (state_err ~ 0.005).
- Predictor 예측을 완벽히 활용해도 이 오차만큼의 상한.
- Predictor 를 더 정확히 만들어도 다른 병목 (physics ceiling) 이 먼저 도달됨.

### 6.5 Multi-modality vs Guidance 트레이드오프

- LIBERO Goal suite (그리고 부분적으로 spatial) 는 여러 유효 궤적이 존재.
- Goal-directed guidance (velocity residual, Qwen 세부 조건, best-of-N filtering) 는 특정 mode 로 붕괴 유도.
- Object 같은 단일-mode 태스크에선 +, Goal 같은 multi-modal 태스크에선 −. **평균으로는 cancel out**.

### 6.6 Endpoint 조건의 근본 한계

- 우리가 조건으로 쓰는 것은 "**7 스텝 뒤 endpoint state 하나**". 궤적 shape 정보 없음.
  - 접근 각도, pre-grasp orientation, 감속 프로파일, gripper 타이밍 → endpoint 에 안 담김.
- 이런 shape 정보를 조건에 담으려면 **waypoint sequence 예측**이 필요. 지금 구조로는 임계.

---

## 7. 결론

1. **8-dim state 조건이 DINO latent (196×768) 조건과 등가** — VLA 에서 head 가 필요한 미래 정보는 endpoint state 로 충분.
2. **예측된 endpoint state 를 활용하는 4가지 범주 (passive cond, aux loss, denoising steering, test-time filtering) 모두 LIBERO 에서 baseline 을 유의미하게 넘지 못함**.
3. 원인은 (a) LIBERO physics ceiling, (b) head 표현력 이미 충분, (c) predictor 오차 상한, (d) multi-modal/guidance trade-off, (e) endpoint 조건의 구조적 한계 (궤적 shape 부재) 의 복합.
4. **본질적 개선은 조건 강화가 아닌 다음 축에서 필요**:
   - Predictor 를 waypoint sequence 로 확장 → 궤적 shape 제공.
   - Gripper / discrete decision 을 별도 채널로 처리.
   - Task phase (reach vs contact) 별 adaptive gating.
   - LIBERO 를 넘어선 harder benchmark 로 headroom 확보.

---

## 8. Future Work

### 8.1 Waypoint sequence predictor
Predictor 출력을 endpoint 하나 → `[r_{t+1}, ..., r_{t+H}]` 궤적 전체로. Head 는:
- 각 waypoint 를 별도 조건으로 → step-wise tracking 가능.
- Gripper 채널의 시점별 값 → open/close 타이밍 조건으로 제공.
- Multi-modal 궤적은 top-K waypoint sequence 로 표현 가능.

이 구조에선 진짜 closed-loop 제어 (LQR, MPC) 도 하이브리드로 결합 가능.

### 8.2 Contact/discrete decision channel
현재 gripper 는 continuous action 채널 (0~1) 로 취급되어 flow-matching 이 데모 분포에서 "언제 열고 닫는지" 를 학습. 이를 명시적 discrete 결정 채널로 분리:
- Predictor 가 별도로 gripper action timing 을 출력.
- Head 는 continuous 채널 (pose) 만 담당.
- 접촉 시점의 불연속성을 자연스럽게 처리.

### 8.3 Adaptive guidance gating
Task phase (reach / grasp / release) 별로 guidance 강도 자동 조절:
- Reach phase 에선 강한 goal-directed guidance.
- Grasp / release 에선 데모 스타일 우선 (guidance off).
- Phase 추정은 gripper 상태 + object proximity 로 가능.

### 8.4 Harder benchmark 확장
LIBERO 는 saturation. 실제 개선 여지가 있는 벤치마크에서 우리 방법론들을 재검증:
- **LIBERO-Plus**: MWM-v2-FO 72.87% 로 headroom 충분.
- **Contact-rich manipulation** (assembly, deformable): 접촉 중심.
- **Real robot deployment**: 시뮬-실기 gap 에서 conditioning 견고성 검증.

### 8.5 새로운 action head 구조
Flow-matching 을 대체하거나 보강할 후보:
- **Waypoint-conditioned diffusion policy** (Chi et al. 확장).
- **Hierarchical policy** (high-level waypoint + low-level tracker).
- **Consistency flow matching** (ManiFlow 등 최신 기법).

---

## 9. 부록: 실험 재현

### 9.1 파일 위치
- Framework: `starVLA/model/framework/VLA_MWM_StatePredictor*.py`
- Run scripts: `scripts/run_mwm_state_pred_*.sh`
- Eval: `scripts/eval_libero_all.sh`
- Results: `results/mwm-state-pred*/`, `results/eval/*_mwm-state-pred*/`

### 9.2 학습 세팅 (공통)
- Stage1 (predictor): 30k step, `L_pred = L1(r_target_pred, r_target_gt)`, batch=16×2GPU.
- Stage2 (head + 새 모듈): 30k step, `L = L_pred + L_action`, LIBERO all mix.
- Warmup: 3000 step. Cosine LR schedule.
- Qwen LoRA rank=16.

### 9.3 Eval 세팅
- 4-suite (spatial, object, goal, libero_10) × 10 task × 50 trial = 500 rollout / suite.
- Server-client 구조 (`server_policy.py` + `examples/LIBERO/eval_libero.py`).
- Seed = 7 고정.

### 9.4 하드웨어
- 2× NVIDIA RTX PRO 6000 Blackwell.
- 학습 1회: ~24시간 (모듈 크기에 따라 13~26시간).
- Eval 1회 (4-suite × 500): ~6시간.
