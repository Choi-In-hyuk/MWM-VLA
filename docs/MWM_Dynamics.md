# MWM-Dynamics — Predictive Closed-Loop Action Head

**Framework**: `VLA_MWM_Dynamics`
**Baseline**: MWM-v2-FO (`VLA_DINO_StreamingMamba_FutureOnly`, LIBERO 94.2%)
**Author intent**: Inhyuk Choi
**Status**: Phase 2a warmup running (2026-08-13)

---

## 1. 왜 이걸 만드나

### 기존 baseline (MWM-v2-FO) 의 근본 한계

Baseline action head 입력:
```
cond = cond_proj([s_present ⊕ s_target_pred])   # DINO latent 직접 주입
      → cross-attn K/V
Loss = velocity MSE only
```

**문제**:
- Head 는 `s_target_pred` 를 다른 VLM/language token 과 구분 없이 그냥 attention 으로 소비
- Physical meaning (미래 상태) 이 attention 을 거치면서 사라짐 — 그냥 벡터
- BC loss 는 이 latent 를 "잘 쓰라" 고 강제하지 않음 → cross-attn 이 이걸 무시하는 게 최적일 수도 있음

제어 이론 언어로: baseline 은 **"미래를 조건 feature 로 넘긴 순수 imitation"**, 진짜 predictive control 아님.

### 세 가지 요구를 동시에 만족해야 함

1. **학습 시 error 신호로 head 훈련** (미래를 supervision 으로 활용)
2. **추론 시 미래 예측이 능동적으로 action 생성에 개입** (실시간 폐루프)
3. **미래 예측이 있을 때만 가능한 제어** (deadbeat terminal tracking, preview control, ambiguity resolution)

### 핵심 아이디어

학습된 action-conditional dynamics `f_θ(s_present, a) → ŝ_next` 를 도입.
- **Reference**: `s_target_pred` (predictor 가 알려주는 목표 미래)
- **Actual (simulated)**: `f_θ(s_present, a)` (head 액션이 만들 예측 미래)
- **Error**: `e = s_target_pred − f_θ(s_present, a)`
- 이 `e` 를 flow-matching velocity 에 conditioning + gradient path 로 흘려서 head 를 학습

제어 이론 매핑: **`u = feedforward + K·e`** (PID 의 P 항). Flow-matching 이 feedforward, `w(τ)·project(e_τ)` 가 feedback.

---

## 2. 아키텍처 상세

### 모듈 계층 (Stage 2)

```
┌─ Frozen (외부 사전학습 + Stage 1 결과) ────────────┐
│  DINO ViT-B/14     : Meta DINOv2 pretrained       │
│  Qwen-VL 2B        : HF pretrained backbone        │
│  Predictor P       : StreamingMamba, Stage 1 결과  │
└────────────────────────────────────────────────────┘

┌─ Trainable ─────────────────────────────────────────┐
│  Qwen LoRA         : r=16, α=32 (backbone 그대로)   │
│  Head π (DiT-B)    : flow matching action head      │
│  E1 pool_project   : error token 생성 (~8M)         │
│  E4 cross_attn_proj: velocity correction 생성 (~1M) │
│  K_θ (scalar)      : E4 gain, init 0                │
│  f_θ (신규 모듈)    : action-cond dynamics (~4M)     │
└─────────────────────────────────────────────────────┘
```

### 신규 모듈 상세

#### `f_θ` — Action-Conditional Dynamics Head

**파일**: [starVLA/model/modules/world_model/action_dynamics.py](../starVLA/model/modules/world_model/action_dynamics.py)

**입출력**:
```
Input:  s_present [B, N=512, D=768]   # DINO latent, 2 views × 256 patches
        a         [B, H=7, D_a=7]     # action chunk
Output: ŝ_next   [B, N=512, D=768]   # 예측된 미래 latent (residual)
```

**구조** (F1: tiny cross-attn, ~3.6M params):
```
a_emb = MLP(a) + pos_emb            # [B, H, D']  D'=256
                     │
   ┌─────────────────┼──────────────┐
   │                                │
s_present  ──► self-attn (2 layer) ──► q  [B, N, D']
                     │
              cross-attn (2 layer): q ← k,v = a_emb
                     │
              MLP → dim 768
                     │
              + s_present (residual, output layer zero-init)
                     │
                   ŝ_next
```

**Zero-init 결정**: output layer `vis_out[1]` 을 zero-init.
- 초기엔 f(s, a) = s (identity residual) → warmup 안전
- Phase 2a 에서 L_dyn 으로 학습 → identity 에서 벗어남
- Phase 2b 진입 시 이미 non-trivial → L_consistency gradient path 살아남

> **설계 히스토리 (혼란 방지용)**: 개발 중 이 zero-init 을 두고 두 번 flip 이 있었음.
> (1) 초기 추가 → (2) grad smoke TEST 5 실패로 "d(f)/d(a)=0 이라 consistency dead-gradient" 판단, 임시 제거 → (3) 리뷰어 지적 "over-fix, Phase 2b 진입 시엔 f 가 이미 warmed up 이라 문제 없음, smoke test 를 warmup-이후 상태로 검증하면 됨" 반영, **zero-init 복원** + smoke test 에 `f.vis_out` perturbation 추가.
> **최종 상태 (현재 코드)**: zero-init 유지. Phase 2a step 50 에서 `dyn_cos == dyn_ident_cos` 로 확인됨 (f 가 정확히 identity residual).

#### `E1: ErrorTokenProjector` — self-attn context injection

**파일**: [starVLA/model/framework/VLA_MWM_Dynamics.py](../starVLA/model/framework/VLA_MWM_Dynamics.py) `ErrorTokenProjector` class

- Input: `e_stop` [B, N=512, D=768]
- Output: `e_tok` [B, N_tokens=8, D_head=768]
- 방법: learned query × cross-attn to error patches
- `ff.last` zero-init: 초기엔 query embed 만 (near-constant), 학습으로 refine

#### `E4: VelocityCorrectionProjector` — velocity residual injection

**파일**: 같은 파일 `VelocityCorrectionProjector` class

- Input: `e_stop` [B, N=512, D=768]
- Output: `correction` [B, H=7, action_dim=7]
- 방법: learned per-action-step queries × cross-attn to error patches → MLP → action space

**중요**: `to_action` 은 **zero-init 하지 않음**. `K_θ=0` 만으로 초기 안정성 확보.
이유: 둘 다 zero-init 하면 `K * gate * correction = 0 * 0` 로 gradient chain 죽음 → E4 가 영원히 학습 안 됨.

#### `K_θ` (learnable scalar)

- init 0 → 초기엔 E4 correction 이 velocity 에 0 만큼 기여 (안전)
- `d(L)/dK` 는 nonzero (gate * correction * error) → K 는 step 1 부터 학습 시작
- K 가 성장하면 그때부터 E4 upstream (correction projector) 도 gradient 받음

### Head 전체 forward

```
[입력]
  vlm_action_tokens    [B, Na, 2048]      Qwen 출력 (semantic intent)
  s_present, s_target_pred [B, N, 768]    DINO latent (frozen predictor 결과)
  robot_state          [B, 8]              proprioception
  a_gt                 [B, H, 7]           supervision (학습만)

[Flow matching sampling — 방식 A (teacher-forced)]
  noise ~ N(0, I)
  τ     ~ Beta(1.5, 1.0)
  a_τ   = (1-τ) · noise + τ · a_gt

[f forward]
  s_hat_tau = f_θ(s_present, a_τ)

[Error 신호 계산]
  e_deploy = s_target_pred.detach() − s_hat_tau     # gradient 경로 있음 (via s_hat_tau → f)
  e_stop   = e_deploy.detach()                       # 완전 detached (constant)
  e_tok    = e1_project(e_stop)                      # E1
  correction = e4_project(e_stop)                    # E4

[Head (DiT-B) forward]
  sa_embs = [state, future_tokens, action_features, e_tok]   # e_tok 이 self-attn 시퀀스에 추가
  v_flow = DiT(sa_embs, cross=vlm_action_tokens, τ)

[Velocity 합성 — E4 (feedforward + feedback)]
  w_τ = τ²
  v   = v_flow + K_θ · w_τ · correction    ← PID 스타일
```

### 데이터/신호 흐름 도식

```
Qwen ─► vlm_tokens ────────────────────────────► cross-attn K/V
                                                        │
DINO ─► s_present ┬─► f_θ (a_τ) ──► ŝ_τ ──► e ──┐    │
                  │                              │    │
                  └─► s_target_pred (frozen P)   │    │
                                    │            │    │
                                    └────► e_deploy ──┼─► E1 (self-attn token)
                                                       └─► E4 (velocity residual)
                                                                    │
noise + a_gt ──interp──► a_τ ──► DiT ──► v_flow ──►  v = v_flow + w(τ)·correction
                                                                    │
                                                            L_flow ← v vs (a_gt − noise)
```

---

## 3. Loss 구조 (gradient path 분리)

### 세 개의 gradient stream

#### Stream 1: f 학습 (target = 항상 `s_target_gt`)
```python
L_dyn         = ‖f(s_present, a_gt)  − s_target_gt‖₁         # a_gt 분포
L_dyn_via_ref = ‖f(s_present, a_τ)   − s_target_gt‖₂         # a_τ (interp) 분포, augmentation
```
- Gradient path: `L → f params only`
- Head 파라미터로 안 흐름
- 두 loss 로 f 는 두 분포 모두 커버

**중요**: 두 loss 모두 target 은 `s_target_gt` (ground truth). `s_target_pred` (predictor 예측) 을 target 으로 쓰면 predictor bias 가 f 로 새어들어감.

#### Stream 2: head flow matching (표준 flow matching)
```python
L_flow = ‖v − (a_gt − noise)‖²
```
- Gradient path: `L → v → v_flow (→ DiT params) + correction (→ e4_project params)`
- `e_stop = e_deploy.detach()` → E4 correction 은 head params 로만 gradient (f 로 안 감)
- `e_tok = e1_project(e_stop.detach())` → E1 도 head 로만 (f 격리)

#### Stream 3: consistency (F 조건만) — head 로 진짜 gradient
```python
# f params 잠깐 freeze (⚠ requires_grad_(False) 만 사용, torch.no_grad() 절대 금지)
with freeze_params_context(f):                        # forward 는 autograd ON
    a_hat = a_τ + (1 − τ) · v_flow                    # head 의존
    s_hat = f(s_present, a_hat)                        # gradient 정상 흐름

L_consistency = w_τ · ‖s_target_pred.detach() − s_hat‖²
```
- Gradient path: `L → s_hat → a_hat → v_flow → head params`
- **head 가 "e 를 줄이는 action" 을 직접 학습하는 유일한 stream**
- `w_τ = τ²` 로 두 근거 (f OOD at small τ + Euler 근사 오차 at small τ) 동시 완화

### 최종 loss
```
L_total = L_flow                                    (main, anchor)
        + β · (L_dyn + λ_aug · L_dyn_via_ref)       (β=0.1, λ=0.5)
        + γ · L_consistency                          (γ ramp 0→target, F 조건만)
```

---

## 4. Ablation 계획 (7 조건, config flag 로만 분기)

| # | E4 K(τ) | e_stop detach | L_dyn_via_ref | L_consistency | 의미 |
|---|---|---|---|---|---|
| **A** | off | — | off | off | Baseline (pure BC + cond) |
| **B** | K=τ² | ✓ | on | off | **권장 (implicit weighting)** |
| C | K=const | ✓ | on | off | 게이팅 필요성 |
| D | K=τ² | ✗ | on | off | co-adapt 시험 (detach 필요성) |
| E | E4 off, E1 만 | — | on | off | "context 주입" vs "velocity feedback" |
| **F** | K=τ² | ✓ | on | on (w(τ)) | **explicit head gradient** |
| G | E4 off | — | on | on | consistency 단독 |

**핵심 비교**:
- **A ↔ B**: closed-loop 도입 자체의 효과
- **B ↔ F**: implicit weighting 만으로 충분한가 vs explicit gradient 필요한가 (최대 관심사)
- **B ↔ E**: velocity feedback 이 context 주입보다 나은가 (구조적 novelty 검증)

### Config flag 대응
```
--mwm_e4_enabled True|False
--mwm_e4_gate tau_sq|const|off
--mwm_e4_detach True|False
--mwm_e1_enabled True|False
--mwm_e1_num_tokens 8
--mwm_consistency_enabled True|False
--mwm_beta_dyn 0.1
--mwm_lambda_aug 0.5
--mwm_gamma_consistency 0.5
```

---

## 5. 결정적 구현 주의사항

### (1) `torch.no_grad()` 절대 금지 (F 조건 consistency)
```python
# ❌ WRONG — head gradient path 통째로 죽음
with torch.no_grad():
    s_hat = f(s_present, a_hat)

# ✅ CORRECT — f params 만 잠금, forward autograd 유지
for p in f.parameters(): p.requires_grad_(False)
s_hat = f(s_present, a_hat)
for p in f.parameters(): p.requires_grad_(True)
```

### (2) f 학습 target 은 언제나 `s_target_gt`
Predictor 편향이 f 로 새어들어가지 않도록.  
`s_target_pred` 는 head 조건 신호 (E1/E4/consistency) 에서만.

### (3) `e_deploy` (grad 있음) 과 `e_stop` (완전 detach) 변수 분리
```python
e_deploy = s_target_pred.detach() - s_hat_tau       # f 로 grad 감 (원한다면 e4_detach=False)
e_stop   = e_deploy.detach()                         # 완전 constant (velocity injection 용)
```
분리 안 하면 detach 실수로 gradient path 죽거나 반대로 원치 않는 co-adapt 발생.

### (4) Predictor 는 완전 frozen
- `s_target_pred = P(...)` 를 `torch.no_grad()` 안에서 계산
- Stage 2 optimizer 에서 predictor params 완전 제외
- 3-way collusion (predictor-f-head) 원천 차단

### (5) `w(τ) = τ²` 를 두 loss 에 동일 적용
두 근거 (f OOD + Euler 근사 오차) 방향 일치 → 공통 스케줄로 일관성.

### (6) Zero-init 지점
- `f.vis_out`: zero-init (identity 시작, warmup 안전)
- `e1.ff.last`: zero-init (초기엔 query embed 만)
- `K_θ`: zero-init (E4 초기 기여 0)
- `e4.to_action`: **zero-init 안 함** (K_θ 와 곱해지면 gradient chain 죽음)

---

## 6. Verification (커밋 전 필수)

### 6.1 Gradient path smoke test
**파일**: [scripts/smoke_mwm_dynamics_grad.py](../scripts/smoke_mwm_dynamics_grad.py)

6개 assertion:
1. `L_dyn` → f 만 grad
2. `L_dyn_aug` → f 만 grad
3. `L_flow (detach=True)` → head/e1/e4/K 만 grad, f 격리
4. `L_flow (detach=False)` → f 도 co-adapt
5. **`L_consistency` → head 만 grad, f 격리** (F 조건 유효성)
6. 통합 loss → 모두 grad

이 6개 통과 == gradient 배선 정확.

### 6.2 End-to-end smoke
**파일**: [scripts/smoke_mwm_dynamics_e2e.py](../scripts/smoke_mwm_dynamics_e2e.py)

실제 backbone ckpt + tiny fake batch 로 stage 2a + stage 2b 각각 forward + backward.
- 모든 loss stream 반환값 확인
- 각 stage 에서 예상 grad 유무 확인

---

## 7. 학습 파이프라인

### 스크립트
**파일**: [scripts/run_mwm_dynamics.sh](../scripts/run_mwm_dynamics.sh)

3-phase 자동 실행:
```
Phase 2a → Phase 2b (B condition) → LIBERO 4-suite eval
```

### 실행 커맨드
```bash
# 전체 자동 실행
bash scripts/run_mwm_dynamics.sh all

# 부분 실행
bash scripts/run_mwm_dynamics.sh phase2a
bash scripts/run_mwm_dynamics.sh phase2b
bash scripts/run_mwm_dynamics.sh eval

# 환경 변수 override
PHASE2A_STEPS=20000 PHASE2B_STEPS=30000 NPROC=2 bash scripts/run_mwm_dynamics.sh all
```

### Phase 2a: f warmup
```
Trainable: f_θ only (~5M)
Frozen:    predictor, DINO, Qwen, head, E1/E4
Loss:      L_dyn = L1(f(s_present, a_gt), s_target_gt)
Recipe:    batch 16, lr 2e-4, warmup 1000, cosine decay
목표:      f 가 identity 에서 벗어나 dynamics 획득 (dyn_cos > dyn_ident_cos + 0.05)
```

### Phase 2b: joint (B condition 권장)
```
Trainable: head, cond_proj, f, E1/E4, K_θ, Qwen LoRA (~170M)
Frozen:    predictor, DINO, Qwen backbone
Loss:      L_flow + β·(L_dyn + λ_aug·L_dyn_via_ref)   (B 조건: consistency off)
Recipe:    batch 16, lr 1e-4, warmup 1000, cosine decay
목표:      head 가 error signal 을 활용해 정확한 action 학습
```

### Eval
```
LIBERO 4-suite: spatial → object → goal → 10
50 trials/task × 40 tasks = 2000 rollout
비교: baseline 94.2% vs 우리
```

---

## 8. 파일 요약

| 파일 | 역할 |
|---|---|
| [starVLA/model/modules/world_model/action_dynamics.py](../starVLA/model/modules/world_model/action_dynamics.py) | 신규: f_θ (ActionDynamicsHead) |
| [starVLA/model/framework/VLA_MWM_Dynamics.py](../starVLA/model/framework/VLA_MWM_Dynamics.py) | 신규: 프레임워크 (E1/E4 projection + 3 loss streams + closed-loop inference) |
| [starVLA/model/modules/action_model/GR00T_ActionHeader.py](../starVLA/model/modules/action_model/GR00T_ActionHeader.py) | 확장: `extra_self_tokens` + `velocity_residual` optional kwargs + `on_step` callback |
| [scripts/train_mamba_wm.py](../scripts/train_mamba_wm.py) | 확장: `stage2a_f`/`stage2b_joint` + `--mwm_*` flag 15개 |
| [scripts/run_mwm_dynamics.sh](../scripts/run_mwm_dynamics.sh) | 신규: 파이프라인 자동 실행 |
| [scripts/smoke_mwm_dynamics_grad.py](../scripts/smoke_mwm_dynamics_grad.py) | 신규: 6개 gradient-path assertion |
| [scripts/smoke_mwm_dynamics_e2e.py](../scripts/smoke_mwm_dynamics_e2e.py) | 신규: 실제 backbone forward+backward |
| [starVLA/model/framework/VLA_DINO_StreamingMamba_FutureOnly.py](../starVLA/model/framework/VLA_DINO_StreamingMamba_FutureOnly.py) | Baseline (참조용, 수정 안 함) |

---

## 9. B 조건 실험 결과 (2026-08-14 → 2026-08-16 학습, 2026-08-17 eval)

### 학습 궤적 완료

**Phase 2a 최종 (step 20000)**:
```
dyn=0.624  dyn_cos=0.834  dyn_ident_cos=0.773  gap=0.061
```
예상 범위 안. f 가 identity 에서 명확히 벗어남.

**Phase 2b 최종 (step 30000)**:
```
flow=0.056  dyn=0.619  dyn_aug=0.801  pred_cos=0.827
```
flow 안정적 감소 (0.91 → 0.056). f drift 없음.

### LIBERO Eval — **Baseline 대비 대폭 하락** ⚠️

| Suite | Baseline (MWM-v2-FO) | **Ours (MWM_Dynamics B)** | Δ |
|---|---|---|---|
| libero_spatial | 94.4% | 90.8% | −3.6% |
| libero_object | 98.0% | 97.0% | −1.0% |
| libero_goal | 94.0% | 82.2% | −11.8% |
| libero_10 (long) | 89.8% | **56.8%** | **−33.0%** |
| **평균** | **94.1%** | **81.7%** | **−12.4%** |

Easy suite (object, spatial) 은 유지되었지만 **long-horizon (libero_10) 에서 대참사**.

### 🔴 실패 원인 분석

**Final checkpoint 진단**:
```python
K_theta: tensor([0.0094])   # 거의 0
```

**우리 B 조건 = 사실상 "context 손실만 있고 gain 없는" 상태**:

| 메커니즘 | 의도 | 실제 |
|---|---|---|
| Head 입력에서 DINO 제거 | separation principle | ✅ 실행 — **정보 손실만 발생** |
| E1 (error token in self-attn) | error context 신호 | ⚠️ 그냥 token 하나 추가한 수준 |
| E4 (velocity residual K_θ·correction) | PID-style feedback | ❌ **K_θ ≈ 0 → 사실상 dead** |
| L_consistency (F 조건) | head 로 explicit gradient | ❌ B 조건에선 off |

### 우리가 착각한 두 가정

1. **"E1 만으로도 head 가 error 활용법을 배울 것"** → X. 그냥 다른 token 처럼 attention 으로 흡수됨. "error" 라는 physical meaning 강제 X
2. **"K_θ 는 gradient 로 자동 성장할 것"** → X. init 0 에서 0.0094 까지밖에 못 큼. E4 gate `w(τ)=τ²` 가 초반에 매우 작아서 K_θ 의 gradient 신호가 미미했을 것

### 반성

**우리가 한 건 진짜 error-based 학습이 아니었음**. 이론상만 있었고 실제 mechanism 은 dead. 결과적으로:
- Baseline 대비 head 입력 정보만 삭감한 상태 (**degraded baseline**)
- Long-horizon 에서 spatial context 부족이 누적되어 -33% 하락

**진짜 error-based 였으려면**:
- F 조건 (consistency loss) 필수 → head 로 explicit "reduce e" gradient
- 또는 K_θ init 크게 (0.1~0.5) → E4 처음부터 작동

---

## 10. Novelty 재평가 (B 실패 반영)

### B 실험이 보여준 것
- **가정 오류**: E1 tokens + K_θ 자동 성장으로 error-based 학습이 될 것 → **오답**. 실제로는 E4 dead, E1 은 그냥 context, effective architecture 는 "degraded baseline"
- **정보 손실 우선순위**: 우리가 head 입력에서 DINO 제거로 얻은 것보다 잃은 게 훨씬 큼 (-12.4% avg, -33% long-horizon)

### 남은 valid novelty (α 방향)
1. **VLA 세팅에서 learned compact state (Mini-Dreamer style)** — DINO SSL feature 대체
2. State 안에 language intent embedding (L1) — "situation = 물리 환경 + 의도" 통합 표현
3. Encoder + latent dynamics + optional decoder joint 학습 for physical AI

### 남은 valid novelty (β 방향 — α 성공 시)
1. Learned state 위에서 error-based flow matching
2. F 조건 (1-step consistency with frozen critic) 이 진짜 필요한가 실증

### 취약점 (정직)
- 아직 성능 검증 안 됨. α 가 baseline 도 못 넘길 수도 있음
- Mini-Dreamer 자체는 Hafner 원조. 우리 조합만 novel
- Physical AI 스토리는 있지만 정량 근거 (성능 향상) 없으면 방어 어려움

### 결론
- 방법론 novelty: **중** (조합 + physical AI 서사)
- 실험적 기여: **α 결과에 전적으로 달림**. Baseline 을 넘거나 최소 유지해야 valid
- B 결과는 negative result 로 부록에 언급 가능 ("naive integration of error signal fails when architecture regresses")

---

## 11. 서사 (α 방향으로 재작성)

> Frozen SSL feature (DINO) 는 image classification 용 표현이지 로봇 action 결정에 optimal 이 아니다. 특히 language intent 와 분리되어 있어 physical AI 에 맞지 않는다.
> 우리는 **learned compact state** 를 도입하여 DINO patches + language intent + robot proprioception 을 하나의 상황 표현으로 통합하고, 이 안에서 latent dynamics 를 학습한다.
> Baseline 대비 정량 향상이 확인되면, 그 위에 **learned dynamics 를 critic 으로 하는 error-based flow matching action head** 를 얹어 폐루프 제어를 실현한다 (Phase β).

### (레거시) B 조건 서사 — 폐기
> ~~Reactive controller 는 원리적으로 tracking error 를 학습 신호로 쓸 수 없다~~ (이 방향은 K_θ dead + head 정보 손실로 실증 실패)

---

## 12. 다음 실험 β' (확정) — Consistency-only + EMA Teacher

### 재설계 근거

B 실패에서 배운 것 → 재설계 원칙:

| 실패 원인 (B) | 재설계 (β') |
|---|---|
| Head 입력 DINO 제거 → 정보 손실 (-33% long-horizon) | **Head 입력에 DINO 유지** (baseline 그대로) |
| E1 tokens → 그냥 context 로 흡수, error 아님 | **E1 삭제** |
| E4 K_θ ≈ 0 → dead | **E4 삭제** |
| Consistency (F) off → head 로 gradient 없음 | **Consistency ONLY 로** ⭐ |
| Predictor 출력이 noisy → error signal 신뢰도 낮음 | **EMA teacher 로 predictor denoising** |

**핵심 원칙**: E1/E4 는 실패. Head 로 진짜 error gradient 를 흘리는 유일한 경로 = **consistency loss** 를 유일한 mechanism 으로.

### Stage 1' — Predictor EMA teacher fine-tune

**동기**: DINO SSL feature 는 stochastic. Predictor 가 이 noisy target 을 학습 → 출력도 noisy. Error-based 학습에 signal-to-noise 낮아짐.

**해결**: EMA teacher 로 temporal smoothing → denoised prediction.

**Recipe** (기존 predictor 를 fine-tune, 처음부터 재학습 X):
```python
# Init from existing baseline predictor
teacher = deepcopy(predictor)
for p in teacher.parameters(): p.requires_grad_(False)

# Each step
s_pred = predictor(past, present, qwen)                        # 학생 출력
target_dino = DINO(future_frame)                               # 원본 target (noisy)

with torch.no_grad():
    target_ema = teacher(past, present, qwen)                  # EMA teacher (smoothed)

L_dino = L1(s_pred, target_dino)                               # ground truth
L_ema  = L1(s_pred, target_ema.detach())                       # consistency with slow-moving self
L = L_dino + 0.5 · L_ema

# EMA update
for p_t, p_s in zip(teacher.parameters(), predictor.parameters()):
    p_t.data = 0.996 · p_t.data + 0.004 · p_s.data
```

**Hyperparameters**:
- EMA decay: **0.996** (BYOL 표준, ~250 step half-life)
- λ_ema: **0.5**
- Steps: **~5k step fine-tune** (기존 predictor 시작이라 짧게)

### Stage 2a' — f warmup (기존과 동일)

```
Trainable: f_θ only (~5M)
Loss:      L_dyn = L1(f(s_present, a_gt), s_target_gt)
Steps:     10-20k
```
변경 없음.

### Stage 2b' — Consistency-only Joint Training

**아키텍처** (baseline 최소 변경):
```
[Frozen — Stage 1' 결과]
Predictor P (EMA teacher 로 denoising 됨)
     ↓
s_target_pred (clean reference)

[Head input — baseline 그대로]
cond = cond_proj([s_present, s_target_pred])          ← DINO 유지, 정보 손실 없음
head_input = (cond, robot_state, a_τ, τ)

[Head forward — baseline 그대로]
v_flow = DiT(cond, robot_state, a_τ, τ)               ← baseline architecture

[NEW: consistency mechanism]
a_hat = a_τ + (1-τ) · v_flow                          # 1-step Euler to final
with frozen_params(f):                                # f는 critic
    s_hat = f(s_present, a_hat)                        # gradient flows through a_hat
L_consistency = w(τ) · ‖s_target_pred - s_hat‖²
```

**Loss**:
```
L = L_flow + β · L_dyn + γ · L_consistency

β = 0.1                                (f 유지 학습)
γ = 0 → 0.5 linear ramp 5k steps       (초반 head 안정화 후 서서히 도입)
w(τ) = τ                               (linear gate, not τ² — 초기 signal 좀 더 살리기)
```

### Gradient path (세 streams 명확)

```
Stream 1: f 학습 (isolated)
  L_dyn → f params only

Stream 2: head flow matching (baseline)
  L_flow → v_flow → DiT/cond_proj params

Stream 3: consistency (β' 핵심) ⭐
  L_consistency → s_hat → a_hat → v_flow → DiT params
  (f params requires_grad_(False), forward autograd ON)
```

### Train vs Inference 비대칭 (의도적)

**Training**:
- Predictor forward → s_target_pred
- Head forward with cond=[s_present, s_target_pred]
- Extra: f forward on a_hat, consistency loss backprop through head

**Inference**:
- Predictor forward → s_target_pred (baseline 그대로)
- Head forward with same cond (baseline 그대로)
- **f, consistency 없음. 완전 baseline forward, zero overhead**

**의미**:
- 학습 때 error signal 이 head weight 에 shaped → head 가 s_target_pred 를 "reference" 로 학습
- 추론 때 명시적 error correction 없지만 head 가 이미 그렇게 학습되어 있음
- 이전 계획 (E4 runtime correction) 은 B 에서 실패 (K_θ dead) → runtime error 는 접기
- 대신 training-time consistency 로 확실한 gradient path 확보

### 왜 이게 나은가

1. **Head 정보 손실 zero**: DINO cond 유지 → long-horizon 정보 확보
2. **Error 학습 mechanism 명확**: consistency loss 한 개, gradient path 명료
3. **Reference 신뢰도 향상**: EMA teacher denoising → error signal-to-noise 개선
4. **아키텍처 단순**: E1/E4 없음, K_θ 없음, tuning parameter 줄어듦
5. **Runtime overhead zero**: 추론은 baseline 과 동일

### 전체 파이프라인 (β')

```
Stage 1' (신규):
  Predictor fine-tune with EMA teacher
  5-10k steps
  Trainable: predictor + Qwen LoRA
  Loss: L_dino + 0.5·L_ema

Stage 2a' (기존):
  f warmup
  10-20k steps
  Trainable: f_θ only
  Loss: L_dyn

Stage 2b' (재설계):
  Consistency-only joint
  30k steps
  Trainable: head + cond_proj + f + Qwen LoRA
  Frozen: predictor (Stage 1' 결과)
  Loss: L_flow + 0.1·L_dyn + γ·L_consistency (γ ramp)

Eval:
  LIBERO 4-suite × 50 trials
```

### 폐기된 방향 (B 실험 + α 논의로 배운 것)

- ❌ **Head 입력에서 DINO 완전 제거** (B) — long-horizon 에 치명타
- ❌ **E1 만으로 error 학습 유도** (B) — 그냥 context token 으로 흡수됨
- ❌ **K_θ init 0 으로 자동 성장 기대** (B) — 30k step 도 안 커짐
- ❌ **E4 runtime velocity correction** — K_θ dead 로 실증적 실패
- ⏸️ **Learned state (α, Mini-Dreamer)** — Predictor 를 encoder 로 쓰는 정체성 애매. 보류
- ⏸️ **DDPO/DPPO RL fine-tune** — β' 성공 후 확장 옵션
