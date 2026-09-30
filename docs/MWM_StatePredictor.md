# MWM State Predictor — 아키텍처 · 결과 · 액션 생성 방식

MWM-v2-FO의 **DINO latent 예측**을 **로봇 state 예측**으로 대체한 변형 3종.

## 1. 아키텍처

### 1.1 Baseline: `VLA_MWM_StatePredictor`
파일: [starVLA/model/framework/VLA_MWM_StatePredictor.py](../starVLA/model/framework/VLA_MWM_StatePredictor.py)

**핵심 가설**: action head가 필요한 것은 고차원 시각 미래가 아니라 "action-relevant subspace의 language-grounded goal"이다. 로봇 state (EE pose + gripper, 8-dim) 자체가 그 조건에 부합하고, Qwen이 이미 scene context를 action_tokens로 나르고 있으므로 DINO는 predictor 경로에서 완전히 스킵.

**파이프라인**:
```
(r_past, r_present, Qwen action_tokens) --Mamba--> r_target_pred [B, 8]
cond = state_cond_proj([r_present, r_target_pred, r_target_pred - r_present])
                                                    [B, K, qwen_dim]
L_pred   = L1(r_target_pred, r_target_gt)
L_action = flow_matching(cond, actions, r_present)
```

- **Mamba predictor**: `state_dim=512, depth=6`, **per-dim tokenization** — state 각 scalar가 하나의 token (`tokens_per_frame=r_dim=8, dino_dim=1`). 8→512 up-projection 낭비 방지, `patch_pos`가 자연스럽게 "which dim" 임베딩 역할.
- **cond_proj**: `[r_present, r_target_pred, Δ]` (3×8) → seed MLP → K=8 tokens × qwen_dim=2048. 약 0.57M params.
- **DINO는 parent __init__에서 로드되지만 forward에서 호출되지 않음** (surgery 최소화, 방향이 맞으면 완전 제거 가능).

### 1.2 `_AuxHead` — backbone regularizer (Method 3)
파일: [starVLA/model/framework/VLA_MWM_StatePredictor_AuxHead.py](../starVLA/model/framework/VLA_MWM_StatePredictor_AuxHead.py)

1-step consistency로 `a_hat = a_tau + (1-t)·v_pred` 재구성 → 작은 MLP(H·action_dim → 128 → 8)가 `a_hat`으로부터 state delta 예측 → **GT delta**로 supervise.
```
L_aux = MSE(aux_head(a_hat), r_target_gt - r_present)
L_total = w_pred·L_pred + w_action·L_flow + λ_aux·L_aux    (λ_aux = 0.1)
```
- Zero-init last layer → safe warmup.
- Deploy 시 aux head 폐기, inference 비용 baseline과 동일.

### 1.3 `_Feedback` — closed-loop physical feedback (Method 1)
파일: [starVLA/model/framework/VLA_MWM_StatePredictor_Feedback.py](../starVLA/model/framework/VLA_MWM_StatePredictor_Feedback.py)

```
a_hat = a_tau + (1-t)·v_pred            # 1-step Euler 재구성
r_hat = r_present + integrator(Σ a_hat) # Linear(7→8), zero-init, no bias
L_feedback = MSE(r_hat, r_target_pred.detach())
L_total = w_pred·L_pred + w_action·L_flow + λ_fb·L_feedback  (λ_fb = 0.1)
```
- Head + integrator만 학습 (predictor는 detach — `L_pred`로만 학습).
- Aux와 차이: target이 **predictor 예측** (GT 아님), 모듈이 fixed linear integrator.

### 1.4 세 변형 비교
|  | Baseline | Aux (M3) | Feedback (M1) |
|---|---|---|---|
| Target | — | `r_target_gt` (data GT) | `r_target_pred` (Mamba 예측) |
| 모듈 | — | 작은 MLP | Linear 7→8, zero-init |
| 역할 | — | Backbone regularizer | Closed-loop consistency |
| Deploy 비용 | baseline | baseline (aux 폐기) | baseline (integrator 폐기) |

## 2. 액션 생성 방식

핵심: **그냥 flow-matching에 조건 토큰을 넣는 것.** DINO 토큰이 있던 자리에 state에서 뽑은 토큰이 들어갈 뿐.

`predict_action` ([코드](../starVLA/model/framework/VLA_MWM_StatePredictor.py#L231-L269)):
1. Qwen action tokens 추출 (present frame).
2. `r_past` (per-episode buffer) + `r_present` (sim에서 수신) → Mamba로 `r_target_pred`.
3. `cond = _make_cond(r_present, r_target_pred)` — K=8 state 조건 토큰.
4. `action_model.predict_action(cond, r_present)` — flow-matching DiT head가 action chunk 생성.
5. `_past_state = r_present` 저장. `episode_reset()`에서 클리어.

**Feedback 변형도 inference는 동일**: `integrator`와 `L_feedback`은 **training-time loss일 뿐**. Inference 경로는 부모 클래스에서 상속받아 그대로 사용하므로 integrator는 forward에서 호출조차 되지 않음. 즉 rollout 중 "예측 state − 실제 state" 오차로 action을 보정하는 고전 제어(PID/MPC/tracking law)가 **아니라**, 학습 시 head gradient에 "네가 뽑은 action chunk를 integrator로 굴려보면 목표 state 근처에 떨어져야 한다"는 regularizer를 넣은 것.

정리:
- **Baseline / Aux**: 조건 토큰만 바뀐 flow-matching. 완전한 open-loop 생성.
- **Feedback**: 학습 loss만 closed-loop 스타일, inference는 여전히 open-loop.

진짜 제어식 피드백(rollout 중 error correction, receding horizon replan)을 원한다면 별도 구현 필요.

## 3. LIBERO 결과 (500 rollout, success rate)

| Suite | StatePredictor | +AuxHead | +Feedback |
|---|---|---|---|
| spatial | **93.4%** | 92.6% | 92.8% |
| object | 98.8% | 98.6% | **99.4%** |
| goal | 94.8% | 90.4% | **96.0%** |
| 10 | 89.0% | **89.4%** | 86.8% |
| **avg** | **94.0%** | 92.75% | 93.75% |

- Baseline state predictor가 평균 최고 (94.0 vs 92.75 / 93.75).
- Feedback은 goal / object에서 최고, long-horizon (10) 에서는 오히려 하락.
- Aux는 전반적으로 이득 없음.
- **참고**: MWM-v2-FO (DINO latent) 메인이 94.2% LIBERO — state로 대체해도 사실상 동등. 가설(“조건은 action-relevant subspace이면 충분”)이 뒷받침되나, aux/feedback closed-loop training loss 시도는 이 스케일에서 이득 없음.

## 4. 고찰 — 세 접근법 각각의 의미와 한계

세 변형 모두 **inference 경로는 동일한 flow-matching**이다. Predictor가 뽑은 `r_target_pred` (7스텝 뒤 endpoint state)를 조건 토큰으로 DiT head에 넣어 action chunk를 생성. 차이는 **학습 시 어떻게 supervise 하느냐**뿐. 아래는 각 접근이 실제로 무엇을 시도한 것인지, 왜 그런 결과가 나왔는지의 해석.

### 4.1 Baseline `StatePredictor` — "조건만 바꿔도 되나?"

**의도**: DINO latent (768-dim × N tokens) 라는 고차원 시각 미래 조건을 **8-dim endpoint state**로 축소해도 되는지의 ablation. 만약 head가 실제로 필요한 조건이 "action-relevant subspace의 goal"이라면, state로도 충분해야 함.

**결과 해석**: 94.0% avg — DINO latent 버전 (94.2%) 과 **사실상 동등**. 이는 head가 요구하는 조건 정보가 실제로 얼마 안 된다는 강한 증거. Qwen action_tokens가 이미 시각/언어 context를 다 나르고 있으니, predictor 경로의 역할은 "**어디로 갈지의 goal 좌표**"만 제공하는 것으로 충분.

**한계 · 열린 질문**:
- Endpoint 하나만 예측 → 접근 궤적, gripper timing, 다중 모드가 조건에 안 담김. 이걸 다 처리하는 건 결국 head (Qwen 조건 위에서).
- Predictor의 실제 부담이 작다는 얘기이기도 함 → predictor 자체를 더 무겁게 만들 유인이 없음.
- "정말 필요한 최소 조건이 뭔지" 를 더 밀면: 오히려 predictor를 없애고 Qwen action_tokens만 조건으로 쓰는 ablation이 다음 단계.

### 4.2 `_AuxHead` — "Head 내부에 물리적 anchoring을 주입"

**의도**: Head가 뽑는 action chunk가 어디로 이어질지를 head 자신이 알도록 강제. Flow-matching의 1-step consistency로 `a_hat`을 재구성하고, 작은 MLP가 그걸로 GT state delta를 예측. Gradient가 v_pred → head 내부까지 흘러들어감. Deploy에는 안 씀 (regularizer only).

**결과 해석**: 평균 92.75%, baseline보다 낮음. 즉 이 anchoring은 **이득이 없거나 미세하게 해로움**.
- 원인 추정: LIBERO delta-EEF action space에선 `a_hat`과 state delta의 관계가 이미 거의 선형이라 GT delta를 맞추는 게 head에 새로운 정보를 주지 않음. 오히려 "물리 재구성"을 잘하려는 pressure가 데모 스타일 (velocity profile, 접근 shape) 을 뭉갤 수 있음.
- 특히 `goal` (94.8 → 90.4) 에서 크게 떨어진 게 시사적 — goal 태스크는 여러 다른 궤적이 유효한 multi-modal 상황이 많은데, GT delta anchoring이 mode-average 쪽으로 밀었을 가능성.

**한계**: Anchoring target이 GT라서 predictor와 무관 → 이 loss는 "predictor를 활용하는 실험"이 아니라 "head에 물리 regularizer 다는 실험". 별개 얘기임.

### 4.3 `_Feedback` — "예측 goal에 도달하는 action 인가?"의 학습 시 검산

**의도**: Predictor가 뽑은 `r_target_pred` 를 실제 목표로 삼고, head가 뽑은 action chunk를 tiny linear integrator로 굴렸을 때 정말 그 목표에 도달하는지 MSE로 supervise. Loss는 physical unit (m/rad). Head + integrator만 학습, predictor는 detach.

**결과 해석**: 평균 93.75%, baseline과 큰 차이 없음. Detail을 보면:
- `object` 99.4%, `goal` 96.0% — **baseline보다 명확히 상승**. 이 태스크들은 goal이 비교적 잘 정의되고, 접근 궤적의 shape이 단순해서 "goal에 정말 도달하는 action" pressure가 그대로 이득이 됨.
- `10` (long-horizon) 89.0 → 86.8 **하락**. Long-horizon에선 여러 sub-goal이 있고, predictor가 뽑는 endpoint가 애매하거나 노이지 → 그 잘못된 goal에 도달하려는 pressure가 오히려 해로움. Feedback은 predictor 오차에 취약한 구조.

**한계 · 이론적 의미**:
- Integrator가 `Linear(7→8, zero-init, no bias)` — 학습 가능한 1-step tracking model. 데이터로 최적화 가능한 이 tracker가 성능을 크게 못 올렸다는 점이 중요: **(start, endpoint) → action_chunk 매핑은 순수 물리학으로 안 풀린다**는 신호. Gripper 타이밍, 접근 shape 같은 정보가 endpoint에 없음.
- 즉 "학습 시에 물리 제약을 힌트로 준다" 는 아이디어가, endpoint-only setup에선 물리적으로 부족한 정보로 힌트를 주는 셈. Waypoint sequence까지 predictor가 뽑으면 이 프레임이 살아날 여지 있음.

### 4.4 종합

- **State가 조건으로 충분하다** — Baseline이 증명. 이건 강한 결과.
- **Head에 물리 regularizer** (Aux) — 이 스케일에선 이득 없음. Anchoring이 데모 스타일과 충돌하는 방향으로 작용할 수 있음.
- **예측 goal에 대한 학습 시 tracking loss** (Feedback) — Task-dependent. Goal이 잘 정의된 태스크(object, goal)엔 +, 애매한 태스크(long-horizon 10)엔 −. 평균으로는 무의미.
- **공통 병목**: predictor가 endpoint 하나만 예측 → 접근 궤적, gripper timing, 다중 모드는 여전히 head가 flow-matching으로 흡수. Predictor를 waypoint sequence + gripper schedule 예측기로 확장하면 세 프레임(특히 Feedback + 진짜 MPC) 모두가 다시 살아날 여지.
- **Inference는 셋 다 open-loop**. 학습 시 loss만 다름. 진짜 closed-loop 제어(rollout error correction, receding horizon) 는 아직 없음.
