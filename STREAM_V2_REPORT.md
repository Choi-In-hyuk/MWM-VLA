# Streaming-Mamba World Model v2 — LIBERO-10 Report

## 요약
- baseline `VLA_DINO_Mamba_Diff` 위에 **2-frame world-model predictor**를 얹은 streaming 구조.
- libero_10 단독 학습 (15k + 15k step), 50 trial/task 평가.
- **Total success rate = 93.2%** (baseline `v2lora_libero_all` libero_10 91.2% 대비 +2.0%p).

## 구조

### Predictor 입력/출력
| 입력 | 무엇 |
|---|---|
| `s_past`  | DINOv2 임베딩 of frame **t−7** ([B, N, 768]) |
| `s_present` | DINOv2 임베딩 of frame **t** ([B, N, 768]) |
| `action_tokens` | Qwen3-VL action tokens, 현재 frame + 언어 instruction ([B, Na, 2048]) |
| SSM `states` | **None** (carry 미사용) |

| 출력 | 무엇 |
|---|---|
| `s_target_pred` | DINOv2 latent of frame **t+7** 예측 ([B, N, 768]) |

Mamba-2 sequence layout per chunk: `[ action(Na) | obs(2*N) | query(N) ]`
- 2-frame obs에 per-frame `time_emb` 추가 (frame 0 / frame 1 구분).
- `d_conv=1`, depth=12, state_dim=1024, d_state=64.
- robot_state는 **predictor에 안 들어감** (action head에만).

### Action head
- 입력: `cond = cond_proj(concat(s_present, s_target_pred))`, robot_state(현재), action GT(학습 시).
- Flow-matching DiT.
- 출력: 7개 action chunk.

### 학습/추론 alignment
- dataloader `obs_indices = [-H, 0, H]` (H=7). 한 sample = [base-7, base, base+7].
- action = `action[base : base+H]` (현재부터 7개).
- base<H인 sample은 dataloader front-padding으로 frame 0 복제 → cold-start case 자연 학습.
- 추론 주기 = 7. predict_action 호출 시 내부 `_past_frame` 캐시에서 t-7 frame 가져옴, 첫 호출(t=0)은 past=present로 cold.

## 학습 설정
| 항목 | 값 |
|---|---|
| dataset | libero_10 (단독) |
| stage1 | predictor만 학습 (L_pred), 15000 step |
| stage2 | predictor + cond_proj + action_model + Qwen LoRA, L_pred + L_action, 15000 step |
| per-gpu batch | 16 |
| DDP | 2 GPU |
| effective batch | 32 |
| warmup | 500 |
| LoRA | r=16, alpha=32 (q/k/v/o_proj) |
| backbone | VLA-JEPA-LIBERO.pt (frozen 외 LoRA만 학습) |

## 학습 metrics (last step)
| stage | pred_cos | pred_loss (L1) | action_loss |
|---|---|---|---|
| stage1 (predictor) | 0.9124 | 0.4711 | — |
| stage2 (joint) | 0.9242 | 0.4384 | 0.0104 |

## 평가 결과 (50 trial/task = 500 episodes per suite)
| 모델 | 학습 데이터 | budget (stage1+2) | Spatial | Object | Goal | LIBERO-10 | Avg |
|---|---|---|---|---|---|---|---|
| baseline `v2lora_libero_all` | libero_all | 50k+50k | 92.8 | 99.8 | 92.0 | 91.2 | 93.95 |
| **streaming v2 (libero_10 단독)** | libero_10 | 15k+15k | — | — | — | **93.2** | — |
| streaming v2 (libero_all, 25k+25k) — 폐기 | libero_all | 25k+25k | 93.4 | 98.8 | 92.4 | 89.8 | 93.6 |
| **streaming v2 (libero_all, 50k+50k)** — 진행 예정 | libero_all | 50k+50k | TBD | TBD | TBD | TBD | TBD |

### 25k+25k 시도 (폐기)
- libero_all 5배 데이터인데 baseline과 같은 학습 budget으로 비교하려고 25k로 줄여서 시도.
- 결과: baseline 93.95% vs 우리 93.6% (Δ −0.35) — noise 범위.
- 문제: 25k step × eff bs 32 = 800k sample / 500k 데이터 → 약 **1.6 epoch만**. libero_10 단독 (8 epoch)보다 학습 부족 가능성.
- 결정: baseline과 같은 50k+50k step으로 재학습.

## 모델 크기
| 모듈 | 총 파라미터 | trainable |
|---|---|---|
| Qwen3-VL backbone | 2134.0 M | 6.4 M (LoRA) |
| DINOv2 ViT-B | 86.6 M | 0 |
| V-JEPA encoder (미사용, ckpt 포함) | 326.0 M | 0 |
| V-JEPA predictor (미사용) | 161.6 M | 0 |
| Mamba predictor (streaming) | 133.7 M | 133.7 M |
| cond_proj | 1.6 M | 1.6 M |
| Flow-matching action head | 155.2 M | 155.2 M |
| **TOTAL** | **약 3.0 B** | **296.9 M** |

추론에 실제 사용되는 파라미터만 합치면 약 **2.5 B** (V-JEPA 모듈 제외).

## 비교 대상 (참고)
| 모델 | LIBERO Spatial | Object | Goal | LIBERO-10 | Avg |
|---|---|---|---|---|---|
| VLA-JEPA (논문) | 96.2 | 99.6 | 97.2 | 95.8 | 97.2 |
| baseline `v2lora_libero_all` (우리) | 92.8 | 99.8 | 92.0 | 91.2 | 93.95 |
| **streaming v2 (이번 실험, libero_10만)** | — | — | — | **93.2** | — |

## 파일 경로
- 학습 ckpt: `results/stream_libero10/stage2/checkpoints/mamba_wm_final.pt`
- 학습 로그: `results/stream_libero10/{pred,stage2}/train.log`
- 평가 로그: `results/eval/libero_10_stream_v2/eval.log`
- 학습 스크립트: `scripts/run_streaming_libero10.sh`
- 평가 스크립트: `scripts/eval_libero_dino.sh`
- 프레임워크: `starVLA/model/framework/VLA_DINO_StreamingMamba.py`
- predictor: `starVLA/model/modules/world_model/mamba_world_model.py` (`StreamingMambaPredictor`)
