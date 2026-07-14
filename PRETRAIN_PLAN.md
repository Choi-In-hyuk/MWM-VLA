---
title: Streaming Mamba World Model — SSv2 + Droid Pretraining → LIBERO Fine-tune Plan
status: planning
last_updated: 2026-07-08
---

# 개요

Streaming Mamba World Model (v2, 2-frame stateless)를 SSv2 + Droid 대규모 비디오
데이터로 **predictor pretraining** 하고, 이후 LIBERO에서 **action-conditioned
fine-tune**하여 LIBERO 4-suite에서 평가한다. (LIBERO-Plus는 후속 과제)

VLA-JEPA 파이프라인의 철학(대규모 self-supervised video pretraining → downstream
action tuning)을 따르되, 우리 architecture(streaming Mamba-2 predictor + Qwen3-VL
+ DINOv2 + Flow-matching DiT)에 맞게 조정한다.

# 1. 큰 그림 (2-stage)

```
Stage A: Pretraining  (SSv2 + Droid, no action)
   ├─ Data: 이미지 3-frame [t-H, t, t+H] + language 만 사용
   ├─ Trainable: Mamba predictor (from scratch) + Qwen LoRA (r=16, warm-start)
   ├─ Frozen: DINOv2, Qwen backbone, (action head 없음)
   ├─ Loss: L_pred = L1(s_target_pred, s_{t+H}) + cos regularizer
   └─ Output: pretrained predictor + adapted Qwen LoRA

Stage B: LIBERO Fine-tune  (기존 stream_libero_all과 동일)
   ├─ Data: LIBERO 4-suite LeRobot (state, action, images, instruction)
   ├─ Init: Stage A ckpt (predictor + LoRA)
   ├─ Trainable: predictor + LoRA + cond_proj + action head
   ├─ Frozen: DINOv2, Qwen backbone
   ├─ Loss: L_pred + L_action
   └─ Output: final ckpt → LIBERO 4-suite eval
```

# 2. 데이터셋

## 2.1 SSv2 (Something-Something v2)
- 위치: `/mnt/4TB_2/jamvla/datasets/ssv2/` (37 GB, 220,847 webm)
- 구조: `<id>.webm` + `labels/{train,validation}.json` (label class text)
- 언어 조건: class name (e.g. "Moving something up") 을 instruction으로 사용
- **state / action 없음** → predictor pretraining 전용
- Frame rate: 12 fps

## 2.2 Droid (lerobot/droid_1.0.1)
- 위치: `/mnt/4TB_2/jamvla/datasets/DroidLerobot/` (412 GB)
- 구조: `data/chunk-*/file-*.parquet` + `videos/{key}/chunk-*/file-*.mp4`
- Video keys 3개: `observation.images.exterior_1_left`, `exterior_2_left`, `wrist_left`
  - 학습에는 **exterior_1_left + wrist_left** 2개만 사용 (LIBERO와 시점 수 맞춤)
- 언어 조건: `language_instruction` (자연어 태스크 설명)
- **state / action 은 사용하지 않음** (action-space mismatch 회피)
- Frame rate: 15 fps

## 2.3 왜 두 데이터셋만?
- VLA-JEPA와 동일 조합 → 직접 비교 가능
- SSv2 = human hand motion + 다양한 언어 (action semantics 학습)
- Droid = robot manipulation (LIBERO 도메인 근접)
- SSv2:Droid 비율은 대략 **1:1 sample-level mixing** (VLA-JEPA와 유사)

# 3. Predictor 입출력 (Stage A/B 동일)

기존 streaming v2 구조를 그대로 사용:

| slot | 의미 | 크기 |
|---|---|---|
| `action_tokens` | Qwen3-VL의 언어+비전 조건 토큰 (현재 frame + instruction) | [B, Na, 2048] |
| `s_past`  | DINOv2 latent of frame `t-H` | [B, 256, 768] |
| `s_present` | DINOv2 latent of frame `t` | [B, 256, 768] |
| `query` | learnable query for `t+H` prediction | [B, 256, D] |
| SSM `states` | 사용 안 함 (stateless streaming) | — |

Sequence layout per chunk:
```
[ action(Na) | obs_past(256) | obs_present(256) | query(256) ]  ~= 792 tokens
```

- 2-frame obs에 per-frame `time_emb` 추가 (frame 0 / frame 1 구분)
- Output: `s_target_pred` = DINOv2 latent of frame `t+H` 예측 [B, 256, 768]

**Frame indexing (H=7 유지)**
- LIBERO와 같은 H=7로 통일 (fine-tune 정합성)
- SSv2/Droid에도 obs_indices = `[-7, 0, +7]` 적용
- Cold-start (base<H): 기존 dataloader front-padding 사용 (frame 0 복제)

# 4. Trainable / Frozen 정책 (핵심 결정)

## Stage A (pretraining)
| 컴포넌트 | 상태 | LR |
|---|---|---|
| DINOv2 ViT-B | frozen | — |
| Qwen3-VL-2B backbone | frozen | — |
| **Qwen LoRA (r=16, α=32)** — VLA-JEPA-LIBERO.pt에서 warm-start | **train** | 1e-5 |
| **Mamba predictor (from scratch, ~134M)** | **train** | 1e-4 |
| cond_proj | 사용 안 함 | — |
| Action head | 사용 안 함 | — |

**LoRA를 왜 켜는가**
- Qwen이 원래 학습된 predictor는 V-JEPA2 style (multi-frame → multi-frame block masking)
- 우리 Mamba predictor는 2-frame → 1-frame stateless → 언어 토큰 소비 방식 다름
- LoRA로 언어 표현을 새 predictor 문법에 맞게 미세 재정렬 (full FT는 2 GPU에 무리)
- LoRA weight 초기화: **VLA-JEPA-LIBERO.pt에서 상속** (LIBERO 편향이 어차피 도움됨)

## Stage B (LIBERO fine-tune)
| 컴포넌트 | 상태 | LR |
|---|---|---|
| DINOv2 | frozen | — |
| Qwen backbone | frozen | — |
| Qwen LoRA | train | 1e-5 |
| Predictor | train | 1e-4 |
| cond_proj | train (fresh init) | 1e-4 |
| Action head (Flow-matching DiT) | train (fresh init) | 1e-4 |

L_pred + L_action 병렬 (기존 stream_libero_all stage2와 동일).

# 5. Mamba predictor 크기

**현행 유지: depth=12, state_dim=1024, d_state=64, d_conv=1, headdim=64, chunk_size=64 (~134M)**

근거:
- VLA-JEPA V-JEPA2 predictor(~161M)와 근접
- LIBERO 단독 학습에서 이미 pred_cos 0.93+ 달성 → 표현력 부족 미검출
- 2 GPU × per-gpu bs 16 batch 확보 유리 (크기 키우면 batch 감소 → SNR 나빠짐)
- 데이터가 5–20배 늘어나는 게 오히려 크기보다 신호에 큰 영향 (먼저 실험 후 필요 시 확대)

# 6. Loss

## Stage A
```
L_pred = L1(s_pred, s_{t+H}.detach()) + α * (1 - cos(s_pred, s_{t+H}.detach()))
```
- α = 1.0 (기존과 동일)
- Target은 DINOv2 forward (`s_{t+H}`) 에서 detach

## Stage B
```
L_total = λ_pred * L_pred + λ_action * L_action_flow_matching
```
- λ_pred = 1.0, λ_action = 1.0 (기존과 동일)

# 7. 학습 하이퍼파라미터

## Stage A (pretraining, SSv2 + Droid)
| 항목 | 값 |
|---|---|
| optimizer | AdamW |
| LR (predictor) | 1e-4 |
| LR (Qwen LoRA) | 1e-5 |
| scheduler | cosine + warmup |
| warmup steps | 5000 |
| **max_steps** | **50000** (VLA-JEPA와 동일 budget) |
| per-gpu batch | 16 |
| DDP | 2 GPU |
| effective batch | 32 |
| grad clip | 1.0 |
| precision | bf16 |
| log_every | 50 |
| save_every | 5000 |
| SSv2:Droid mix | 1:1 (sample level) |

## Stage B (LIBERO fine-tune, libero_all)
| 항목 | 값 |
|---|---|
| LR (predictor + LoRA + head + cond_proj) | 1e-4 / 1e-5 (LoRA만) |
| max_steps | 30000 (VLA-JEPA fine-tune budget과 유사) |
| warmup | 2000 |
| per-gpu batch | 16 |
| DDP | 2 GPU |
| resume_ckpt | `results/pretrain_ssv2_droid/checkpoints/predictor_final.pt` |
| 나머지 | 기존 stream_libero_all stage2와 동일 |

# 8. 파일 구조 (예정)

## 새로 만들 것
```
starVLA/dataloader/
  ssv2_dataset.py                 # SSv2 webm loader (video + class text)
  droid_pretrain_dataset.py       # Droid parquet+mp4 loader (video + language only)
  pretrain_mixer.py               # SSv2 + Droid 1:1 mixed sampler

scripts/
  pretrain_predictor_ssv2_droid.sh   # Stage A launcher (DDP x2)
  finetune_libero_from_pretrain.sh   # Stage B launcher (기존 stream_libero_all
                                     #   재활용 + --resume_ckpt)

PRETRAIN_PLAN.md                  # 이 파일
```

## 재사용
```
starVLA/model/framework/VLA_DINO_StreamingMamba.py  # 그대로
starVLA/model/modules/world_model/mamba_world_model.py  # 그대로
scripts/train_mamba_wm.py         # --stage predictor 재활용
                                  # (SSv2/Droid loader만 새로 붙임)
```

# 9. Dataloader 최소 스펙

## SSv2 dataset
- Input: video_path, class_id → class text (`labels.json` lookup)
- Output dict:
  - `images`: [3, H, W, 3] (3 frames at `[t-H, t, t+H]`, 각각 2 view이지만 SSv2는 single-view → 복제하거나 dummy 처리)
  - `instruction`: str (class name)
  - `dataset`: "ssv2"
- Sampling: episode 안에서 random base t 뽑고 `[t-H, t, t+H]` 로드. cold-start면 front-pad.

## Droid pretrain dataset
- Input: parquet row + mp4 (LeRobot layout)
- Output dict:
  - `images`: [3, H, W, 3, 2] (3 frames × 2 view: exterior_1_left + wrist_left)
  - `instruction`: str (`language_instruction`)
  - `dataset`: "droid"
- **state, action 필드 무시**

## Mixer
- SSv2, Droid 각각 IterableDataset → `InterleaveIterableDataset` 1:1
- SSv2 sample은 wrist view가 없으므로 dummy zero image로 채우거나 exterior view 복제
- **논쟁 소지**: SSv2/Droid view 수가 다른 문제 → 아래 결정 필요

### 미해결 이슈: View 수 불일치
- LIBERO fine-tune은 2 view (image + wrist_image)
- Droid pretrain은 2 view (exterior_1_left + wrist_left) OK
- SSv2 pretrain은 1 view만 있음 → **wrist를 exterior 복제**로 처리 (single-view fallback)
- 이 처리로 실제 성능 손해 있을 수 있으나 SSv2 손실 감수 (VLA-JEPA도 유사 처리 추정)

# 10. 평가 계획

## LIBERO 4-suite (Spatial/Object/Goal/LIBERO-10)
- 50 trial × task
- 기존 `scripts/eval_libero_all.sh` 재사용
- Baseline 비교:
  - `v2lora_libero_all` (LoRA baseline, 93.95%)
  - `stream_libero_all` (streaming v2 no-pretraining, 93.7%)
  - **`stream_pretrain_ssv2_droid → libero_all`** (target)

## LIBERO-Plus (후속, 이번 스코프 밖)
- assets 설치는 완료 (RESEARCH.md 참조), 실제 실행은 이번 단계에서 하지 않음
- LIBERO 4-suite 결과 확보 후 별도 단계에서 진행 예정

# 11. 실행 순서

1. **Dataloader 개발**
   - [ ] Droid minimal loader (`droid_pretrain_dataset.py`)
   - [ ] SSv2 loader (`ssv2_dataset.py`)
   - [ ] Mixer (`pretrain_mixer.py`)
   - [ ] Sanity: 각각 1 batch 뽑아 shape/dtype/instruction 확인
2. **Stage A pretraining**
   - [ ] `pretrain_predictor_ssv2_droid.sh` 작성
   - [ ] 500 step smoke test (DDP 정상, loss 감소, 메모리 OK 확인)
   - [ ] 50k step full run → pred_cos, pred_loss 로그
3. **Stage B fine-tune**
   - [ ] `finetune_libero_from_pretrain.sh` (기존 stream_libero_all 기반 + `--resume_ckpt`)
   - [ ] 30k step full run
4. **평가**
   - [ ] LIBERO 4-suite (50 trials/task)
   - [ ] 결과를 `STREAM_V2_REPORT.md` / `RESEARCH.md`에 추가

# 12. 리스크 & 완화

| 리스크 | 완화 |
|---|---|
| Droid mp4 로딩 IO bottleneck (412 GB) | `num_workers=8`, PyAV lazy decode, 필요시 subset caching |
| SSv2 single-view mismatch | wrist=exterior 복제로 우회, 성능 저하 감수 |
| LoRA 학습 불안정 (SSv2 도메인 shift) | LR 1e-5 (매우 낮게), warmup 5000, grad clip 1.0 |
| Pretrain 후 LIBERO 오히려 저하 | ablation: pretrained 없이 vs 있는 것 둘 다 학습 후 비교 |
| 50k step이 부족 | pred_cos plateau 확인 후 필요 시 연장 |

# 13. 의도적으로 **하지 않는** 것

- Droid action supervision (action-space mismatch, gripper convention 이슈 회피)
- Droid state / normalization / gripper 처리 (predictor에 안 쓰이므로 불필요)
- Qwen full FT (2 GPU 불가, LoRA로 대체)
- Predictor 크기 증대 (현재 134M로 충분한 근거)
- View 수 확장 (2 view 유지)
- SSv2 label을 rich prompt로 augment (그냥 class text 사용)

# 14. 완료 판정

이 pretraining이 성공했다 판정 조건:
- Stage A: pred_cos ≥ 0.90 (SSv2+Droid 혼합에서)
- Stage B: LIBERO 4-suite Avg ≥ **94.5%** (기존 stream 93.7% 대비 +0.8%p 이상)

# 관련 문서
- [DESIGN_STREAMING_WM.md](DESIGN_STREAMING_WM.md) — streaming v2 아키텍처 v1–v4 히스토리
- [STREAM_V2_REPORT.md](STREAM_V2_REPORT.md) — libero-only 학습 결과
- [RESEARCH.md](RESEARCH.md) — 전체 프로젝트 방향 / TODO / dead ends
- [README.md](README.md) — repo 개요
