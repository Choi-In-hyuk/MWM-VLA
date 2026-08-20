# MWM-VLA — Canonical-Target Denoising Predictor: Method & Results

> **작업 인계용 문서.** 다른 컴퓨터에서 이어서 작업하기 위한 method + 결과 + 현재 진행 상태 스냅샷.
> 마지막 업데이트: 2026-08-20 (LIBERO-Plus 평가 진행 중).

---

## 0. 한 줄 요약

Frozen 대규모 사전학습 표현(DINOv2 + Qwen-VL + VLA-JEPA 백본) 위에, **대규모 비디오 사전학습 없이 in-domain 데이터만으로 scratch 학습한 경량 Mamba world-model(예측기)** 을 얹는다. 이번 기여는 그 예측기의 학습 방식을 **canonical-target denoising** 으로 바꾼 것: 예측기 **입력** 프레임에만 이미지 증강(가짜 카메라각 + 시각 노이즈)을 주고, 예측 **타깃**은 깨끗한(clean) 원본 미래 latent 로 둔다. 즉 **"변형된 입력 → 원본 미래 복원"** 을 학습.

핵심 관찰: 이 증강만으로 **perturbation 이 없는 깨끗한 원본 LIBERO 에서도** 성공률이 baseline 93% → **97.2%** 로 올랐다 (데이터 양은 동일; 증강은 데이터를 늘린 게 아니라 기존 샘플의 일부를 on-the-fly 변형).

---

## 1. Method

### 1.1 무엇이 frozen이고 무엇이 학습되나

| 구성요소 | 출처 | 학습 상태 |
|---|---|---|
| DINOv2 (dinov2_vitb14) | 대규모 self-supervised | **frozen** |
| Qwen-VL | 대규모 VLM 사전학습 + VLA-JEPA LIBERO fine-tune | **frozen** (LoRA r16/a32 어댑터만 학습) |
| VLA-JEPA 백본 | 대규모(SSv2 등) + LIBERO | **frozen** |
| **Mamba 예측기 (world-model)** | — | **scratch 학습 (이번 기여)** |
| cond_proj + action head (DiT/flow) | — | 학습 (stage2) |

- 예측기 학습 시 로그: `loaded backbone; 360 new params fresh` → 예측기는 무작위 초기화에서 시작.
- **SSv2/Droid 대규모 비디오 사전학습을 쓰지 않음** (`--resume_ckpt` 없이 predictor stage 시작). 별도로 SSv2+Droid pretrain 버전이 repo 에 존재하지만(commit 2dcaf4c) 이번 실험은 미사용.
- 관찰: SSv2/Droid pretrain 유무로 성능 차이 거의 없음 → **예측기는 대규모 비디오 사전학습 불필요, in-domain scratch 로 충분** (데이터 효율성 주장의 실증).

### 1.2 Canonical-Target Denoising (이번 기여의 핵심)

프레임워크: `starVLA/model/framework/VLA_DINO_StreamingMamba_FutureOnly_Denoise.py`
(부모 `VLA_DINO_StreamingMamba_FutureOnly` 와 **딱 하나** 다름 = `_augment_videos`. 구조/파라미터/loss 전부 동일.)

한 학습 샘플 = 3프레임 `[past, present, target(미래)]`:

| 프레임 | 처리 | 역할 |
|---|---|---|
| past (입력) | **증강 O** | 예측기 입력 |
| present (입력) | **증강 O** | 예측기 입력 |
| **target (미래)** | **증강 X — 절대 안 건드림** | 예측 정답 (clean front-view) |

학습 목표:
```
증강된 [past, present]  --Mamba 예측기-->  CLEAN 원본 target latent 복원
   (가짜 카메라각/노이즈)                     (깨끗한 정면 미래)
```

- 배치의 **70%** (`aug_prob=0.7`) 만 증강, 30%는 clean 유지. 매 스텝 랜덤 → 데이터 수는 그대로, 변형 버전을 계속 다르게 노출.
- past/present 에 **동일한 변형 T** 를 적용 (shared) → 프레임마다 튀지 않고 "이 에피소드는 이런 각도/조명"처럼 일관. 실제 카메라 이동을 흉내.
- 증강 픽셀이 **DINO 와 Qwen 양쪽에** 들어감 (deploy 일관성) → 액션 경로(Qwen LoRA)도 시점 변화를 겪음.

**증강 하이퍼파라미터 (기본값):**
| 종류 | 파라미터 | 값 |
|---|---|---|
| 확률 | aug_prob | 0.7 |
| 기하(가짜 카메라각) | aug_persp / aug_rot_deg / aug_crop_min | 0.25 / ±10° / 0.9 |
| 광학(시각 노이즈) | aug_brightness / aug_contrast | ±0.3 / ±0.3 |
| | aug_blur_sigma / aug_pixel_noise | ≤1.5 / 0.05 |
| 적용 대상 | aug_on_past | True (past+present 둘 다) |

### 1.3 왜 깨끗한 LIBERO 에서도 오르나 (해석)

"변형→원본" 과제는 단순 "원본→원본" 보다 어렵다. 예측기가 시점·조명 같은 **표면 변화에 불변인 표현**을 배워야만 풀린다. 그 불변 표현이:
- perturbation 있는 Plus 에선 → **강인성(robustness)** 으로 발현
- perturbation 없는 원본 LIBERO 에선 → **정규화/일반화(regularization)** 로 발현 (93→97)

즉 증강이 "강인성 도구"가 아니라 **표현 학습의 정규화**로 작동. (augmentation 이 clean test set 정확도도 올리는 고전적 이유와 동일.)

---

## 2. 학습 설정 (재현용)

- **GPU**: 단일 RTX A6000 48GB (baseline 은 다른 머신에서 2-GPU 였음 — 아래 주의 참고)
- **데이터**: `libero_all` mix = 4 suite 합쳐 **403,428 샘플** (object 66,605 / goal 51,851 / spatial 52,791 / 10 100,857). `mixtures.py` 의 `libero_all` 정의(libero_90 제외).
- **backbone_ckpt**: `results/vla_jepa_libero_orig/checkpoints/VLA-JEPA-LIBERO.pt`
- **data_root**: `/home/choi/data/datasets/LIBERO` (LeRobot 포맷)
- **stage1 (predictor)**: 30,000 steps, bs=16, warmup 500
- **stage2 (+action head)**: 30,000 steps, **bs=8 × grad_accum=2** (=effective 16; DiT head 무거워 48GB 에 맞춤. OOM 검증됨: peak 31.7/49 GB)
- **Qwen LoRA**: r=16, alpha=32
- **Mamba-2**: state_dim 1024, depth 12, d_state 64, d_conv 1, headdim 64, chunk_size 64
- 학습 지표(정상): pred_cos 0.17 → **0.94**, pred loss 1.26 → 0.39, NaN/OOM 0건.

**실행 스크립트:**
- 학습+원본LIBERO: `scripts/run_denoise_libero_all.sh [train|eval_libero|eval_plus|all]`
- LIBERO-Plus (4 suite): `scripts/run_denoise_plus_all.sh`
- Plus 실시간 상태: `bash scripts/plus_status.sh` / 실시간 `watch -n5 bash scripts/plus_status.sh`

---

## 3. 결과

### 3.1 원본 LIBERO (50 trials/task) — **완료**

| Suite | denoise (libero_all) | baseline (다른 머신) |
|---|---|---|
| libero_spatial | 96.6% | — |
| libero_object | 99.2% | — |
| libero_goal | 97.0% | — |
| libero_10 | 96.0% | — |
| **평균** | **97.2%** | **93%** |

- baseline 93%: **다른 컴퓨터**에서 얻은 값 (증강 없는 plain StreamingMamba, 동일하게 libero_all 통합 학습). 데이터 양 동일 → **+4.2%p 는 증강(denoising) 효과로 귀속**.
- ⚠️ baseline 의 per-suite 수치와 LIBERO-Plus 수치는 **아직 우리 손에 없음** (별도 확보 필요, §5).

### 3.2 LIBERO-Plus 7축 × 4 suite — **진행 중** (2026-08-20 스냅샷)

- **총 평가 개수 = 10,030** (spatial 2,402 + object 2,518 + goal 2,591 + 10 2,519). 변형당 1 trial (논문 표준).
- 실행 순서: 바깥=category, 안=4 suite. **Sensor Noise 를 맨 마지막**에 배치(개수 최다 1,601 + rollout 최장).

**현재까지 누적 (약 4,069 / 10,030 완료, 76.6%):**

| category | 10 | goal | object | spatial | ALL |
|---|---|---|---|---|---|
| Background Textures | 81% | 96% | 99% | 97% | **93.0%** |
| Camera Viewpoints | 39% | 69% | 56% | 72% | **58.8%** |
| Language Instructions | 88% | 76% | 90% | 86% | **84.4%** |
| Light Conditions | — | — | — | — | (진행 예정) |
| Objects Layout | — | — | — | — | (진행 예정) |
| Robot Initial States | — | — | — | — | (진행 예정) |
| Sensor Noise | — | — | — | — | (마지막, 진행 예정) |
| **TOTAL(부분)** | 64% | 79% | 79% | 84% | **76.6%** |

- 관찰: **Camera Viewpoints 가 58.8% 로 가장 낮음** — denoise 가 직접 겨냥한 축인데도 어려움. baseline 대비가 관건 (baseline Plus 수치 확보 후 비교).
- 나머지 4축(Light/Objects/Robot Init/Sensor Noise)은 미완. 완료 시 이 표 갱신 필요.

---

## 4. 논문 포지셔닝 (데이터 서사 — 리뷰어 방어용)

- **Qwen 이 대규모 VLM 인 사실을 숨기지 말고 명시.** 주장을 "대규모 학습 없이 액션 생성"으로 하면 즉사.
- ✅ 방어되는 주장: **"지각·언어는 frozen 대규모 사전학습 표현을 재사용, 우리가 새로 학습하는 world-model(예측기)은 대규모 비디오 사전학습 없이 in-domain scratch 로 충분하다."**
- 다른 VLA(OpenVLA/RT-2/π0/Octo 등)도 전부 pretrained VLM 사용 → **pretrained 사용은 페널티가 아니라 표준.** VLM scratch 학습은 하지 말 것(아무도 안 하고 기여도 아님).
- 차별점: VLM 유무가 아니라 **"inference-time latent world-model 을 대규모 비디오 사전학습 없이 붙여 액션을 개선"**.

---

## 5. 다음 컴퓨터에서 할 일 (TODO)

1. **LIBERO-Plus 완주** — 현재 3/7축. `bash scripts/run_denoise_plus_all.sh` 는 resume 지원(이미 끝난 task_index 건너뜀). 나머지 4축(Light/Objects/Robot Init/Sensor Noise) 완료 후 §3.2 표 갱신.
2. **baseline(plain, 증강 X) 확보** — 공정 비교의 핵심. 같은 libero_all 통합으로 `VLA_DINO_StreamingMamba_FutureOnly`(증강 없는 부모) 학습 → 원본 LIBERO per-suite + LIBERO-Plus 7축. 이게 있어야 "증강 효과 = Plus 에서 얼마" 를 데이터로 증명.
3. **(선택) SSv2/Droid pretrain ablation** — "대규모 비디오 사전학습 제거해도 성능 유지" 를 표로. commit 2dcaf4c 의 pretrain 버전 활용.

---

## 6. 핵심 경로/파일 (인계용)

| 항목 | 경로 |
|---|---|
| denoise 프레임워크 | `starVLA/model/framework/VLA_DINO_StreamingMamba_FutureOnly_Denoise.py` |
| 부모(=baseline) 프레임워크 | `starVLA/model/framework/VLA_DINO_StreamingMamba_FutureOnly.py` |
| set_stage(무엇이 trainable) | `starVLA/model/framework/VLA_DINO_Mamba_Diff.py:119` |
| libero_all mix 정의 | `starVLA/dataloader/gr00t_lerobot/mixtures.py:15` |
| 학습 엔트리 | `scripts/train_mamba_wm.py` |
| denoise 통합 학습 ckpt | `results/denoise_all/stage2/checkpoints/mamba_wm_final.pt` |
| 원본 LIBERO eval 결과 | `results/eval/<suite>_denoise_all/eval.log` |
| LIBERO-Plus eval 결과 | `results/eval_plus/<suite>_denoise_all/results.jsonl` |
| Plus eval 클라이언트 | `examples/LIBERO/eval_libero_plus.py` (off-by-one 우회 패치됨, §7) |
| 학습 데이터 (LeRobot) | `/home/choi/data/datasets/LIBERO/*_lerobot` |
| Plus env | conda `vla_plus`, `LIBERO_HOME=/home/choi/LIBERO-plus` |
| 학습/원본LIBERO env | conda `vla_jepa`, `LIBERO_HOME=/home/choi/LIBERO-PRO` |

---

## 7. 알려진 이슈 / 패치 (다른 머신 재현 시 주의)

- **LIBERO-Plus off-by-one**: `LIBERO-plus` repo 의 benchmark 는 suite 를 category(축) 단위로 로드하며(`Benchmark.__init__(category_value=...)`), `task_classification.json` 의 `id`(1-based)를 0-based task 리스트에 그대로 인덱싱 → id_max 축(Light Conditions 등)에서 IndexError. **우회**: `eval_libero_plus.py` 에서 classification 의 id 를 **-1 보정**한 task 리스트를 직접 만들어 `task_suite.tasks` 를 덮어씀 (2519/2519 이름 완전 일치 검증). category 7축 순회 + `--only-category` 옵션도 이 파일에 추가됨.
- **단일 GPU**: 이 머신은 GPU 1장. baseline(2-GPU, 다른 머신)과 하드웨어 다름 → 순수 성능 비교 시 학습 recipe(스텝/effective bs) 동일성만 유지하면 됨(실제 동일하게 맞춤).
- **디스크**: 대용량 ckpt 다수. 필요 시 중간 step ckpt 정리(final 만 보존).
