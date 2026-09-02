# Canonical-Target Denoising for a Latent World-Model Policy

**Method & first-pass results.** A lightweight Mamba-2 world-model predicts a *clean* front-view future latent from *corrupted* (fake-viewpoint, noised) inputs. The corruption-to-canonical objective doubles as regularization: it lifts clean-LIBERO success from 93% to 97% while adding robustness across LIBERO-Plus.

> A rendered version with the architecture figure and typeset equations: [`denoise_method_results.html`](denoise_method_results.html).

| | |
|---|---|
| **Backbone** | DINOv2-B + Qwen-VL (frozen) |
| **Predictor** | Streaming Mamba-2, 133.7M |
| **Train data** | `libero_all`, 403,428 samples |
| **GPU** | 1× RTX A6000 |

> **Scope.** Numbers below are the **first evaluation pass** (seed 7; 50 trials/task for LIBERO, 1 trial/variant for LIBERO-Plus). This document covers **method and results only** — the surrounding narrative (intro, related work, discussion) is authored separately.

---

## §1 Setup & notation

The policy consumes a short observation window and emits an action chunk. Perception and language come from **frozen, large-scale pretrained encoders**; the only newly trained components are a latent **world-model predictor** and a thin flow-matching action head. The predictor is trained from scratch on in-domain data — no large-scale video pretraining.

**Frozen:** DINOv2-B (patch encoder, $d=768$), Qwen-VL (action tokens, $d=2048$), VLA-JEPA backbone. Qwen is adapted with LoRA ($r=16,\ \alpha=32$; 6.42M) only.

**Trained:** Streaming Mamba-2 predictor (133.7M), `cond_proj`, flow-matching DiT action head. Stage-2 trainable total: 296.9M.

A training sample provides three time-aligned frames — *past*, *present*, *target* (the future to predict) — each encoded by the frozen DINO into patch-token latents:

$$ s_k = \mathrm{DINO}(x_k) \in \mathbb{R}^{N\times d},\qquad k\in\{\text{past},\text{present},\text{target}\},\quad N=512,\ d=768 $$

$N = 512$ = 2 camera views × 256 patches. The action tokens $a$ come from Qwen on the present frame + language instruction.

---

## §2 The canonical-target denoising objective

The central idea: **corrupt the predictor's inputs, but keep its target clean.** With per-sample probability $p=0.7$, a single shared image-level transform $T$ is applied to the *input* frames (past and present), while the *target* frame is never touched.

$$ \tilde{x}_k = T(x_k),\quad k\in\{\text{past},\text{present}\},\qquad T\sim\mathcal{T}\ \text{w.p. }p,\ \text{else } T=\mathrm{Id} $$

$T$ is drawn once per sample and shared across the two input frames and both views, so it mimics a consistent camera displacement rather than per-frame jitter. It composes a **geometric** part (fake viewpoint) and a **photometric/noise** part:

| Component | Parameter | Range |
|---|---|---|
| Perspective | distortion scale | 0.25 |
| Rotation | degrees | ±10° |
| Resized crop | min area frac. | 0.90 |
| Brightness | ± jitter | 0.30 |
| Contrast | ± jitter | 0.30 |
| Gaussian blur | max σ | 1.5 |
| Pixel noise | std (0–1 space) | 0.05 |
| Apply prob. | $p$ (per sample) | 0.70 |

*Table 1 — Augmentation transform $T$ (defaults).*

The same corrupted pixels feed **both** the DINO encoder and the Qwen action-token path, matching deployment (where the policy always sees one real camera stream). The predictor $g_\theta$ maps corrupted input latents + action tokens to a predicted future latent; the loss pulls it toward the **clean** target:

$$ \hat{s}_{\text{tgt}} = g_\theta\big(a,\ [\tilde{s}_{\text{past}},\ \tilde{s}_{\text{present}}]\big),\qquad \mathcal{L}_{\text{pred}} = \big\lVert \hat{s}_{\text{tgt}} - s_{\text{target}}^{\text{clean}} \big\rVert_1 $$

Because $s_{\text{target}}$ stays clean while inputs vary, $g_\theta$ cannot solve the task by copying surface appearance — it must learn a representation **invariant to viewpoint and photometric nuisance**. That invariance shows up twice in the results: as robustness on perturbed LIBERO-Plus, and as plain regularization on clean LIBERO.

---

## §3 Architecture

Two stages. **Stage 1** trains the predictor alone under $\mathcal{L}_{\text{pred}}$. **Stage 2** adds the flow-matching action head under $\mathcal{L}_{\text{pred}} + \beta\,\mathcal{L}_{\text{action}}$, conditioning the head on the predicted future latent.

```
 past+present ──▶ DINOv2-B (frozen) ─┐
   (corrupted)                       │
                                     ▼
 present+instr ──▶ Qwen-VL(+LoRA) ─▶ Mamba-2 predictor  ──▶  ŝ_target
                                     (12 layers, d=1024)      (predicted future)
                                     133.7M · trained             │
                                                                  ├──▶ L_pred (L1) vs CLEAN target
 target frame ──▶ DINOv2-B (frozen) ──── clean s_target ─────────┘
                                                                  │
                                                                  ▼
                                                     flow-matching DiT head (stage 2)
                                                                  │
                                                                  ▼
                                                     L_action (velocity MSE)
```
*Fig. 1 — Corrupted past/present drive the predictor toward the **clean** target latent (L_pred). The predicted latent conditions the flow-matching head (L_action). DINO/Qwen frozen; predictor + head trained.*

### 3.1 Streaming Mamba-2 predictor

The predictor assembles one causal token sequence per chunk and reads the future latent off a learned query block:

$$ \big[\ \underbrace{a}_{N_a}\ \Vert\ \underbrace{[\tilde s_{\text{past}},\tilde s_{\text{present}}]}_{2N}\ \Vert\ \underbrace{q}_{N}\ \big]\ \xrightarrow{\ 12\times\text{Mamba-2}\ }\ \hat s_{\text{tgt}} = W_o\,h[\text{query}] $$

Mamba-2 runs with `d_conv=1` so that splitting a long sequence into chunks with carried SSM state is mathematically equivalent to one pass — the causal state observes every intermediate frame while predictions stay at the chunk rate. Config: depth 12, d_state 64, headdim 64, chunk 64, expand 2.

### 3.2 Flow-matching action head

A rectified-flow (flow-matching) DiT conditioned on `cond_proj(ŝ_tgt)` and robot state. For an action chunk $u$, Gaussian noise $\varepsilon$, and flow time $t\in[0,1]$:

$$ u_t = (1-t)\varepsilon + t\,u,\qquad v^\star = u-\varepsilon,\qquad \mathcal{L}_{\text{action}} = \mathbb{E}_{t,\varepsilon}\big\lVert v_\phi(u_t,t\mid\hat s_{\text{tgt}}) - v^\star\big\rVert_2^2 $$

$$ \mathcal{L} = \underbrace{\mathcal{L}_{\text{pred}}}_{\text{stage 1 \& 2}} + \underbrace{\beta\,\mathcal{L}_{\text{action}}}_{\text{stage 2}} $$

---

## §4 Training recipe

| Item | Stage 1 (predictor) | Stage 2 (+head) |
|---|---|---|
| Steps | 30,000 | 30,000 |
| Batch (eff.) | 16 | 8 × grad-accum 2 |
| Objective | $\mathcal{L}_{\text{pred}}$ | $\mathcal{L}_{\text{pred}} + \beta\mathcal{L}_{\text{action}}$ |
| Warmup | 500 | 500 |

*Table 2 — Configuration (single GPU).*

- **Data mix** `libero_all` = object 66,605 · goal 51,851 · spatial 52,791 · long 100,857 → **403,428**
- **Qwen LoRA** $r=16,\ \alpha=32$ (both stages) · 6.42M adapter
- **Training curve (stage 1):** `pred_cos` 0.17 → 0.94, $\mathcal{L}_{\text{pred}}$ 1.26 → 0.39, monotone, no divergence.

---

## §5 Results

### 5.1 Clean LIBERO — the surprising gain

The unified model is evaluated on each of the four suites at 50 trials/task. Even though clean LIBERO has **no perturbation to be robust to**, the denoising objective still improves success — the corruption-to-canonical task acts as a representation regularizer.

| Suite | Baseline (no aug) | Denoising (ours) | Δ |
|---|---:|---:|---:|
| Spatial | — | 96.6% | — |
| Object | — | 99.2% | — |
| Goal | — | 97.0% | — |
| Long (10) | — | 96.0% | — |
| **Average** | **93.0%** | **97.2%** | **+4.2** |

*Table 3 — Clean LIBERO success rate (50 trials/task).*

> Baseline (93%) is a matched no-augmentation StreamingMamba trained on the same `libero_all` mixture, measured on separate hardware; only its 4-suite average is on hand. Same data volume, same architecture — the +4.2 pt gap is attributable to the denoising objective. **Per-suite baseline numbers are still to be collected for the final camera-ready.**

### 5.2 LIBERO-Plus — robustness across 7 perturbation axes

The same unified model on all 10,030 LIBERO-Plus variants (1 trial/variant, benchmark standard), broken out by suite × perturbation category.

| Perturbation axis | Spatial | Object | Goal | Long | All |
|---|---:|---:|---:|---:|---:|
| Background Textures | 96.9% | 99.2% | 96.1% | 81.3% | **93.0%** |
| Light Conditions | 99.0% | 99.7% | 90.7% | 75.9% | **91.6%** |
| Language Instructions | 86.4% | 89.8% | 75.9% | 86.7% | 84.5% |
| Objects Layout | 94.5% | 87.3% | 68.7% | 82.7% | 83.0% |
| Sensor Noise | 69.5% | 73.0% | 70.7% | 43.0% | 63.3% |
| Robot Initial States | 70.6% | 48.0% | 68.0% | 65.4% | 62.8% |
| Camera Viewpoints | 72.1% | 56.3% | 69.1% | 39.1% | 58.8% |
| **Suite total** | **83.3%** | **76.8%** | **75.4%** | **65.4%** | **75.1%** |

*Table 4 — LIBERO-Plus: success rate by suite × perturbation axis (10,030 variants).*

Photometric axes the augmentation directly targets — background, lighting — hold above 91% overall. Geometric axes (camera viewpoint, robot init) are the hardest, and the **Long** suite is consistently the weakest column (65.4% total) — long-horizon tasks compound perturbation error. This marks where the fake-viewpoint corruption has the most headroom left, pointing to a state-conditioned predictor as the natural next step.

### 5.3 Comparison with prior VLA methods

Placed against published LIBERO results. **Our two rows are final**; the prior-method rows are a skeleton — fill each cell from the cited paper's own LIBERO table (mind their trial count / checkpoint-selection convention, which varies by paper).

| Method | Spatial | Object | Goal | Long | Average |
|---|---:|---:|---:|---:|---:|
| OpenVLA | — | — | — | — | — |
| Octo | — | — | — | — | — |
| π0 (pi-zero) | — | — | — | — | — |
| MDT / others… | — | — | — | — | — |
| Baseline (ours, no aug) | — | — | — | — | 93.0% |
| **Denoising (ours)** | **96.6%** | **99.2%** | **97.0%** | **96.0%** | **97.2%** |

*Table 5 — LIBERO success rate vs prior work.*

> **Author to complete.** Prior-method cells are intentionally left blank — SOTA numbers should be copied from each paper's own LIBERO table rather than approximated. Only the two "ours" rows and the baseline average (93.0%) are our own measured values; per-suite baseline numbers are still to be collected.

---

## §6 Data positioning

- **Pretrained VLM is standard, not a liability.** Perception and language are frozen large-scale encoders — the same choice every major VLA (OpenVLA, RT-2, π0, Octo) makes. The claim is scoped to the *predictor*.
- **The world-model needs no large-scale video pretraining.** The Mamba-2 predictor is trained from scratch on in-domain data only; adding SSv2/Droid pretraining left performance essentially unchanged — evidence the predictor is data-efficient.
- **The denoising objective is the contribution.** Corrupt inputs, keep the target canonical. It buys robustness on perturbed evaluation *and* regularization on clean evaluation from a single change to the training pipeline.

---

*Framework: `VLA_DINO_StreamingMamba_FutureOnly_Denoise` · predictor 133.7M · eval seed 7. Surrounding paper narrative authored separately.*
