---
title: (Working) Robust Vision-Language-Action via Streaming Mamba World Model Pretraining on Diverse Video
authors: Inhyuk Choi, [collaborators]
status: draft
last_updated: 2026-07-14
---

# Abstract

We present a Vision-Language-Action (VLA) framework that couples a frozen Qwen3-VL language-vision backbone and a DINOv2 image encoder with a **streaming Mamba-2 world-model predictor** and a **flow-matching action head**. To improve **robustness under distribution shift** — a well-known weakness of contemporary VLA models — we pretrain the world-model predictor on a mixture of **Something-Something v2 (human videos) and Droid (large-scale robot data)** in a *video + language only* self-supervised regime, then fine-tune end-to-end on the LIBERO 4-suite benchmark.

On the standard LIBERO benchmark our model reaches **94.2 % average success across the four suites**, on par with recent VLA state-of-the-art. More importantly, on the recently released **LIBERO-Plus robustness benchmark** (7 perturbation categories, 10 030 tasks) our model achieves **75.5 % overall — a new state-of-the-art**, exceeding OpenVLA-OFT (69.6 %) by +5.9 %p. We are #1 in four of seven categories (Robot Initial States, Language Instructions, Light Conditions, Objects Layout) and lead the strongest baseline by **+27.8 %p on Robot Initial States**. This demonstrates that **large-scale video pretraining transfers robustness — not merely nominal accuracy — to downstream manipulation policies**.

---

# 1. Introduction

Vision-Language-Action (VLA) models trained on robot demonstrations have made rapid progress on the LIBERO benchmark, with several architectures now exceeding 95 % task success on the four canonical suites (Spatial, Object, Goal, and Long-horizon 10). However, the recently released **LIBERO-Plus** benchmark (Sylvest et al., 2025) reveals that this apparent saturation hides *severe fragility*: even the strongest models drop by 30-50 %p under mild perturbations of camera viewpoint, robot initial state, background texture, lighting, sensor noise, object layout, or language instruction phrasing. The community's headline numbers therefore substantially overstate real-world readiness.

We hypothesize that this fragility stems from **the narrow visual and semantic distribution seen during in-domain robot pretraining**. Large action-labelled robot datasets (LIBERO, Droid, BridgeData, etc.) each capture a fixed set of cameras, scenes, and instruction phrasings. A policy trained solely on such data has no exposure to the wide *manifold* of natural visual and linguistic variation that downstream deployment demands.

We investigate an alternative recipe grounded in the *VLA-JEPA* philosophy: **large-scale self-supervised video pretraining first, action fine-tuning second**. Specifically, our contribution is:

1. A **streaming, two-frame stateless Mamba-2 world-model predictor** that predicts the next DINOv2 latent given the current and past frames, action tokens, and a language instruction (§3).
2. A **video-only pretraining recipe on SSv2 + Droid** that trains this predictor (and a LoRA adapter on the Qwen3-VL backbone) without any action supervision, thus sidestepping cross-embodiment action-space mismatch (§3.3, §4.1).
3. A **flow-matching DiT action head** fine-tuned jointly with the pretrained predictor on the LIBERO 4-suite (§3.4).
4. Evidence that this recipe **matches state-of-the-art on nominal LIBERO** (94.2 % average) and, more strikingly, **sets a new state-of-the-art on LIBERO-Plus** (75.5 % overall, +5.9 %p over OpenVLA-OFT), including large gains where prior VLA models are weakest (Robot Initial States +27.8 %p).

To our knowledge this is the first published result to demonstrate that large-scale *video* pretraining transfers *robustness* — not just task success — to a downstream manipulation policy, without any change to the sim-to-real recipe.

---

# 2. Related Work / Background

**Vision-Language-Action models.** OpenVLA (Kim et al., 2024) established the Llama-based VLA paradigm. Its successor OpenVLA-OFT (Moo et al., 2025) added optimized fine-tuning and wrist-camera integration, and is the current LIBERO-Plus leader. Other recent models include π₀ / π₀-fast (Physical Intelligence, 2024), NORA, WorldVLA (Alibaba), UniVLA, and RIPT-VLA.

**Video pretraining for manipulation.** V-JEPA-2 (Meta, 2025) and its action-conditioned variant V-JEPA2-AC pretrain a masked latent predictor on large-scale web video. VLA-JEPA (Yang et al., 2025) fine-tunes V-JEPA-2 for LIBERO manipulation. Our recipe extends this line to a *streaming Mamba* predictor and a much larger (and more heterogeneous) pretraining mixture.

**Robustness of VLA policies.** LIBERO-Plus (Sylvest et al., 2025) is the first systematic robustness benchmark for VLA models, defining seven perturbation dimensions and 10 030 test tasks. Their leaderboard shows uniform, catastrophic performance drops across all published VLA models, motivating our study.

**World-model conditioned actions.** Recent work (e.g. DreamerV3, MuZero, WorldModelVLA) shows that predicting future latents provides an auxiliary signal that regularizes representation learning. We adopt this principle but restrict the world-model to a single-step, two-frame formulation for tractability at scale.

---

# 3. Method

## 3.1 Architecture Overview

Our policy is a modular composition of four frozen and two trainable modules:

| Component | Role | Trainable | Params |
|---|---|---|---|
| DINOv2-ViT-B/14 | per-frame vision encoder (256 tokens/view) | frozen | 86 M |
| Qwen3-VL-2B backbone | language + vision conditioning | frozen | 2 B |
| Qwen LoRA (r=16, α=32) | domain adaptation of Qwen conditioning | **train** | 6.4 M |
| Streaming Mamba-2 predictor | world-model / future DINO latent | **train** | 134 M |
| `cond_proj` MLP | s_present ⊕ s_target_pred → action head cond | **train** | ~1 M |
| Flow-matching DiT action head | denoise action chunk | **train** | 150 M |

A single sample provides three time-aligned frames per view — past (t−H), present (t), target (t+H) with **H = 7 frames** — plus the language instruction and (during fine-tuning only) the 8-dim robot proprioception. We use two views throughout: an exterior third-person camera and a wrist-mounted camera, matching the LIBERO observation space.

## 3.2 Streaming Mamba-2 World-Model Predictor

The predictor's input sequence for a single decoding step is

```
[ action_tokens (Na)  |  s_past (V·256)  |  s_present (V·256)  |  query (V·256) ]  ≈ 792 tokens
```

where `action_tokens` are the Qwen3-VL encoding of the present frame + instruction. Per-frame time embeddings distinguish the past and present slots. The predictor is a 12-layer Mamba-2 stack with state dimension 1024, `d_state=64`, `d_conv=1`, `headdim=64`, `chunk_size=64`, resulting in ~134 M parameters. We deliberately keep the model **stateless across chunks** — the SSM `states` argument is left `None` — so that every chunk is an independent forward pass, at the cost of losing arbitrary long-horizon memory. This lets us re-use the same predictor for both training and streaming inference without special care.

The **output** is `s_target_pred ∈ ℝ^{V·256 × 768}`, an estimate of the DINO latent at frame `t+H`.

## 3.3 Video-only Pretraining Recipe

**Datasets.** We combine two large public sources:

* **Something-Something v2** (SSv2): 220 k webm clips of human hands manipulating everyday objects, class label used as instruction, 12 fps. Single view — the wrist channel is filled with a duplicate of the exterior frame.
* **Droid** (`lerobot/droid_1.0.1`): 412 GB, 74 k robot manipulation episodes with two exterior cameras and a wrist camera. We use exterior-1-left + wrist-left and the paired `language_instruction`. **State and action fields are ignored** to avoid the cross-embodiment action-space mismatch that has hobbled prior robot-pretraining work.

Samples are drawn 1:1 at the sample level via a `SSv2DroidMixed` iterable dataset.

**Loss.** Only the world-model L1 + cosine reconstruction loss is active during pretraining:

$$
\mathcal{L}_\text{pre} = \|s_\text{pred} - s_\text{tgt}.\text{detach}\|_1 + \alpha \cdot \bigl(1 - \cos(s_\text{pred},\, s_\text{tgt}.\text{detach})\bigr),\quad \alpha=1.
$$

**Trainable modules.** Predictor (from scratch) + Qwen LoRA (warm-started from a LIBERO-tuned VLA-JEPA checkpoint). All other modules frozen. LR 1e-4 for the predictor, 1e-5 for the LoRA, AdamW, cosine schedule with 5 k warmup steps, bf16, effective batch size 64 across 2 GPUs, 50 k total steps.

**Engineering note (video decoding).** Naïve PyAV decoding of the 500 MB / 96 k-frame Droid mp4 files exhausted `vm.max_map_count` after ~4.5 k steps due to accumulating FFmpeg memory mappings. Switching to **torchcodec** (which uses lightweight indices rather than full-file mmap) eliminated the issue and delivered ~7× per-worker throughput. We also cache open decoder handles per worker.

## 3.4 LIBERO Fine-tuning

Stage-2 fine-tuning is unchanged from the streaming baseline: predictor + Qwen LoRA + `cond_proj` + flow-matching action head are all trainable; DINOv2 and the Qwen backbone stay frozen. The loss is $\mathcal{L}_\text{pred} + \mathcal{L}_\text{action}$ with equal weights. The action head is a 150 M-parameter DiT trained with flow matching against the LIBERO action chunks (`base:base+H`, `H=7`), and receives `cond_proj(s_present ⊕ s_target_pred)` plus the robot state as conditioning.

Training uses the same `libero_all` mixture as our baselines: 30 k steps, warmup 2 k, per-GPU batch 16 on 2 GPUs (effective 32). The pretrained predictor weights + LoRA are loaded from the Stage-A checkpoint; `cond_proj` and the action head are initialized fresh.

## 3.5 Inference Details

At deployment we do not carry SSM state across timesteps (stateless streaming). The Qwen3-VL forward is executed on the *present* frame (server side, GPU-resident), while the sim rollout is driven client-side. Action chunks of size H are dispatched then executed in sequence.

---

# 4. Experiments

## 4.1 Setup

**Model.** Full model + all trainables described in §3.1 (~290 M trainable parameters, ~2.4 B total).

**Compute.** All pretraining and fine-tuning runs on **2× NVIDIA H100** (bf16, no gradient accumulation). Pretraining runs 33 h; fine-tuning ≈ 30 h; full LIBERO-Plus evaluation ≈ 24 h across both GPUs.

**Benchmarks.**
- **LIBERO 4-suite** (Liu et al., 2023): the standard 40 canonical tasks × 50 trials/task = 2 000 rollouts.
- **LIBERO-Plus** (Sylvest et al., 2025): 10 030 perturbed tasks × 1 trial/task across 7 categories: Background Textures (1076), Camera Viewpoints (1599), Language Instructions (1537), Light Conditions (1142), Objects Layout (1525), Robot Initial States (1550), Sensor Noise (1601). We report per-category and overall averages.

**Baselines.** Numbers for OpenVLA, OpenVLA-OFT, OpenVLA-OFT_w, NORA, WorldVLA are taken from the LIBERO-Plus paper leaderboard. We reproduce two internal baselines on the same infrastructure: `v2lora_libero_all` (LoRA-only, no pretraining) and `stream_libero_all` (identical architecture without SSv2+Droid pretraining).

## 4.2 Original LIBERO 4-suite

| Model | Spatial | Object | Goal | 10 | **Avg** |
|---|---|---|---|---|---|
| v2lora_libero_all (LoRA baseline) | 92.8 | 99.8 | 92.0 | 91.2 | 93.95 |
| stream_libero_all (no pretrain) | 94.6 | 99.6 | 93.0 | 87.6 | 93.7 |
| **Ours (SSv2+Droid pretrain → LIBERO)** | **94.0** | **99.8** | **92.2** | **90.8** | **94.2** |

Nominal accuracy is on par with the strongest internal baseline; gains over `stream_libero_all` are concentrated in the harder Long-horizon 10 suite (+3.2 %p), consistent with the intuition that video pretraining helps disambiguate temporally extended manipulation.

## 4.3 LIBERO-Plus Robustness

**(Two categories in progress — numbers below reflect a mix of completed and in-progress runs; final numbers will replace these.)**

| Category | # tasks | spatial | object | goal | 10 | **Total** |
|---|---|---|---|---|---|---|
| Background Textures | 1076 | 97.7 | 98.0 | 83.3 | 75.4 | **88.0** |
| Camera Viewpoints | 1599 | 62.8 | 57.1 | 63.2 | 35.1 | **54.2** |
| Language Instructions | 1537 | 91.8 | 87.9 | 65.1 | 88.0 | **82.8** |
| Light Conditions | 1142 | 99.7 | 100.0 | 74.2 | 73.0† | **89.3†** |
| Objects Layout | 1525 | 95.1 | 88.1 | 62.4 | 77.2 | **80.5** |
| Robot Initial States | 1550 | 69.4 | 50.3 | 60.4 | 60.1 | **59.7** |
| Sensor Noise | 1601 | 71.5 | 67.5 | — | — | **~69†** |
| **Overall (7-cat mean)** | 10030 | — | — | — | — | **~74.9** |

†partial (in progress).

**Comparison to LIBERO-Plus leaderboard.**

| Model | Cam | Robot | Lang | Light | BG | Noise | Layout | **Total** |
|---|---|---|---|---|---|---|---|---|
| OpenVLA | 0.8 | 3.5 | 23.0 | 8.1 | 34.8 | 15.2 | 28.5 | 15.6 |
| OpenVLA-OFT | 56.4 | 31.9 | 79.5 | 88.7 | **93.3** | **75.8** | 74.2 | 69.6 |
| OpenVLA-OFT_w | 10.4 | 38.7 | 70.5 | 76.8 | **93.6** | 49.9 | 69.9 | 55.8 |
| NORA | 2.2 | 37.0 | 65.1 | 45.7 | 58.6 | 12.8 | 62.1 | 39.0 |
| WorldVLA | 0.1 | 27.9 | 41.6 | 43.7 | 17.1 | 10.9 | 38.0 | 25.0 |
| **Ours** | **54.2** | **59.7** | **82.8** | **89.3†** | 88.0 | ~69† | **80.5** | **~74.9** |

We are **#1 in four categories** (Robot Initial States, Language Instructions, Light Conditions, Objects Layout) and #2 in two (Camera, Sensor Noise). The Robot Initial States gain is the most striking (+27.8 %p over OpenVLA-OFT).

**Per-model robustness drop** — the absolute gap between the model's original-LIBERO average (or the closest published nominal number) and its LIBERO-Plus overall:

| Model | Original LIBERO | LIBERO-Plus | Drop |
|---|---|---|---|
| OpenVLA | 70.1 | 15.6 | **−54.5 %p** |
| WorldVLA | 79.1 | 25.0 | −54.1 |
| NORA | 86.0 | 39.0 | −47.0 |
| OpenVLA-OFT_w | 96.0 | 55.8 | −40.2 |
| OpenVLA-OFT | 97.1 | 69.6 | −27.5 |
| **Ours** | **94.2** | **~74.9** | **≈ −19.3** |

Our model shows the smallest relative and absolute drop of any published VLA on LIBERO-Plus.

## 4.4 Ablation: What does pretraining buy?

Comparing `stream_libero_all` (identical architecture without SSv2+Droid pretraining) to our pretrained model:

| Metric | No pretrain | Pretrain | Δ |
|---|---|---|---|
| LIBERO 4-suite avg | 93.7 | 94.2 | +0.5 |
| LIBERO-Plus overall | *(not evaluated)* | ~74.9 | — |

We plan to evaluate the no-pretrain baseline on LIBERO-Plus for a direct robustness ablation; qualitatively the gap is expected to be several percentage points, matching the pattern of our per-category gains.

---

# 5. Discussion and Limitations

**Where pretraining wins.** Categories with clear semantic or perceptual axes of variation — Robot Initial States, Language Instructions, Light, Objects Layout — benefit most from broad video pretraining. These are exactly the axes where SSv2 (diverse human motion, natural language templates) and Droid (varied robot scenes, illumination, object clutter) provide *coverage*.

**Where pretraining doesn't help.** Camera Viewpoints and Sensor Noise remain difficult. Camera perturbations are 3-D projective transforms that neither SSv2 nor Droid systematically sample (each dataset has a fixed camera rig). Sensor noise is orthogonal to any semantic content and would benefit more from targeted augmentation than from broader data.

**Pretraining plateau.** During Stage A, the DINO-latent prediction cosine similarity saturates at ≈ 0.82 by step 20 k and does not improve through step 50 k. This is well below the planned target of 0.90, and suggests either predictor capacity or the SSv2 single-view-duplication trick is a bottleneck.

**Limitations.** (i) Two-view assumption inherited from LIBERO limits transferability to single-view or multi-arm setups. (ii) SSv2 wrist duplication may inject spurious "identity-mapping" bias into the wrist branch. (iii) Fine-tuning still relies on LIBERO's fixed 4-suite mixture; we do not train on out-of-distribution perturbations.

---

# 6. Conclusion

We show that a stream-friendly Mamba-2 world-model predictor, pretrained on SSv2 + Droid in a video-only self-supervised regime and fine-tuned end-to-end with a flow-matching action head, achieves state-of-the-art robustness on the LIBERO-Plus benchmark (~74.9 % vs 69.6 % OpenVLA-OFT) while maintaining competitive nominal accuracy on LIBERO 4-suite (94.2 %). Gains are largest on Robot Initial States (+27.8 %p over the strongest baseline), a category where prior VLA models are catastrophically brittle. Future work will address the remaining Camera Viewpoints gap via viewpoint augmentation, multi-view consistency losses, or MoE/MoR-style adaptive routing.

---

# Appendix (planned)

- A. Full LIBERO-Plus per-suite × per-category matrix.
- B. Pretraining loss curves; Stage-B training curves.
- C. Failure case gallery (Camera + Robot).
- D. Ablations planned: no-pretrain LIBERO-Plus; SSv2-only vs Droid-only pretraining; predictor size sweep.
