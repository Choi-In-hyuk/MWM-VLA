<h3 align="center" style="font-size:44px; font-weight:bold; color:#9C276A; margin: 0;">
  MWM-VLA: Mamba World Model for<br/>Inference-Time Action Generation in VLA
</h3>

<div align="center">
<p>
  <img src="https://img.shields.io/badge/Task-Vision--Language--Action-blue.svg" alt="VLA">
  <img src="https://img.shields.io/badge/World%20Model-Mamba%20(SSM)-purple.svg" alt="Mamba">
  <img src="https://img.shields.io/badge/Benchmark-LIBERO%20%2F%20LIBERO--Plus-green.svg" alt="LIBERO">
</p>
</div>

---

> **TL;DR.** MWM-VLA runs a lightweight **Mamba (SSM)** latent world model at
> **inference time** to condition action generation. A per-frame **DINOv2**
> encoder produces the current latent `s_0`; a **Qwen3-VL** VLM contributes
> language + visual intent tokens; the Mamba predictor rolls the latent forward
> to `s_end`; and a **flow-matching head** decodes `(s_0, s_end, state)` into an
> action chunk. The world model is active every control cycle — not just an
> auxiliary training loss.

## Highlights

- **Inference-time world model.** Unlike VLA-JEPA (which drops its world model at test time), MWM-VLA's Mamba predictor runs at every control step and directly conditions the action head.
- **Streaming 2-frame predictor.** The predictor consumes past (`t−H`) and present (`t`) DINO latents with per-frame time embedding and predicts the future latent (`t+H`) — no SSM state carry, so the training and inference distributions stay identical.
- **VLM intent conditioning.** Qwen3-VL (with LoRA) turns the instruction + current image into intent tokens that steer the latent rollout.
- **Endpoint + flow matching.** The predictor targets the chunk-endpoint latent (a large, learnable change), and a flow-matching head decomposes `(s_0 → s_end)` into the action chunk.
- **Pretraining on unlabeled video.** The world model is pretrained on **SSv2 + Droid** before task-specific fine-tuning, giving a better latent-dynamics prior.

## Method

```
                 ┌──────────── Qwen3-VL (intent) ──────────┐
 instruction ───►│ language + current-frame understanding  │── intent tokens ─┐
                 └─────────────────────────────────────────┘                  │
                                                                              ▼
 past frame ─────► DINO ──► s_{t-H} ─┐                             Streaming Mamba
                                     ├─► (+ time embed) ─────────► predictor  ──► s_{t+H}
 current frame ──► DINO ──► s_t ─────┘                                              │
                                                                                    ▼
 robot state ─────────────────────────► Flow-matching head (s_t, s_{t+H}, state) ─► action chunk
```

**Training.** Two stages:

| Stage | Trains | Loss | Notes |
|---|---|---|---|
| `predictor` | Streaming Mamba predictor (+ Qwen-LoRA) | `L_pred` | 2-frame input (t−H, t) → predict t+H; no SSM state carry |
| `stage2` | predictor fine-tune + flow-matching head (+ Qwen-LoRA) | `L_pred + L_action` | Head is conditioned on the *predicted* `s_end` (inference-consistent) |

Optional: **`pretrain`** stage runs the predictor on SSv2 + Droid before the task-specific stages.

**Inference.** Every control cycle:
`(past, current) → DINO → (s_{t-H}, s_t)`; `(s_{t-H}, s_t, Qwen tokens) → Mamba → s_{t+H}`;
`(s_t, s_{t+H}, state) → flow-matching head → action chunk`.

## Results

| Benchmark | Setting | Success |
|---|---|---|
| LIBERO-10 (50 trials × 10 tasks = 500 eps) | Streaming Mamba WM + Qwen-LoRA | **94.2%** |
| LIBERO-Plus | Streaming Mamba WM + SSv2/Droid pretrain | **72.87%** |

Single-seed numbers; LIBERO's action sampling is non-deterministic (≈±2% at 500 episodes).

## Code map

| Path | What |
|---|---|
| [`starVLA/model/framework/VLA_DINO_StreamingMamba.py`](starVLA/model/framework/VLA_DINO_StreamingMamba.py) | Streaming 2-frame world-model VLA (main model) |
| [`starVLA/model/framework/VLA_DINO_StreamingMamba_FutureOnly.py`](starVLA/model/framework/VLA_DINO_StreamingMamba_FutureOnly.py) | Future-only variant |
| [`starVLA/model/framework/VLA_DINO_Mamba_JEPA.py`](starVLA/model/framework/VLA_DINO_Mamba_JEPA.py) | JEPA-style DINO training variant |
| [`starVLA/model/framework/VLA_DINO_Mamba_Diff.py`](starVLA/model/framework/VLA_DINO_Mamba_Diff.py) | Endpoint predictor + flow-matching head (predecessor of Streaming) |
| [`starVLA/model/modules/world_model/mamba_world_model.py`](starVLA/model/modules/world_model/mamba_world_model.py) | `StreamingMambaPredictor` (Mamba-2, 2-frame obs + time embed) |
| [`scripts/train_mamba_wm.py`](scripts/train_mamba_wm.py) | Trainer (`--stage predictor\|stage2`, `--qwen_lora`, …) |
| [`scripts/pretrain_ssv2_droid.py`](scripts/pretrain_ssv2_droid.py) | Predictor pretraining on SSv2 + Droid |
| [`scripts/run_streaming_libero10.sh`](scripts/run_streaming_libero10.sh) | Two-stage training chain on LIBERO-10 |
| [`scripts/run_streaming_libero_all_future_only.sh`](scripts/run_streaming_libero_all_future_only.sh) | Future-only variant, libero_all suite |
| [`scripts/run_finetune_libero_from_pretrain_100k.sh`](scripts/run_finetune_libero_from_pretrain_100k.sh) | Fine-tune from SSv2+Droid pretrain |
| [`scripts/eval_libero_dino.sh`](scripts/eval_libero_dino.sh) | LIBERO evaluation |
| [`scripts/eval_libero_plus_2gpu.sh`](scripts/eval_libero_plus_2gpu.sh) | LIBERO-Plus evaluation (2-GPU) |

## Quick start

### 1. Install

Full environment setup (Python 3.10 / CUDA 13.0 / PyTorch 2.12 / mamba-ssm) is in **[INSTALL.md](INSTALL.md)**.

```bash
conda env create -f environment/vla_jepa.yml
conda activate vla_jepa
pip install -e .
```

### 2. Train

```bash
# Streaming Mamba WM on LIBERO-10 (predictor stage + stage2 with Qwen-LoRA)
bash scripts/run_streaming_libero10.sh all

# Optional: pretrain the predictor on SSv2 + Droid first
bash scripts/run_pretrain_ssv2_droid.sh
# then fine-tune on LIBERO
bash scripts/run_finetune_libero_from_pretrain_100k.sh
```

### 3. Evaluate

```bash
# LIBERO-10, 50 trials/task
bash scripts/eval_libero_dino.sh libero_10 50 18099 \
  results/stream_libero10/stage2/checkpoints/mamba_wm_final.pt stream_v2 0

# LIBERO-Plus (2-GPU)
bash scripts/eval_libero_plus_2gpu.sh
```

Dataset paths (LIBERO, LIBERO-Plus, SSv2, Droid) are configured in the scripts — see [INSTALL.md §5](INSTALL.md#5-datasets).

## Model release

Trained MWM-VLA checkpoints (Streaming Mamba WM + Qwen-LoRA + flow-matching head, plus the SSv2+Droid-pretrained predictor) will be released. Until then, the training scripts above reproduce the reported numbers.

## Acknowledgement

Built on top of [VLA-JEPA](https://arxiv.org/abs/2602.10098). Uses
[Mamba](https://github.com/state-spaces/mamba) as the latent world model,
[DINOv2](https://github.com/facebookresearch/dinov2) as the frame encoder, and
[Qwen3-VL](https://github.com/QwenLM/Qwen3-VL) as the VLM.

## License

Apache License 2.0 — see [LICENSE](LICENSE).
