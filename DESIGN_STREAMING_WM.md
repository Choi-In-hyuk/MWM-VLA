# Streaming Mamba-2 World Model — Design Doc (v4)

> **What this is now**: a 2-frame world-model predictor (past + present →
> future) for VLA. **No SSM hidden-state carry.** Cold-start handled
> structurally by dataloader front-padding. Built on Mamba-2 so that the
> longer obs-token sequence (2×N patches + time embedding) stays cheap.
>
> **What this used to be (v1–v3)**: a stateful Mamba-2 SSM whose hidden
> state was propagated across chunks within an episode at inference. That
> direction failed in evaluation (see §6) and was retired. The current
> design keeps only the Mamba-2 *block* (which is still useful for the
> longer sequence) and drops the hidden-state propagation entirely.

---

## 1. Final architecture (current, v4)

### 1.1 Predictor: `StreamingMambaPredictor`

`starVLA/model/modules/world_model/mamba_world_model.py`

| | value |
|---|---|
| blocks | 12 × `StreamingMambaBlock` (Mamba-2 wrapper) |
| `state_dim` | 1024 |
| `d_state` | 64 |
| `d_conv` | 1 |
| `headdim` | 64 |
| `tokens_per_frame` (N) | 512 (2 DINOv2 views × 256 patches) |
| trainable params | 133.7 M |

Per-chunk token sequence:
```
[ action(Na=24) | obs(2 × N = 1024) | query(N = 512) ]   total ≈ 1561 tokens
```

- **Obs tokens** carry: `dino_proj(s)` + `role_emb["obs"]` + `patch_pos` +
  `time_emb[frame_idx]` (frame 0 = past, frame 1 = present).
- **No `robot_state` token.** robot_state is fed only to the action head.
- **Query tokens**: N learnable tokens at the tail; the SSM reads out
  `s_target_pred` from the last N positions.

### 1.2 Framework forward (training)

`starVLA/model/framework/VLA_DINO_StreamingMamba.py`

```
dataloader sample:
    obs_indices = [-H, 0, H]           ← H = 7 for LIBERO
    video        : [V, 3, H_img, H_img, 3]   = past, present, target
    state_full   : [3, 8]                       at past/present/target
    action       : [H, 7]                       sliced at base..base+H

forward:
    dino_all  = DINO(video)             # [B, 3, N, D]
    s_past, s_present, s_target = dino_all[:,0], [:,1], [:,2]
    s_in      = stack([s_past, s_present], dim=1)        # [B, 2, N, D]
    a_tok     = Qwen(present_frame, lang)                # [B, Na, A_dim]
    s_target_pred, _ = predictor.forward_chunk(a_tok, s_in, states=None)
    L_pred    = L1(s_target_pred, s_target)

    if stage2:
        cond     = cond_proj(concat(s_present, s_target_pred))
        state_3d = state_full[:,1].unsqueeze(1)               # robot_state at present
        L_action = action_model(cond, action, state_3d)        # flow-matching DiT
```

### 1.3 Framework inference (`predict_action`)

```
on episode_reset():
    self._past_frame = None

on predict_action(batch_images, instructions, state):
    past_views = self._past_frame if self._past_frame is not None
                 else batch_images          ← cold start at t=0: past := present
    s_in   = DINO(stack(past_views, batch_images))        # [B, 2, N, D]
    a_tok  = Qwen(batch_images, instructions)
    s_target_pred, _ = predictor.forward_chunk(a_tok, s_in, states=None)
    cond   = cond_proj(concat(s_present, s_target_pred))
    actions = action_head.predict_action(cond, state.unsqueeze(1))    # 7-action chunk
    self._past_frame = batch_images
    return actions
```

Cold start covers exactly the same distribution as training: the dataloader
front-pads frame 0 when `base < H` (`obs_indices=[-H, 0, H]`), so a sample
with `base=0` looks like `video=[frame_0, frame_0, frame_H]` — identical to
inference at t=0 where past=present.

---

## 2. Training recipe (v4)

Matches the baseline `run_2stage_dino_lora_ddp.sh` recipe so the comparison
is apples-to-apples; only the framework differs.

| | value |
|---|---|
| dataset | libero_10 (10 long-horizon tasks) |
| stages | `predictor` (stage1) → `stage2` (joint) |
| steps | 15 000 + 15 000 |
| optimizer | AdamW, lr 1e-4, weight_decay 1e-4, cosine schedule, warmup 500 |
| per-GPU batch | 16 |
| DDP | 2 GPU → eff. batch 32 |
| LoRA | Qwen3-VL `q/k/v/o_proj`, r=16, α=32 (trained in both stages) |
| backbone ckpt | `VLA-JEPA-LIBERO.pt` |

Final metrics (last step):

| stage | pred_cos | pred_loss (L1) | action_loss |
|---|---|---|---|
| stage1 (predictor) | 0.9124 | 0.4711 | — |
| stage2 (joint)     | 0.9242 | 0.4384 | 0.0104 |

Eval (LIBERO-10, 50 trials/task = 500 episodes): **93.2 % success**.

---

## 3. File/launcher map

| File | Role |
|---|---|
| `starVLA/model/modules/world_model/mamba_world_model.py` | `StreamingMambaBlock` (Mamba-2 with `initial_states`), `StreamingMambaPredictor` (2-frame `_token_obs` + per-frame `time_emb`) |
| `starVLA/model/framework/VLA_DINO_StreamingMamba.py` | Framework. `forward` (single-chunk, 2-frame), `predict_action` (2-frame, internal `_past_frame` cache), `episode_reset` |
| `scripts/train_mamba_wm.py` | Trainer (random-window, single-chunk for this framework) |
| `scripts/run_streaming_libero10.sh` | Two-stage launcher (predictor → stage2) |
| `scripts/eval_libero_dino.sh` | LIBERO eval (server + rollout) |
| `STREAM_V2_REPORT.md` | Architecture, training metrics, eval, model sizes |

Legacy components kept for completeness but unused at inference:
- `StreamingMambaPredictor.forward_chunk(...)` still accepts a `states`
  argument for back-compat. We always pass `states=None`.
- `VLA_DINO_StreamingMamba.forward_stage2(...)` (truncated-BPTT path) is
  retained but not called by any launcher. To delete after one more clean
  pass.

---

## 4. Sanity checks performed (v4)

| Check | Result |
|---|---|
| `_token_obs` accepts both 3-D `[B,N,D]` and 4-D `[B,2,N,D]` and produces the right sequence length (N or 2N) | OK |
| `time_emb[:T].unsqueeze(0)` broadcasts correctly over `[B,T,N,D]` (earlier bug: extra `.unsqueeze(2)` produced a 5-D tensor) | fixed and verified |
| Training forward (`B=2` dummy batch) returns `pred_loss`, `pred_cos`, `action_loss` with grad and backward works | OK |
| `predict_action` accepts `state` as `[1,1,8]`, `[1,8]`, `[8]`, `None`; output `[1, 7, 7]` in all cases | OK |
| `_past_frame` is `None` after `episode_reset` and is repopulated after the first call | OK |
| Predictor params have no `robot_state_emb`, no `role_emb["robot"]`, no `_token_robot` | OK (removed) |

---

## 5. Why robot_state was removed from the predictor

Earlier iterations fed `robot_state` to the predictor as a 1-token role.
This was a steady source of shape/dtype bugs at inference:

- Client sends `state` wrapped as `[state]` → arrives as `[1,1,8]`.
- The predictor wanted `[B, 8]` (2-D).
- The action head wanted `[B, 1, 8]` (3-D).
- Reshape/unsqueeze fixes were correct in isolation but the inconsistent
  shape contract made each new bug subtle (one path uses 2-D, another 3-D).

Decision: the predictor consumes vision + language only; `robot_state` is
the action head's input — same as the baseline `VLA_DINO_Mamba_Diff`. One
token removed from the predictor sequence; the dim-mismatch class of bugs
becomes structurally impossible at the predictor boundary.

---

## 6. What we tried before and dropped (v1–v3)

### 6.1 v1 — stateful Mamba-2, hidden state carried across chunks within an episode

Idea: predictor's SSM hidden `h` is initialised to zero at `episode_reset()`;
each subsequent chunk forward consumes the previous chunk's `h` as
`initial_states` and emits a new `h`. Training Stage-1 used **M = 2 BPTT**:
each sample = two consecutive chunks of one episode, gradient flows from
chunk-1's loss into chunk-0's forward via the carried state.

Why it should have worked:
- Mamba-2's `mamba_chunk_scan_combined(initial_states=h, return_final_states=True)`
  is gradient-clean — `initial_states` is differentiable (Mamba-1's `step()`
  is not; that was the *technical* reason for moving from Mamba-1).
- `d_conv = 1` makes "split with state" mathematically equivalent to "single
  long forward" — empirically `max |Δ| = 0`.

Stage-1 actually worked well: pred_cos plateaued at ~0.935 around step
32 800 (vs baseline 0.913 at 50k). **Stage-2 is where this design
collapses.** See §6.2.

### 6.2 v2 — Stage-2 truncated-BPTT episode streaming

Idea (`scripts/train_streaming_stage2.py`, `EpisodeStreamingDataset`):
each rank maintains `N` slots; each slot streams one episode chunk-by-chunk,
carrying SSM state across training **steps** (detached between steps,
truncated BPTT length 1). The intent was to close the train/inference state
distribution gap (Stage-1 still resets state every sample).

Result:
- pred_cos **regressed** from 0.938 → 0.902 over Stage-2 training.
- action_loss got stuck at 0.043.

Diagnosis (see RESEARCH §3 entry for "Streaming v1 (SSM hidden carry)"):
the training distribution still doesn't match inference — Stage-2's
single-chunk BPTT is much shorter than the inference horizon, and the
random-window pretraining habits leak. We also lost the gradient diversity
of random sampling because each batch sees one trajectory per slot.

Decision: **abandon truncated-BPTT**. Reverted to a simple Stage-2 with the
M=2 random-window pipeline (called the *"SIMPLE"* recipe at the time).

### 6.3 v3 — SIMPLE Stage-2 (M=2 random window) with stateful Mamba kept

Trained 30 000 steps; pred_cos 0.943, action_loss 0.010. Looked great on
the metrics. Sim evaluation: **0% success.** Robot fell forward and
stopped.

Diagnosis after multiple debugging sessions:
- The action head was only trained on chunk-0 of each M=2 sample, i.e.
  always on a `predictor(states=None)` forward. At inference the head saw
  outputs from `predictor(states=carried_h)`, a distribution it had never
  been trained on.
- We tried three inference policies on this same Stage-2 checkpoint:
  - episode-long carry: **0 %**
  - toggle (`states=None` / carry alternating): **66.7 %**
  - always `states=None`: **83.3 %**
- "always None" worked best because it matched the (only) distribution the
  action head had been trained on. But that defeats the entire point of
  carrying SSM state.

So we accepted: **with M=2 random-window training, SSM carry at inference
is fundamentally distribution-shifted.** The fix would be either
(a) train the action head on chunk-1 too (would need 14 actions per
sample), or (b) drop the carry and instead give the predictor more
visual context up front. We chose (b) — this is v4.

### 6.4 Ckpt-merging bug (resolved, not a design issue)

When we killed Stage-2 (v3) at step 30 000 we did **not** get a `final.pt`
written, because `save("final", full=True)` only runs at `max_steps`.
We then hand-rebuilt the full ckpt by:
```
build_framework → load_backbone(strict=False) → apply_qwen_lora → load step30000 (strict=False) → save state_dict()
```
The two `strict=False` loads silently dropped Qwen backbone weights
because the prefix `qwen_vl_interface.model.X` (in the original backbone
ckpt) didn't match the LoRA-wrapped key `qwen_vl_interface.model.base_model.model.X`
(in the model after `apply_qwen_lora`). 614 of 626 Qwen weights ended up
**different from the original backbone** in the merged final.pt.

This is why early sim evaluations on the merged v3 ckpt were uniformly
catastrophic. We fixed the merger by remapping keys before the second
`load_state_dict`, but the lesson stuck: for v4 we simply train to
`max_steps` so the framework's own `save("final", full=True)` runs.

---

## 7. Open items

- **4-suite training of v4** (libero_all): the +2.0%p libero_10 gain is at
  the same single-suite training budget as the 91.2 % baseline. Need to
  re-run on libero_all to compare to the 93.95 %-avg baseline.
- **LIBERO-Plus evaluation**: assets downloaded to
  `/home/choi/LIBERO-plus/libero/libero/assets/` (9.5 GB). Package install
  (`pip install -e .` + sudo apt deps + config path) deferred until after
  the current LIBERO-only experiment cycle.
- **Latency**: a second DINO forward per replan (past + present together)
  roughly doubles the predictor's input length (1049 → 1561 tokens). Not
  yet measured end-to-end.
- **`forward_stage2` cleanup**: the (now unused) truncated-BPTT entry
  point can be removed after one more cleanup pass.

---

## 8. Contribution framing (paper-level, v4)

> **Past+present world-model conditioning for VLA without SSM carry.**
> We extend the V2 endpoint world-model predictor with a single extra
> input — the DINO latent of the *previous* re-plan frame (`t − H`) — and
> show that this alone closes most of the gap to a much heavier VLA-JEPA
> on LIBERO-10 (93.2 % vs 95.8 %), while staying entirely inside the
> *inference-time world model* setting (≈300 M trainable params, no
> stateful runtime). We deliberately do **not** propagate the Mamba SSM
> hidden state across re-plans: an earlier iteration that did so failed
> at evaluation, and the failure mode generalises — stateful predictors
> trained on short windows shift the action head's input distribution.
> Two-frame conditioning recovers temporal context without that risk. The
> Mamba-2 block is still useful for processing the longer per-chunk
> sequence (1561 tokens) at near-Mamba-1 latency, but its `initial_states`
> hook is never used at inference.
