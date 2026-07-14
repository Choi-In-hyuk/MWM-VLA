# Copyright 2026 VLA-JEPA research. MIT License.
"""
VLA_DINO_StreamingMamba — streaming-Mamba world model on top of V2+LoRA.

Difference vs `VLA_DINO_Mamba_Diff`
-----------------------------------
The Mamba predictor is replaced by `StreamingMambaPredictor`, which:

  * In training, processes M consecutive chunks (each carrying chunk-start
    DINO + #mid-chunk stream frames at relative offsets +2/+4/+6) as ONE
    concatenated Mamba sequence. One s_end prediction per chunk.
  * At inference, runs `forward_for_inference` at each chunk start and
    `step_stream` for every mid-chunk frame as it arrives, carrying SSM
    hidden state across all calls within an episode.

Everything else (Qwen + LoRA, DINOv2 frozen, FlowmatchingActionHead) is
inherited unchanged from `VLA_DINO_Mamba_Diff`.

Stage 1 (predictor)
-------------------
Predict per-chunk s_end. L = L1(s_end_pred, s_end_gt) summed over M chunks.
"""
from typing import List, Optional

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from starVLA.model.framework.VLA_DINO_Mamba_Diff import VLA_DINO_Mamba_Diff
from starVLA.model.modules.world_model.mamba_world_model import StreamingMambaPredictor
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)


@FRAMEWORK_REGISTRY.register("VLA_DINO_StreamingMamba")
class VLA_DINO_StreamingMamba(VLA_DINO_Mamba_Diff):
    def __init__(self, config=None, **kwargs):
        # parent (`VLA_DINO_Mamba_Diff`) builds Qwen, DINO, action_model, cond_proj,
        # and a `MambaStatePredictor`. We then REPLACE the predictor with the
        # streaming variant.
        super().__init__(config=config, **kwargs)

        mcfg = getattr(config.framework, "mamba_wm", None)
        get = (lambda k, d: getattr(mcfg, k, d)) if mcfg is not None else (lambda k, d: d)

        # Streaming Mamba-2 hyper-parameters
        sm_state_dim = int(get("stream_state_dim", 1024))
        sm_depth = int(get("stream_depth", 12))
        sm_d_state = int(get("stream_d_state", 64))     # Mamba-2 defaults to 64+
        sm_d_conv = int(get("stream_d_conv", 1))         # 1 keeps split-with-state equivalent to concat
        sm_expand = int(get("stream_expand", 2))
        sm_headdim = int(get("stream_headdim", 64))
        sm_chunk_size = int(get("stream_chunk_size", 64))
        # robot_state is consumed by the action head only (predictor is
        # vision-only), so robot_state_dim here only sizes the action-head input.
        self.robot_state_dim = int(get("robot_state_dim", 8))

        dino_dim = self.dino.num_channels                            # 768
        qwen_dim = self.qwen_vl_interface.model.config.hidden_size   # 2048

        # === Drop parent's stateless `MambaStatePredictor` and install the streaming one
        del self.mamba_predictor
        self.mamba_predictor = StreamingMambaPredictor(
            state_dim=sm_state_dim,
            action_token_dim=qwen_dim,
            dino_dim=dino_dim,
            tokens_per_frame=self.tokens_per_frame,
            depth=sm_depth,
            d_state=sm_d_state,
            d_conv=sm_d_conv,
            expand=sm_expand,
            headdim=sm_headdim,
            chunk_size=sm_chunk_size,
        )

        # Re-apply freeze (parent's freeze ran on the old predictor; we need
        # to mark the new one as trainable).
        self.freeze_backbone()

        # Inference-time SSM hidden state, carried across chunks within an
        # episode. Reset on `episode_reset` (called by the eval server).
        self._infer_states = None

        # Default seq_len (M chunks per training sample). Trainer reads `self.seq_len_M`.
        self.seq_len_M = int(get("seq_len_M", 2))

        n_pred = sum(p.numel() for p in self.mamba_predictor.parameters())
        logger.info(
            f"[StreamingMamba] Mamba-2 predictor: state_dim={sm_state_dim} depth={sm_depth} "
            f"d_state={sm_d_state} d_conv={sm_d_conv} headdim={sm_headdim} chunk_size={sm_chunk_size}"
            f" | params={n_pred / 1e6:.1f}M | seq_len_M={self.seq_len_M}"
        )

    # ------------------------------------------------------------------ trainer hooks
    @property
    def video_horizon(self):
        """Two input frames (t-7, t) + one target frame (t+7) = 3 frames."""
        return 2 * self.horizon + 1

    @property
    def obs_indices(self):
        """Frame delta indices relative to the dataloader's base_index.

        Layout:    base-H,  base,    base+H
                    ↑        ↑        ↑
                   past   present   target
        The dataloader front-pads (replicates frame 0) when base<H, which gives
        natural cold-start coverage. The action chunk is action[base:base+H]
        (default action_indices), aligned with `present`.
        """
        H = self.horizon
        return [-H, 0, H]

    # ------------------------------------------------------------------ checkpoint helpers
    def lean_save_key(self, k: str) -> bool:
        # New: predictor weights + cond_proj + action_model + Qwen LoRA adapters.
        if k.startswith(("mamba_predictor.", "cond_proj.", "action_model.")):
            return True
        if k.startswith("qwen_vl_interface.") and "lora_" in k:
            return True
        return False

    # ------------------------------------------------------------------ DINO helpers (multi-frame)
    @torch.no_grad()
    def _encode_dino_per_frame(self, frames):
        """frames: [B, V, T, H, W, 3] uint8 -> [B, T, V*N, dino_dim]. Frozen."""
        return self._dino_latents_impl(frames)

    # ------------------------------------------------------------------ PRETRAIN FORWARD (video+lang only)
    def forward_pretrain(self, batch: dict, **kwargs):
        """Video+language pretraining forward (no state, no action).

        Used for large-scale pretraining on SSv2 + Droid where action supervision
        is intentionally dropped (avoids cross-embodiment action-space mismatch).

        Batch format (from `pretrain_mixer.pretrain_collate`):
            images_past    : list[list[PIL]] length B, each inner list has V views
            images_present : same
            images_target  : same
            instructions   : list[str] length B
            datasets       : list[str] ("ssv2" or "droid") length B (unused here)

        Loss: L1(s_target_pred, s_target) + cosine metric.
        """
        device = self.cond_proj.weight.device
        B = len(batch["instructions"])
        V = len(batch["images_past"][0])

        # === Stack 3 frames into np [B, V, 3, H, W, 3] uint8 for DINO
        def _pil_to_np(pil_list_per_batch):
            # pil_list_per_batch: list[list[PIL]] length B, each with V PIL
            out = np.stack([
                np.stack([np.asarray(pil.resize((self.dino_size, self.dino_size)))
                          for pil in per_b], axis=0)   # [V, H, W, 3]
                for per_b in pil_list_per_batch
            ], axis=0)                                  # [B, V, H, W, 3]
            return out

        v_past    = _pil_to_np(batch["images_past"])    # [B, V, H, W, 3]
        v_present = _pil_to_np(batch["images_present"])
        v_target  = _pil_to_np(batch["images_target"])
        videos_np = np.stack([v_past, v_present, v_target], axis=2)  # [B, V, 3, H, W, 3]
        videos = torch.from_numpy(videos_np).to(device)

        with torch.autocast("cuda", dtype=torch.float32):
            dino_all = self._encode_dino_per_frame(videos).float()   # [B, 3, N, D]
        s_past    = dino_all[:, 0]
        s_present = dino_all[:, 1]
        s_target  = dino_all[:, 2]
        s_in      = torch.stack([s_past, s_present], dim=1)          # [B, 2, N, D]

        # === Qwen action tokens (PRESENT frame, all V views)
        instructions = batch["instructions"]
        batch_images = batch["images_present"]  # list[list[PIL]]
        # Ensure PILs are at DINO size for consistency with training forward
        batch_images_resized = [
            [pil.resize((self.dino_size, self.dino_size)) for pil in per_b]
            for per_b in batch_images
        ]
        action_tokens = self._qwen_action_tokens(batch_images_resized, instructions)

        # === Predictor (no SSM state carry)
        s_target_pred, _ = self.mamba_predictor.forward_chunk(
            action_tokens, s_in, states=None
        )

        L_pred = F.l1_loss(s_target_pred, s_target)
        cos_avg = F.cosine_similarity(s_target_pred, s_target, dim=-1).mean()

        return {
            "loss":      L_pred,
            "pred_loss": L_pred,
            "pred_cos":  cos_avg,
        }

    # ------------------------------------------------------------------ TRAINING FORWARD
    def forward(self, examples=None, **kwargs):
        # Route: pretrain batches come as a single dict with keys "images_past" /
        # "images_present" / "images_target" / "instructions". Training batches
        # come as `List[dict]` from the LIBERO dataloader. Distinguish by type.
        if isinstance(examples, dict) and "images_past" in examples:
            return self.forward_pretrain(examples, **kwargs)
        return self._forward_libero(examples, **kwargs)

    def _forward_libero(self, examples: List[dict] = None, **kwargs):
        """Two-frame world-model training.

        Per example:
          example['video']      : np.uint8 [V, 3, H, W, 3]
                                  obs_indices = [-H, 0, H] so frames are
                                  [past, present, target]. Front-padding by the
                                  dataloader replicates frame 0 for past when
                                  base<H — natural cold-start coverage.
          example['state_full'] : np.float [3, 8]   robot_state at same 3 timepoints
          example['action']     : np.float [action_horizon, action_dim]
                                  default slice action[base:base+H] = present-aligned
          example['lang']       : str

        Forward (single chunk, no SSM state carry):
          inputs = (action_tokens at present frame, robot_state at present, [s_past, s_present])
          s_target_pred = predictor.forward_chunk(...)
          L_pred  = L1(s_target_pred, s_target)
          L_action = action_model(cond(s_present, s_target_pred), action, r_present.unsqueeze(1))
        """
        device = self.cond_proj.weight.device
        B = len(examples)

        # === Encode all 3 frames with DINO
        videos = torch.from_numpy(np.stack([e["video"] for e in examples])).to(device)  # [B,V,3,H,W,3]
        with torch.autocast("cuda", dtype=torch.float32):
            dino_all = self._encode_dino_per_frame(videos).float()                       # [B, 3, N, D]

        s_past    = dino_all[:, 0]
        s_present = dino_all[:, 1]
        s_target  = dino_all[:, 2]
        s_in      = torch.stack([s_past, s_present], dim=1)                              # [B, 2, N, D]

        # === Robot state at present (index 1)
        states_all = torch.from_numpy(
            np.stack([e["state_full"] for e in examples])
        ).to(device, dtype=torch.float32)                                                # [B, 3, 8]
        r_present = states_all[:, 1]                                                     # [B, 8]

        # === Qwen action tokens — from PRESENT frame (index 1)
        from PIL import Image
        batch_images = []
        for e in examples:
            v = e["video"]                                   # np [V, 3, H, W, 3]
            pil_views = [
                Image.fromarray(v[v_i, 1]).resize((self.dino_size, self.dino_size))
                for v_i in range(v.shape[0])
            ]
            batch_images.append(pil_views)
        instructions = [e["lang"] for e in examples]
        action_tokens = self._qwen_action_tokens(batch_images, instructions)             # [B, Na, A_dim]

        # === Single-chunk predictor forward (no SSM state carry, no robot_state token)
        s_target_pred, _ = self.mamba_predictor.forward_chunk(
            action_tokens, s_in, states=None
        )

        L_pred  = F.l1_loss(s_target_pred, s_target)
        cos_avg = F.cosine_similarity(s_target_pred, s_target, dim=-1).mean()

        w = self.loss_weights
        # === Action loss (flow-matching head) — only when stage2.
        L_action = torch.zeros((), device=device)
        if w["action"] > 0 and "action" in examples[0]:
            # Dataset returns action_horizon actions starting at base+H (present)
            actions = torch.tensor(
                np.array([e["action"] for e in examples]),
                device=device, dtype=torch.float32,
            )                                                # [B, action_horizon, action_dim]
            rep = self.config.trainer.get("repeated_diffusion_steps", 4)
            cond = self._cond(s_present, s_target_pred)
            state_3d = r_present.unsqueeze(1)
            with torch.autocast("cuda", dtype=torch.float32):
                L_action = self.action_model(
                    cond.repeat(rep, 1, 1),
                    actions.repeat(rep, 1, 1),
                    state_3d.repeat(rep, 1, 1),
                )

        out = {
            "pred_loss": L_pred,
            "pred_cos": cos_avg,
            "action_loss": L_action,
        }
        out["loss"] = w["pred"] * L_pred + w["action"] * L_action
        return out

    # ------------------------------------------------------------------ TRAINING (Stage-2 truncated-BPTT)
    def forward_stage2(self, examples, state_in=None, mask_reset=None):
        """Stage-2 single-chunk forward for truncated-BPTT training.

        Each call processes ONE chunk per slot (batch element). Each slot
        belongs to a continuing episode whose previous-chunk SSM state is
        carried in `state_in`. The trainer drives the slot rotation and
        passes `mask_reset[slot]=True` for slots that just began a new
        episode (zero state for those slots).

        Parameters
        ----------
        examples   : list[dict] of length B. Each entry has:
            'video'       : np.uint8 [V, 2, H, W, 3]  — chunk-start + GT target frames
            'state_full'  : np.float [2, 8]            — proprioception aligned
            'lang'        : str
        state_in   : list[per-layer tensor] from previous step (per-slot batch dim),
                     or None for the very first step.
        mask_reset : bool tensor or list[bool] of length B. True for slots that
                     just started a new episode.

        Returns
        -------
        dict with:
            'loss'       : weighted total
            'pred_loss'  : L_pred (L1 on s_end)
            'action_loss': flow-matching loss on actions
            'pred_cos'   : cosine sim metric
            'state_out'  : list[per-layer tensor]  — caller `.detach()`s
        """
        device = self.cond_proj.weight.device
        B = len(examples)

        # === Encode (chunk-start, target) frames with DINO
        videos = torch.from_numpy(np.stack([e["video"] for e in examples])).to(device)
        with torch.autocast("cuda", dtype=torch.float32):
            dino_all = self._encode_dino_per_frame(videos).float()             # [B, 2, N, D]
        s_0 = dino_all[:, 0]                                                    # [B, N, D]
        s_end_gt = dino_all[:, 1]

        states_all = torch.from_numpy(
            np.stack([e["state_full"] for e in examples])
        ).to(device, dtype=torch.float32)                                        # [B, 2, 8]
        r_0 = states_all[:, 0]

        # === Qwen action tokens (chunk-start frame, primary views)
        from PIL import Image
        batch_images = []
        for e in examples:
            v = e["video"]
            pil_views = [
                Image.fromarray(v[v_i, 0]).resize((self.dino_size, self.dino_size))
                for v_i in range(v.shape[0])
            ]
            batch_images.append(pil_views)
        instructions = [e["lang"] for e in examples]
        action_tokens = self._qwen_action_tokens(batch_images, instructions)    # [B, Na, A_dim]

        # === Apply per-slot reset to state_in: zero-out the slots that just
        # started a new episode. Mamba-2 SSM state shape is
        # [B, nheads, headdim, d_state] per layer.
        if state_in is not None and mask_reset is not None:
            mask = torch.as_tensor(mask_reset, device=device, dtype=torch.bool)  # [B]
            if mask.any():
                state_in = [
                    torch.where(
                        mask.view(-1, 1, 1, 1), torch.zeros_like(st), st
                    )
                    for st in state_in
                ]

        # === Mamba-2 chunk forward with carried state
        s_end_pred, state_out = self.mamba_predictor.forward_chunk(
            action_tokens, s_0, states=state_in
        )

        # === Predictor loss
        L_pred = F.l1_loss(s_end_pred, s_end_gt)
        cos = F.cosine_similarity(s_end_pred, s_end_gt, dim=-1).mean()

        # === Action loss (flow-matching head)
        actions = torch.tensor(
            np.array([e["action"] for e in examples]), device=device, dtype=torch.float32
        )
        rep = self.config.trainer.get("repeated_diffusion_steps", 4)
        cond = self._cond(s_0, s_end_pred)
        state_3d = r_0.unsqueeze(1) if r_0 is not None else None   # [B, 1, 8]
        with torch.autocast("cuda", dtype=torch.float32):
            L_action = self.action_model(
                cond.repeat(rep, 1, 1),
                actions.repeat(rep, 1, 1),
                state_3d.repeat(rep, 1, 1) if state_3d is not None else None,
            )

        w = self.loss_weights
        out = {
            "pred_loss": L_pred,
            "pred_cos": cos,
            "action_loss": L_action,
            "loss": w["pred"] * L_pred + w["action"] * L_action,
            "state_out": state_out,
        }
        return out

    # ------------------------------------------------------------------ INFERENCE: episode-aware
    def episode_reset(self):
        """Call at the start of each episode.

        Two-frame world model has NO SSM state carry; we only clear the past-
        frame buffer so that t=0 uses a cold-start (past = present)."""
        self._past_frame = None

    @torch.inference_mode()
    def predict_action(self, batch_images, instructions, state=None, **kwargs):
        """Two-frame inference.

        Input contract (one call = one re-plan):
          batch_images : [[PIL_view0, PIL_view1]]   PRESENT frame views
          state        : robot_state at present     shape [B, 8] (or [B,1,8])
        Internally we hold `self._past_frame` (PIL views from the previous
        re-plan, i.e. t-H). For t=0 (no past yet) we copy present into past.
        """
        device = self.cond_proj.weight.device
        B = len(batch_images)

        # === Qwen action tokens — from PRESENT frame
        action_tokens = self._qwen_action_tokens(batch_images, instructions)

        # === Build [past_views, present_views] for DINO
        if not hasattr(self, "_past_frame") or self._past_frame is None:
            past_views = batch_images   # cold start: past = present
        else:
            past_views = self._past_frame

        def _stack(views_batch):
            out = []
            for sample in views_batch:
                vs = [torch.from_numpy(np.asarray(img.convert("RGB").resize((256, 256))))
                      for img in sample]
                out.append(torch.stack(vs))
            return torch.stack(out)                                # [B, V, 256, 256, 3]

        past_arr    = _stack(past_views)     # [B, V, 256,256,3]
        present_arr = _stack(batch_images)
        # DINO encoder expects [B,V,T,H,W,3]; T=2 (past, present).
        frames = torch.stack([past_arr, present_arr], dim=2).to(device)  # [B,V,2,H,W,3]
        dino_2 = self._dino_latents(frames)                              # [B, 2, N, D]
        s_past, s_present = dino_2[:, 0], dino_2[:, 1]
        s_in = torch.stack([s_past, s_present], dim=1)                   # [B, 2, N, D]

        # === Robot state
        if state is not None:
            r = torch.from_numpy(np.array(state)).to(device, dtype=torch.float32)
            if r.ndim > 2:
                r = r.reshape(-1, self.robot_state_dim)
        else:
            r = torch.zeros(B, self.robot_state_dim, device=device)

        # === Predictor (single chunk, no SSM state carry, no robot_state token)
        s_target_pred, _ = self.mamba_predictor.forward_chunk(
            action_tokens, s_in, states=None
        )

        # === Action head
        cond = self._cond(s_present, s_target_pred)
        r_for_head = r.unsqueeze(1) if r.ndim == 2 else r
        with torch.autocast("cuda", dtype=torch.float32):
            actions = self.action_model.predict_action(cond, r_for_head)

        # Advance the past buffer for next re-plan
        self._past_frame = batch_images

        return {"normalized_actions": actions.float().cpu().numpy()}
