"""FutureOnly + Denoise + robot-state as Mamba initial hidden state (+ future-state aux loss).

Motivation
----------
In the base FutureOnly/Denoise model the predictor is called with
`states=None` — the Mamba SSM starts from a ZERO hidden state and robot_state
is fed only to the action head, where it is redundant with the images (the arm
is visible) and gets ignored (dropping state didn't change performance).

This variant makes the predictor actually USE proprioception, two ways at once:

  (1) INJECTION — robot_state at present is encoded into the Mamba SSM's
      *initial hidden state* (per layer). Instead of rolling the future latent
      out from a blank memory, the predictor starts from "the robot is in THIS
      physical pose". Same prediction task as before (predict CLEAN future
      DINO), only the starting memory changes.

  (2) PRESSURE — an auxiliary head predicts the FUTURE robot_state
      (states_all[:, 2]) from the predicted future latent, with L_state. This
      forces the predictor to encode/carry the proprioceptive signal (otherwise
      it can't hit the future-state target), so the injected state can't just be
      ignored the way the action-head token was.

Everything else (augmentation → clean-future denoising, frozen backbone, Qwen
LoRA, action head) is inherited unchanged.

Injection design note
---------------------
Each Mamba layer's SSM state is [B, nheads, headdim, d_state] (for our config
32*64*64 = 131072 elems/layer, 12 layers). Mapping 8-D robot_state straight to
that is a parameter blow-up, so we generate a low-rank per-(layer, head, d_state)
factor and broadcast over headdim. The final projection is near-zero
initialized so training STARTS equivalent to the zero-init baseline and learns
to lean on state gradually (keeps the SSM stable early).
"""
import random
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from starVLA.model.framework.VLA_DINO_StreamingMamba_FutureOnly_Denoise import (
    VLA_DINO_StreamingMamba_FutureOnly_Denoise,
)
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)


@FRAMEWORK_REGISTRY.register("VLA_DINO_StreamingMamba_FutureOnly_Denoise_StateInit")
class VLA_DINO_StreamingMamba_FutureOnly_Denoise_StateInit(
    VLA_DINO_StreamingMamba_FutureOnly_Denoise
):
    def __init__(self, config=None, **kwargs):
        super().__init__(config=config, **kwargs)

        mcfg = getattr(config.framework, "mamba_wm", None)
        get = (lambda k, d: getattr(mcfg, k, d)) if mcfg is not None else (lambda k, d: d)
        self.robot_state_dim = int(get("robot_state_dim", 8))
        # lambda for L_state. Kept below 1 so the DINO future prediction (L_pred)
        # stays the primary objective; the normalized-delta L_state now sits at
        # ~1-3 magnitude (vs the old ~0.002), so 0.5 gives it real but secondary
        # pressure rather than letting state prediction dominate the loss.
        self.state_aux_weight = float(get("state_aux_weight", 0.5))   # lambda for L_state
        self.state_init_scale = float(get("state_init_scale", 1.0))  # optional overall gain
        # Typical |future - present| state delta over the prediction horizon
        # (measured ~0.022 mean-abs on LIBERO). We divide the delta target by
        # this so the normalized target is ~unit-scale and smooth_l1 gradients
        # are meaningful instead of vanishing on a tiny raw delta.
        self.state_delta_scale = float(get("state_delta_scale", 0.03))

        # Predictor already exists (built in StreamingMamba.__init__ via super()).
        # Build state modules NOW so they are visible to the optimizer (which
        # collects trainable params before the first forward) and to set_stage.
        self._build_state_modules()

        logger.info(
            f"[Denoise_StateInit] robot_state -> Mamba initial hidden state "
            f"(inject) + future-state aux loss (lambda={self.state_aux_weight}). "
            f"robot_state_dim={self.robot_state_dim}"
        )

    # ------------------------------------------------------------------ build
    def _build_state_modules(self):
        """Create state encoder + future-state head from predictor SSM dims."""
        p = self.mamba_predictor
        blocks = p.blocks
        self.depth = len(blocks)
        b0 = blocks[0]
        self.nheads = b0.nheads
        self.headdim = b0.headdim
        self.d_state = b0.d_state
        self.state_dim = p.state_dim
        self.dino_dim = p.dino_dim

        # 8 -> [depth * nheads * d_state]; broadcast over headdim at use time.
        # Low-rank hidden, near-zero final proj so t=0 ~= zero-init baseline.
        hidden = 256
        out_dim = self.depth * self.nheads * self.d_state
        self.state_encoder = nn.Sequential(
            nn.Linear(self.robot_state_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )
        nn.init.zeros_(self.state_encoder[-1].weight)
        nn.init.zeros_(self.state_encoder[-1].bias)

        # future latent [B, N, dino_dim] -> mean-pool -> future robot_state [B, 8]
        self.state_pred_head = nn.Sequential(
            nn.Linear(self.dino_dim, 256),
            nn.GELU(),
            nn.Linear(256, self.robot_state_dim),
        )
        n = sum(pm.numel() for pm in self.state_encoder.parameters()) + \
            sum(pm.numel() for pm in self.state_pred_head.parameters())
        logger.info(
            f"[Denoise_StateInit] built state modules: depth={self.depth} "
            f"nheads={self.nheads} headdim={self.headdim} d_state={self.d_state} "
            f"({n/1e6:.2f}M params)"
        )

    # ------------------------------------------------------------------ trainable set
    def _trainable_mods(self):
        # extend parent's set so set_stage's freeze-all pass also covers our modules.
        # Guard: parent __init__ may call this before our modules exist.
        mods = super()._trainable_mods()
        extra = tuple(
            m for m in (getattr(self, "state_encoder", None),
                        getattr(self, "state_pred_head", None))
            if m is not None
        )
        return mods + extra

    def set_stage(self, stage: str):
        """Same stage schedule as parent, but ALSO train the state modules in
        BOTH stages (they must learn alongside the predictor so the injected
        initial-state + future-state prediction are actually used). Parent only
        flips requires_grad on its own `train` tuple, so we turn ours on here."""
        super().set_stage(stage)
        for m in (self.state_encoder, self.state_pred_head):
            for p in m.parameters():
                p.requires_grad_(True)
            m.train()
        # expose L_state weight so the trainer logs it (it shows nonzero-weight
        # components of loss_weights) and so it's part of the recorded schedule.
        self.loss_weights["state"] = self.state_aux_weight
        logger.info("[Denoise_StateInit] set_stage: state modules trainable, "
                    f"loss_weights={self.loss_weights}")
        return self

    def _make_initial_states(self, r_present):
        """robot_state [B, 8] -> list(depth) of [B, nheads, headdim, d_state].

        We produce a [B, depth, nheads, d_state] factor and broadcast it across
        the headdim axis (each head-channel shares the proprio bias). near-zero
        init => early training this is ~0 == the zero-init baseline.
        """
        B = r_present.shape[0]
        factor = self.state_encoder(r_present)                       # [B, depth*nheads*d_state]
        factor = factor.view(B, self.depth, self.nheads, self.d_state)
        factor = factor * self.state_init_scale
        states = []
        for l in range(self.depth):
            # [B, nheads, d_state] -> [B, nheads, headdim, d_state] (broadcast headdim)
            s = factor[:, l].unsqueeze(2).expand(B, self.nheads, self.headdim, self.d_state)
            states.append(s.contiguous())
        return states

    # ------------------------------------------------------------------ forward
    def _forward_libero(self, examples: List[dict] = None, **kwargs):
        from PIL import Image

        device = self.cond_proj.weight.device

        videos = torch.from_numpy(np.stack([e["video"] for e in examples])).to(device)
        if self.training and self.aug_prob > 0.0:
            videos = self._augment_videos(videos)

        with torch.autocast("cuda", dtype=torch.float32):
            dino_all = self._encode_dino_per_frame(videos).float()   # [B, 3, N, D]

        s_past    = dino_all[:, 0]
        s_present = dino_all[:, 1]
        s_target  = dino_all[:, 2]                                    # CLEAN target
        s_in      = torch.stack([s_past, s_present], dim=1)          # [B, 2, N, D]

        states_all = torch.from_numpy(
            np.stack([e["state_full"] for e in examples])
        ).to(device, dtype=torch.float32)                             # [B, 3, 8]
        r_present = states_all[:, 1]                                  # [B, 8]
        r_future  = states_all[:, 2]                                  # [B, 8]
        # L_state target = the CHANGE the robot undergoes, not the absolute
        # future pose. Future pose ~= present pose (over ~7 frames the arm moves
        # only ~4% of the state magnitude), so predicting absolute future is
        # trivial ("copy present") and gives almost no learning pressure -- we
        # saw L_state collapse to ~0.002. The delta (future - present) is the
        # actual dynamics signal; normalizing by its typical scale makes the
        # smooth_l1 magnitude meaningful so gradients don't vanish.
        r_delta   = (r_future - r_present) / self.state_delta_scale   # [B, 8]

        # Qwen action tokens from the (possibly augmented) present frame
        batch_images = []
        for b, e in enumerate(examples):
            v = videos[b].cpu().numpy()
            pil_views = [
                Image.fromarray(v[v_i, 1]).resize((self.dino_size, self.dino_size))
                for v_i in range(v.shape[0])
            ]
            batch_images.append(pil_views)
        instructions = [e["lang"] for e in examples]
        action_tokens = self._qwen_action_tokens(batch_images, instructions)

        # (1) INJECTION: robot_state -> Mamba initial hidden state
        init_states = self._make_initial_states(r_present.to(self.cond_proj.weight.dtype))

        s_target_pred, _ = self.mamba_predictor.forward_chunk(
            action_tokens, s_in, states=init_states
        )

        L_pred  = F.l1_loss(s_target_pred, s_target)
        cos_avg = F.cosine_similarity(s_target_pred, s_target, dim=-1).mean()

        # (2) PRESSURE: predict the (normalized) future state DELTA from the
        # predicted future latent. Forces the predictor to encode how the robot
        # will move, which it can only do by using the injected present state.
        pooled = s_target_pred.mean(dim=1)                            # [B, dino_dim]
        r_delta_pred = self.state_pred_head(pooled.to(self.state_pred_head[0].weight.dtype))
        L_state = F.smooth_l1_loss(r_delta_pred, r_delta.to(r_delta_pred.dtype))

        w = self.loss_weights
        L_action = torch.zeros((), device=device)
        if w["action"] > 0 and "action" in examples[0]:
            actions = torch.tensor(
                np.array([e["action"] for e in examples]),
                device=device, dtype=torch.float32,
            )
            rep = self.config.trainer.get("repeated_diffusion_steps", 4)
            cond = self.cond_proj(s_target_pred)
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
            "state_loss": L_state,
        }
        out["loss"] = (
            w["pred"] * L_pred
            + w["action"] * L_action
            + self.state_aux_weight * L_state
        )
        return out
