"""
VLA_DINO_StreamingMamba_FutureOnly — same as StreamingMamba, but action head
conditions on the PREDICTED FUTURE latent only (not concat with present).

Rationale (ongoing research direction):
    Under sensor perturbations the present frame latent (s_present) is
    corrupted. Feeding it as part of the action-head conditioning propagates
    that corruption into action decoding. Since s_target_pred is the WM's
    prediction anchored to the clean training manifold, using it alone may
    give the action head a cleaner conditioning signal.

Only two things differ from the parent:
    1) training forward: cond = cond_proj(s_target_pred)
    2) inference:        cond = cond_proj(s_target_pred)

Everything else (predictor, LoRA, DINO, action head, robot_state routing) is
inherited unchanged.

`cond_proj` (parent: `nn.Linear(dino_dim=768, qwen_dim=2048)`) is unchanged —
it applies per token; only the sequence length halves (2N -> N).
"""
from typing import List

import numpy as np
import torch

from starVLA.model.framework.VLA_DINO_StreamingMamba import VLA_DINO_StreamingMamba
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)


@FRAMEWORK_REGISTRY.register("VLA_DINO_StreamingMamba_FutureOnly")
class VLA_DINO_StreamingMamba_FutureOnly(VLA_DINO_StreamingMamba):
    def __init__(self, config=None, **kwargs):
        super().__init__(config=config, **kwargs)
        logger.info("[FutureOnly] action head conditions on s_target_pred only "
                    "(s_present removed from cond)")

    # ------------------------------------------------------------------ TRAINING FORWARD
    def _forward_libero(self, examples: List[dict] = None, **kwargs):
        """Same as parent's _forward_libero, but the action head sees only
        the predicted future latent (s_present is NOT concatenated)."""
        import torch.nn.functional as F
        from PIL import Image

        device = self.cond_proj.weight.device

        # === Encode all 3 frames with DINO
        videos = torch.from_numpy(np.stack([e["video"] for e in examples])).to(device)
        with torch.autocast("cuda", dtype=torch.float32):
            dino_all = self._encode_dino_per_frame(videos).float()   # [B, 3, N, D]

        s_past    = dino_all[:, 0]
        s_present = dino_all[:, 1]
        s_target  = dino_all[:, 2]
        s_in      = torch.stack([s_past, s_present], dim=1)          # [B, 2, N, D]

        # === Robot state at present
        states_all = torch.from_numpy(
            np.stack([e["state_full"] for e in examples])
        ).to(device, dtype=torch.float32)                             # [B, 3, 8]
        r_present = states_all[:, 1]                                  # [B, 8]

        # === Qwen action tokens — from PRESENT frame
        batch_images = []
        for e in examples:
            v = e["video"]
            pil_views = [
                Image.fromarray(v[v_i, 1]).resize((self.dino_size, self.dino_size))
                for v_i in range(v.shape[0])
            ]
            batch_images.append(pil_views)
        instructions = [e["lang"] for e in examples]
        action_tokens = self._qwen_action_tokens(batch_images, instructions)

        # === Predictor (unchanged: still sees past + present)
        s_target_pred, _ = self.mamba_predictor.forward_chunk(
            action_tokens, s_in, states=None
        )

        L_pred  = F.l1_loss(s_target_pred, s_target)
        cos_avg = F.cosine_similarity(s_target_pred, s_target, dim=-1).mean()

        w = self.loss_weights
        L_action = torch.zeros((), device=device)
        if w["action"] > 0 and "action" in examples[0]:
            actions = torch.tensor(
                np.array([e["action"] for e in examples]),
                device=device, dtype=torch.float32,
            )
            rep = self.config.trainer.get("repeated_diffusion_steps", 4)
            # === KEY CHANGE: condition on PREDICTED FUTURE only
            cond = self.cond_proj(s_target_pred)                     # [B, N, qwen_dim]
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

    # ------------------------------------------------------------------ INFERENCE
    @torch.inference_mode()
    def predict_action(self, batch_images, instructions, state=None, **kwargs):
        """Same as parent's predict_action, but cond = cond_proj(s_target_pred)."""
        device = self.cond_proj.weight.device
        B = len(batch_images)

        action_tokens = self._qwen_action_tokens(batch_images, instructions)

        # Build [past, present] frame stack for the predictor
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
            return torch.stack(out)

        past_arr    = _stack(past_views)
        present_arr = _stack(batch_images)
        frames = torch.stack([past_arr, present_arr], dim=2).to(device)  # [B,V,2,H,W,3]
        dino_2 = self._dino_latents(frames)                              # [B, 2, N, D]
        s_past, s_present = dino_2[:, 0], dino_2[:, 1]
        s_in = torch.stack([s_past, s_present], dim=1)                   # [B, 2, N, D]

        # Robot state (still goes to action head as a separate token)
        if state is not None:
            r = torch.from_numpy(np.array(state)).to(device, dtype=torch.float32)
            if r.ndim > 2:
                r = r.reshape(-1, self.robot_state_dim)
        else:
            r = torch.zeros(B, self.robot_state_dim, device=device)

        # Predictor (unchanged)
        s_target_pred, _ = self.mamba_predictor.forward_chunk(
            action_tokens, s_in, states=None
        )

        # === KEY CHANGE: condition on PREDICTED FUTURE only
        cond = self.cond_proj(s_target_pred)
        r_for_head = r.unsqueeze(1) if r.ndim == 2 else r
        with torch.autocast("cuda", dtype=torch.float32):
            actions = self.action_model.predict_action(cond, r_for_head)

        # Advance past buffer
        self._past_frame = batch_images

        return {"normalized_actions": actions.float().cpu().numpy()}
