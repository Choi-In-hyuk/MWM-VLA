# Copyright 2026 VLA-JEPA research. MIT License.
"""
VLA_DINO_DualMamba — dual-Mamba pipeline.

  current frame  --DINO(frozen)--> s_0
  lang + current --Qwen(LoRA)----> action tokens   (semantic intent)
  (s_0, tokens)  --Mamba #1-------> s_end          (predicted endpoint latent)
  (s_0, s_end, robot_state) --Mamba #2 (flow-matching) -> action chunk

Compared to VLA_DINO_Mamba_Diff (V2+LoRA), the only change is the action head:
the DiT flow-matching decoder is replaced by a Mamba decoder that consumes a
short self-attention-style sequence

  [ sigma | LN(s_end - s_0) -> Linear | robot_state | noised_action ]

No language conditioning enters the action head — Mamba #1 has already
absorbed Qwen's intent into s_end, so Mamba #2 focuses purely on movement
(how to drive the robot from s_0 to s_end).

Stages
------
predictor : Mamba #1 + Qwen LoRA (L_pred only) — identical to V2+LoRA stage 1.
stage2    : predictor (fine-tune) + Mamba #2 (scratch) + Qwen LoRA;
            L = alpha * L_pred + beta * L_action.

The action head is trained from scratch in stage2, so this stage is run for
substantially more steps than V2 stage2 (e.g. 100k vs 15k).
"""
from typing import List

import numpy as np
import torch
import torch.nn.functional as F

from starVLA.model.framework.VLA_DINO_Mamba_Diff import VLA_DINO_Mamba_Diff
from starVLA.model.modules.action_model.MambaActionHead import MambaActionHead
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)


@FRAMEWORK_REGISTRY.register("VLA_DINO_DualMamba")
class VLA_DINO_DualMamba(VLA_DINO_Mamba_Diff):
    def __init__(self, config=None, **kwargs):
        super().__init__(config=config, **kwargs)   # builds Qwen + Mamba #1 + DiT head + cond_proj

        mcfg = getattr(config.framework, "mamba_wm", None)
        get = (lambda k, d: getattr(mcfg, k, d)) if mcfg is not None else (lambda k, d: d)

        dino_dim = self.dino.num_channels
        action_dim = self.config.framework.action_model.action_dim
        # match base diffusion head's action_horizon (past + current + future)
        action_horizon = self.chunk_len

        # Replace the inherited DiT flow-matching head with a Mamba decoder.
        # cond_proj from the parent is unused here (head takes raw DINO latents).
        del self.action_model
        self.action_model = MambaActionHead(
            latent_dim=dino_dim,
            action_dim=action_dim,
            action_horizon=action_horizon,
            robot_state_dim=int(get("robot_state_dim", 8)),
            embed_dim=int(get("action_embed_dim", 256)),
            n_layer=int(get("action_mamba_layers", 5)),
            num_inference_timesteps=int(get("action_inference_steps", 10)),
        )

        # Re-run freeze logic so the newly-created action head's params are
        # registered as trainable (super().__init__ already called freeze_backbone
        # but at that point self.action_model was still the DiT head).
        self.freeze_backbone()

    # Lean intermediate save: keep all trained params (Mamba #1, Mamba #2, and
    # the Qwen LoRA adapters), so stage2 can resume cleanly from stage1.
    # `cond_proj` is inherited from the parent but unused here.
    _LEAN_PREFIXES = ("mamba_predictor.", "action_model.")

    def lean_save_key(self, k: str) -> bool:
        if k.startswith(self._LEAN_PREFIXES):
            return True
        # Qwen LoRA adapter weights live under qwen_vl_interface.model.*lora_*
        if k.startswith("qwen_vl_interface.") and "lora_" in k:
            return True
        return False

    # ------------------------------------------------------------------ forward
    def forward(self, examples: List[dict] = None, **kwargs):
        device = self.cond_proj.weight.device
        batch_images = [e["image"] for e in examples]
        instructions = [e["lang"] for e in examples]
        videos = torch.from_numpy(np.stack([e["video"] for e in examples])).to(device)
        frames = videos[:, :, [0, self.endpoint]]            # frame 0 and frame H

        with torch.autocast("cuda", dtype=torch.float32):
            if self.train_dino and self.loss_weights["action"] > 0:
                s_0 = self._dino_latents_grad(frames[:, :, [0]]).float()[:, 0]
                s_end_gt = self._dino_latents(frames[:, :, [1]]).float()[:, 0].detach()
            else:
                s = self._dino_latents(frames).float()
                s_0, s_end_gt = s[:, 0], s[:, 1]

        w = self.loss_weights
        if self._qwen_cache is not None and "cache_key" in examples[0]:
            action_tokens = self._cached_action_tokens(examples, device)
        else:
            action_tokens = self._qwen_action_tokens(batch_images, instructions)

        s_end_pred = self.mamba_predictor(s_0, action_tokens)[:, 0]   # [B, N, D]
        L_pred = F.l1_loss(s_end_pred, s_end_gt)
        out = {"pred_cos": F.cosine_similarity(s_end_pred, s_end_gt, dim=-1).mean()}

        L_action = torch.zeros((), device=device)
        if w["action"] > 0:
            actions = torch.tensor(
                np.array([e["action"] for e in examples]), device=device, dtype=torch.float32,
            )
            state = None
            if "state" in examples[0]:
                state = torch.tensor(
                    np.array([e["state"] for e in examples]), device=device, dtype=torch.float32,
                )
            rep = self.config.trainer.get("repeated_diffusion_steps", 4)
            with torch.autocast("cuda", dtype=torch.float32):
                L_action = self.action_model(
                    s_0.repeat(rep, 1, 1),
                    s_end_pred.repeat(rep, 1, 1),
                    actions.repeat(rep, 1, 1),
                    state.repeat(rep, 1) if state is not None else None,
                )

        out["pred_loss"], out["action_loss"] = L_pred, L_action
        out["loss"] = w["pred"] * L_pred + w["action"] * L_action
        return out

    # ------------------------------------------------------------------ inference
    @torch.inference_mode()
    def predict_action(self, batch_images, instructions, state=None, **kwargs):
        device = self.cond_proj.weight.device
        action_tokens = self._qwen_action_tokens(batch_images, instructions)
        views = []
        for sample in batch_images:
            vs = [torch.from_numpy(np.asarray(img.convert("RGB").resize((256, 256)))) for img in sample]
            views.append(torch.stack(vs))
        frame0 = torch.stack(views).unsqueeze(2).to(device)
        s_0 = self._dino_latents(frame0)[:, 0]
        s_end = self.mamba_predictor(s_0, action_tokens)[:, 0]
        st = None
        if state is not None:
            st = torch.from_numpy(np.array(state)).to(device, dtype=torch.float32)
        with torch.autocast("cuda", dtype=torch.float32):
            actions = self.action_model.predict_action(s_0, s_end, st)
        return {"normalized_actions": actions.float().cpu().numpy()}
