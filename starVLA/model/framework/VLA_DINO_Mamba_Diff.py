# Copyright 2026 VLA-JEPA research. MIT License.
"""
VLA_DINO_Mamba_Diff (V2) — VLM-conditioned latent world model + diffusion action head.

Pipeline:
  current frame   --DINO(frozen)-->            s_0           (current latent)
  lang + current  --Qwen(frozen)-->            action tokens (intent: "how it changes")
  (s_0, tokens)   --Mamba predictor(learned)-> s_end         (chunk-END latent, current+H)
  (s_0, s_end, robot state) --FlowmatchingActionHead(learned)-> action chunk

Why endpoint (s_end) not per-frame: consecutive DINO latents (50ms) are nearly identical
-> the action effect is below DINO resolution -> per-frame inverse dynamics is ambiguous
(V1 got 20% on LIBERO-10). Predicting the chunk-END latent gives a LARGE, learnable change;
a diffusion head decomposes the (s_0 -> s_end) transition into the H-action chunk.

Trainable : Mamba predictor, a small DINO->head projection, FlowmatchingActionHead.
Frozen    : DINO, Qwen-VL.

Two-stage training:
  predictor : L_pred only (Mamba predicts s_end ~= DINO(frame_H)).
  stage2    : predictor fine-tuned (NOT frozen) + head; L = a*L_pred + b*L_action,
              head conditioned on the PREDICTED s_end (inference-consistent).
"""
from typing import List, Optional

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from starVLA.model.framework.VLA_JEPA import VLA_JEPA
from starVLA.model.modules.world_model.mamba_world_model import MambaStatePredictor
from starVLA.model.modules.dino_model.dino import get_dino_model
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)


@FRAMEWORK_REGISTRY.register("VLA_DINO_Mamba_Diff")
class VLA_DINO_Mamba_Diff(VLA_JEPA):
    def __init__(self, config=None, **kwargs):
        super().__init__(config=config, **kwargs)   # builds Qwen + (inherited) FlowmatchingActionHead

        mcfg = getattr(config.framework, "mamba_wm", None)
        get = (lambda k, d: getattr(mcfg, k, d)) if mcfg is not None else (lambda k, d: d)
        self.dino_name = get("dino_backbone", "dinov2_vitb14")

        self.dino = get_dino_model(self.dino_name)            # frozen per-frame encoder
        dino_dim = self.dino.num_channels                     # vitb14 -> 768
        num_views = 2
        self.dino_size = 224
        tokens_per_view = (self.dino_size // 14) ** 2          # 256
        self.tokens_per_frame = num_views * tokens_per_view    # 512
        self.num_views = num_views

        qwen_dim = self.qwen_vl_interface.model.config.hidden_size  # 2048
        self.horizon = self.future_action_window_size + 1     # 7 actions -> endpoint = frame H
        self.endpoint = self.horizon                          # frame index of s_end (current+H)

        # endpoint predictor: (s_0, action tokens) -> single future latent s_end
        self.mamba_predictor = MambaStatePredictor(
            state_dim=dino_dim, action_token_dim=qwen_dim,
            tokens_per_frame=self.tokens_per_frame, horizon=1,   # ONE endpoint
            depth=get("predictor_depth", 8),
        )
        # project DINO latents (s_0, s_end) into the diffusion head's cross-attention dim
        # (VLA_JEPA.__init__ sets diffusion cross_attention_dim = Qwen hidden = qwen_dim)
        self.cond_proj = nn.Linear(dino_dim, qwen_dim)

        self.alpha = get("pred_weight", 1.0)    # L_pred weight (stage2)
        self.beta = get("action_weight", 1.0)   # L_action weight (stage2)
        self.loss_weights = {"pred": 1.0, "action": 0.0}

        self.register_buffer("dino_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("dino_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)
        self._qwen_cache = None
        self._qwen_lora = False
        # A-b (MambaVLA-style encoder learning): unfreeze DINO in stage2 only, anchored by the
        # action loss; the L_pred TARGET (s_end_gt) is detached -> no collapse (no EMA needed).
        self.train_dino = bool(get("train_dino", False))
        # eval/from_pretrained path: re-apply LoRA so the saved LoRA state_dict matches.
        # (training applies LoRA via the trainer AFTER load_backbone, with qwen_lora absent here.)
        if get("qwen_lora", False):
            self.apply_qwen_lora(r=get("lora_r", 16), alpha=get("lora_alpha", 32))
        self.freeze_backbone()

    def apply_qwen_lora(self, r=16, alpha=32, dropout=0.05,
                        targets=("q_proj", "k_proj", "v_proj", "o_proj")):
        """Wrap frozen (already fine-tuned) Qwen with LoRA adapters. Call AFTER load_backbone."""
        from peft import LoraConfig, get_peft_model
        cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout,
                         target_modules=list(targets), bias="none")
        self.qwen_vl_interface.model = get_peft_model(self.qwen_vl_interface.model, cfg)
        self._qwen_lora = True
        n = sum(p.numel() for n_, p in self.qwen_vl_interface.named_parameters() if "lora_" in n_)
        logger.info(f"[qwen_lora] applied LoRA r={r} a={alpha} targets={targets} ({n/1e6:.2f}M adapter params)")
        return self

    def _set_qwen_lora_trainable(self, flag: bool):
        if not self._qwen_lora:
            return
        for n_, p in self.qwen_vl_interface.named_parameters():
            p.requires_grad_(flag and ("lora_" in n_))

    # ------------------------------------------------------------------ freezing / stages
    def _trainable_mods(self):
        return (self.mamba_predictor, self.cond_proj, self.action_model)

    def freeze_backbone(self):
        for p in self.parameters():
            p.requires_grad_(False)
        for m in self._trainable_mods():
            for p in m.parameters():
                p.requires_grad_(True)

    def set_stage(self, stage: str):
        """predictor: only Mamba predictor, L_pred. stage2: predictor(fine-tune)+proj+head, L_pred+L_action."""
        for m in self._trainable_mods():
            for p in m.parameters():
                p.requires_grad_(False)
        if stage == "predictor":
            train = (self.mamba_predictor,)
            self.loss_weights = {"pred": 1.0, "action": 0.0}
        elif stage == "stage2":
            train = (self.mamba_predictor, self.cond_proj, self.action_model)
            self.loss_weights = {"pred": self.alpha, "action": self.beta}
        else:
            raise ValueError(f"unknown stage: {stage}")
        for m in train:
            for p in m.parameters():
                p.requires_grad_(True)
        if self.train_dino and stage == "stage2":   # A-b: unfreeze encoder (action-anchored)
            for p in self.dino.parameters():
                p.requires_grad_(True)
            self.dino.train()
            logger.info("[set_stage] train_dino=True -> DINO unfrozen (stage2, action-anchored)")
        self._set_qwen_lora_trainable(True)   # Qwen LoRA trains in BOTH stages
        logger.info(f"[set_stage] stage={stage} loss_weights={self.loss_weights} qwen_lora={self._qwen_lora}")
        return self

    def load_backbone(self, ckpt_path: str):
        sd = torch.load(ckpt_path, map_location="cpu")
        missing, unexpected = self.load_state_dict(sd, strict=False)
        new = ("mamba_predictor", "cond_proj", "dino")
        leaked = [k for k in missing if not k.startswith(new)]
        if leaked:
            logger.warning(f"[load_backbone] non-new missing keys: {leaked[:8]} ...")
        logger.info(f"[load_backbone] loaded backbone; {len(missing)} new params fresh")
        self.freeze_backbone()
        return self

    def load_qwen_cache(self, cache_dir: str):
        import json
        from pathlib import Path
        cache_dir = Path(cache_dir)
        meta = json.load(open(cache_dir / "qwen_meta.json"))
        mem = np.memmap(cache_dir / "qwen.dat", dtype=np.float16, mode="r",
                        shape=(meta["N"], meta["Na"], meta["H"]))
        self._qwen_cache = {"key_to_row": meta["key_to_row"], "mem": mem}
        logger.info(f"[qwen_cache] loaded {meta['N']} windows from {cache_dir}")
        return self

    # ------------------------------------------------------------------ encoders
    def _normalize_frame(self, frames_uint8):
        x = frames_uint8.float() / 255.0
        x = x.permute(*range(x.dim() - 3), x.dim() - 1, x.dim() - 3, x.dim() - 2)
        shape = x.shape
        x = x.reshape(-1, 3, shape[-2], shape[-1])
        return x.reshape(*shape)

    @torch.no_grad()
    def _dino_latents(self, frames):
        """frames: [B,V,T,H,W,3] uint8 -> [B,T, V*tok, dim]. Frozen (no grad)."""
        return self._dino_latents_impl(frames)

    def _dino_latents_grad(self, frames):
        """Grad-enabled DINO encode (A-b: train the encoder)."""
        return self._dino_latents_impl(frames)

    def _dino_latents_impl(self, frames):
        B, V, T, H, W, _ = frames.shape
        x = frames.float() / 255.0
        x = x.permute(0, 1, 2, 5, 3, 4).reshape(B * V * T, 3, H, W)
        x = F.interpolate(x, size=(self.dino_size, self.dino_size), mode="bilinear", align_corners=False)
        x = (x - self.dino_mean) / self.dino_std
        feats = self.dino(x)
        tok, dim = feats.shape[1], feats.shape[2]
        return feats.reshape(B, V, T, tok, dim).permute(0, 2, 1, 3, 4).reshape(B, T, V * tok, dim)

    def _qwen_action_tokens(self, batch_images, instructions):
        replace = {"{actions}": self.replace_prompt, "{e_actions}": self.embodied_replace_prompt}
        template = self.config.datasets.vla_data.get("CoT_prompt", "")
        with torch.no_grad():   # tokenization only
            qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
                images=batch_images, instructions=instructions,
                prompt_replace_dict=replace, prompt_template=template)
        ids = qwen_inputs["input_ids"]
        idx = torch.isin(ids, torch.tensor(self.action_token_ids, device=ids.device)).nonzero(as_tuple=True)
        # grad flows into Qwen only when LoRA is being trained
        with torch.set_grad_enabled(self._qwen_lora), torch.autocast("cuda", dtype=torch.bfloat16):
            out = self.qwen_vl_interface(**qwen_inputs, output_hidden_states=True, return_dict=True)
        last = out.hidden_states[-1]
        B = last.shape[0]
        return last[idx[0], idx[1], :].view(B, -1, last.shape[-1]).float()

    def _cached_action_tokens(self, examples, device):
        c = self._qwen_cache
        rows = [c["key_to_row"][e["cache_key"]] for e in examples]
        return torch.from_numpy(np.ascontiguousarray(c["mem"][rows])).to(device, dtype=torch.float32)

    def _cond(self, s_0, s_end):
        """(s_0, s_end) DINO latents [B,N,D] -> diffusion head conditioning [B, 2N, cross_dim]."""
        return self.cond_proj(torch.cat([s_0, s_end], dim=1))

    # ------------------------------------------------------------------ forward
    def forward(self, examples: List[dict] = None, **kwargs):
        device = self.cond_proj.weight.device
        batch_images = [e["image"] for e in examples]
        instructions = [e["lang"] for e in examples]
        videos = torch.from_numpy(np.stack([e["video"] for e in examples])).to(device)  # [B,V,T,256,256,3]
        frames = videos[:, :, [0, self.endpoint]]            # frame 0 and frame H

        with torch.autocast("cuda", dtype=torch.float32):
            if self.train_dino and self.loss_weights["action"] > 0:
                # A-b: encode s_0 WITH grad (encoder learns via action anchor); the L_pred
                # target s_end_gt stays no-grad + detached -> action-grounded, collapse-free.
                s_0 = self._dino_latents_grad(frames[:, :, [0]]).float()[:, 0]
                s_end_gt = self._dino_latents(frames[:, :, [1]]).float()[:, 0].detach()
            else:
                s = self._dino_latents(frames).float()       # [B,2,N,D]
                s_0, s_end_gt = s[:, 0], s[:, 1]

        w = self.loss_weights
        need_tokens = True   # predictor always needs action tokens
        if self._qwen_cache is not None and "cache_key" in examples[0]:
            action_tokens = self._cached_action_tokens(examples, device)
        else:
            action_tokens = self._qwen_action_tokens(batch_images, instructions)

        # change-aware context masking (training only, predictor stage).
        # Hide patches with large s_0->s_end_gt change with a soft top-K sampler,
        # forcing the predictor to actually predict change instead of copying s_0.
        # No effect on the eval path (predict_action does not pass vis_idx).
        mask_ratio = float(os.environ.get("CHANGE_MASK_RATIO", "0.0"))
        # Optional linear ease-out: during the last `ramp_frac` of training,
        # decay mask_ratio linearly to 0 so the predictor sees the inference
        # (no-mask) distribution before training ends.
        ramp_frac = float(os.environ.get("CHANGE_MASK_RAMP", "0.0"))
        if ramp_frac > 0.0 and hasattr(self, "_step") and hasattr(self, "_max_steps"):
            prog = self._step / max(1, self._max_steps)
            if prog > 1.0 - ramp_frac:
                decay = (prog - (1.0 - ramp_frac)) / ramp_frac        # 0 -> 1
                mask_ratio = mask_ratio * max(0.0, 1.0 - decay)
        vis_idx = None
        if self.training and mask_ratio > 0.0 and w["pred"] > 0:
            B, N, D = s_0.shape
            Nv = max(1, int(round(N * (1.0 - mask_ratio))))
            with torch.no_grad():
                delta = (s_end_gt - s_0).pow(2).sum(dim=-1)          # [B,N]
                temp = float(os.environ.get("CHANGE_MASK_TEMP", "1.0"))
                # Higher delta -> higher prob of being hidden. We sample WITHOUT
                # replacement: keep N-Nv (= Nm) patches with highest hide-score
                # noisy-rank (Gumbel), then visible = remaining Nv.
                norm = (delta - delta.mean(dim=1, keepdim=True)) / (delta.std(dim=1, keepdim=True) + 1e-6)
                hide_score = norm / max(temp, 1e-3)
                # Gumbel-top-k: noisy argmax = sampling without replacement
                g = -torch.log(-torch.log(torch.rand(B, N, device=device).clamp_min(1e-12)))
                noisy = hide_score + g
                # take top (N-Nv) as hidden -> visible = the rest
                Nm = N - Nv
                hide_idx = torch.topk(noisy, k=Nm, dim=1).indices       # [B,Nm]
                # build visible mask
                mask = torch.ones(B, N, dtype=torch.bool, device=device)
                mask.scatter_(1, hide_idx, False)
                vis_idx = mask.nonzero(as_tuple=False)[:, 1].view(B, Nv)

        s_end_pred = self.mamba_predictor(s_0, action_tokens, vis_idx=vis_idx)[:, 0]   # [B,N,D]
        L_pred = F.l1_loss(s_end_pred, s_end_gt)
        out = {"pred_cos": F.cosine_similarity(s_end_pred, s_end_gt, dim=-1).mean()}

        L_action = torch.zeros((), device=device)
        if w["action"] > 0:
            actions = torch.tensor(np.array([e["action"] for e in examples]), device=device, dtype=torch.float32)
            state = None
            if "state" in examples[0]:
                state = torch.tensor(np.array([e["state"] for e in examples]), device=device, dtype=torch.float32)
            rep = self.config.trainer.get("repeated_diffusion_steps", 4)
            cond = self._cond(s_0, s_end_pred)                       # condition on PREDICTED s_end
            with torch.autocast("cuda", dtype=torch.float32):
                L_action = self.action_model(cond.repeat(rep, 1, 1),
                                             actions.repeat(rep, 1, 1),
                                             state.repeat(rep, 1, 1) if state is not None else None)

        out["pred_loss"], out["action_loss"] = L_pred, L_action
        out["loss"] = w["pred"] * L_pred + w["action"] * L_action
        return out

    # ------------------------------------------------------------------ inference
    @torch.inference_mode()
    def predict_action(self, batch_images, instructions, state=None, **kwargs):
        device = self.cond_proj.weight.device
        action_tokens = self._qwen_action_tokens(batch_images, instructions)
        size = self.dino_size
        views = []
        for sample in batch_images:
            vs = [torch.from_numpy(np.asarray(img.convert("RGB").resize((256, 256)))) for img in sample]
            views.append(torch.stack(vs))
        frame0 = torch.stack(views).unsqueeze(2).to(device)         # [B,V,1,256,256,3]
        s_0 = self._dino_latents(frame0)[:, 0]
        s_end = self.mamba_predictor(s_0, action_tokens)[:, 0]
        if os.environ.get("ABLATE_PRED", "0") == "1":
            # bypass Mamba predictor: feed s_0 as the "future" latent
            s_end = s_0
        st = None
        if state is not None:
            st = torch.from_numpy(np.array(state)).to(device, dtype=torch.float32)
        cond = self._cond(s_0, s_end)
        with torch.autocast("cuda", dtype=torch.float32):
            actions = self.action_model.predict_action(cond, st)
        return {"normalized_actions": actions.float().cpu().numpy()}
