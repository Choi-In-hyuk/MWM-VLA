"""
VLA_DINO_StreamingMamba_FutureOnly_Denoise — invariance / "denoising" predictor.

Problem (as framed by the researcher):
    Front view vs side view, clean vs noisy — it is the SAME underlying STATE,
    but DINO represents them differently. We want the Mamba predictor to map any
    such corrupted/rotated observation to the CLEAN, canonical (front-view) future
    latent, so the action head always conditions on a consistent representation.

Approach (canonical-target denoising, no LIBERO-Plus data needed):
    During training, apply an IMAGE-LEVEL augmentation T (perspective/rotation/crop
    = fake camera angle, + brightness/contrast/blur/noise = visual corruption) to
    the predictor's INPUT frames (past + present), while keeping the L_pred TARGET
    as the CLEAN, un-augmented future latent:

        present_aug, past_aug = T(present), T(past)      # SAME T (one camera pose)
        s_target_pred = predictor([DINO(past_aug), DINO(present_aug)])
        L_pred        = L1(s_target_pred, DINO(target_CLEAN))

    Crucially the SAME augmented pixels feed BOTH DINO (predictor input) and the
    Qwen action tokens — at deploy time the whole stack sees the corrupted view,
    so training must too (no clean Qwen / noisy DINO mismatch).

Design decisions (locked with the researcher):
    - SAME T on past & present (a camera pose is consistent within a step).
    - target frame is NEVER augmented (canonical front-view GT to reconstruct).
    - per-sample Bernoulli(aug_prob): a fraction of samples stay fully clean so
      clean-LIBERO performance is preserved and train/test doesn't collapse.
    - "medium" strength: perspective ~<=0.25 distortion, rotate ~+-10deg,
      crop 0.9-1.0, brightness/contrast/blur/noise medium.

Implemented purely inside this framework (no dataloader changes) by augmenting
the `videos` uint8 tensor right after it is assembled, so both DINO and Qwen
inherit the augmented pixels automatically.

Config (framework.mamba_wm, all optional):
    aug_prob         : float = 0.7
    aug_persp        : float = 0.25   # RandomPerspective distortion_scale
    aug_rot_deg      : float = 10.0
    aug_crop_min     : float = 0.9    # min area fraction for random resized crop
    aug_brightness   : float = 0.3    # +- jitter fraction
    aug_contrast     : float = 0.3
    aug_blur_sigma   : float = 1.5    # max gaussian blur sigma
    aug_pixel_noise  : float = 0.05   # gaussian noise std in [0,1] pixel space
    aug_on_past      : bool  = True
"""
import random
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms.v2 import functional as TF
from torchvision.transforms.v2 import RandomPerspective

from starVLA.model.framework.VLA_DINO_StreamingMamba_FutureOnly import (
    VLA_DINO_StreamingMamba_FutureOnly,
)
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)


@FRAMEWORK_REGISTRY.register("VLA_DINO_StreamingMamba_FutureOnly_Denoise")
class VLA_DINO_StreamingMamba_FutureOnly_Denoise(VLA_DINO_StreamingMamba_FutureOnly):
    def __init__(self, config=None, **kwargs):
        super().__init__(config=config, **kwargs)

        mcfg = getattr(config.framework, "mamba_wm", None)
        get = (lambda k, d: getattr(mcfg, k, d)) if mcfg is not None else (lambda k, d: d)
        self.aug_prob        = float(get("aug_prob", 0.7))
        self.aug_persp       = float(get("aug_persp", 0.25))
        self.aug_rot_deg     = float(get("aug_rot_deg", 10.0))
        self.aug_crop_min    = float(get("aug_crop_min", 0.9))
        self.aug_brightness  = float(get("aug_brightness", 0.3))
        self.aug_contrast    = float(get("aug_contrast", 0.3))
        self.aug_blur_sigma  = float(get("aug_blur_sigma", 1.5))
        self.aug_pixel_noise = float(get("aug_pixel_noise", 0.05))
        self.aug_on_past     = bool(get("aug_on_past", True))

        logger.info(
            f"[FutureOnly_Denoise] canonical-target denoising: aug_prob={self.aug_prob} "
            f"persp={self.aug_persp} rot={self.aug_rot_deg} crop_min={self.aug_crop_min} "
            f"bright={self.aug_brightness} contrast={self.aug_contrast} "
            f"blur={self.aug_blur_sigma} pix_noise={self.aug_pixel_noise} "
            f"on_past={self.aug_on_past}. Target = CLEAN front-view future."
        )

    # ------------------------------------------------------------------ augmentation
    def _sample_aug_params(self, H, W):
        """Draw ONE augmentation T (shared across past/present and all views)."""
        # perspective start/end points (shared)
        startpoints, endpoints = RandomPerspective.get_params(W, H, self.aug_persp)
        p = {
            "startpoints": startpoints,
            "endpoints": endpoints,
            "angle": random.uniform(-self.aug_rot_deg, self.aug_rot_deg),
            # random resized crop box (area in [crop_min, 1.0], square-ish)
            "crop_scale": random.uniform(self.aug_crop_min, 1.0),
            "bright": 1.0 + random.uniform(-self.aug_brightness, self.aug_brightness),
            "contrast": 1.0 + random.uniform(-self.aug_contrast, self.aug_contrast),
            "blur": random.uniform(0.0, self.aug_blur_sigma),
            "noise": self.aug_pixel_noise,
        }
        return p

    def _apply_aug(self, img, p):
        """Apply a shared-parameter augmentation to a [C,H,W] float tensor in [0,1].

        Geometric (perspective, rotation, crop) is deterministic given p, so past
        and present get the exact same warp. Photometric/noise likewise."""
        C, H, W = img.shape
        # geometric
        img = TF.perspective(img, p["startpoints"], p["endpoints"])
        img = TF.rotate(img, p["angle"])
        # center random-resized crop (area = crop_scale), then resize back to HxW
        s = p["crop_scale"]
        ch, cw = int(round(H * (s ** 0.5))), int(round(W * (s ** 0.5)))
        top, left = (H - ch) // 2, (W - cw) // 2
        img = TF.resized_crop(img, top, left, ch, cw, [H, W], antialias=True)
        # photometric
        img = TF.adjust_brightness(img, p["bright"])
        img = TF.adjust_contrast(img, p["contrast"])
        if p["blur"] > 1e-3:
            k = 5
            img = TF.gaussian_blur(img, kernel_size=[k, k], sigma=[p["blur"], p["blur"]])
        if p["noise"] > 0:
            img = img + p["noise"] * torch.randn_like(img)
        return img.clamp_(0.0, 1.0)

    def _augment_videos(self, videos):
        """videos: uint8 tensor [B, V, 3, H, W, 3] (frame axis: 0=past,1=present,2=target).

        Returns a uint8 tensor of the same shape where, for a Bernoulli(aug_prob)
        subset of samples, frames past(+optionally 0) and present(1) are replaced by
        a SHARED augmentation T. Frame 2 (target) is NEVER touched.
        """
        B, V, Tn, H, W, _ = videos.shape
        assert Tn >= 3, f"expected >=3 frames (past/present/target), got {Tn}"
        out = videos.clone()
        frames_to_aug = [1] + ([0] if self.aug_on_past else [])  # present (+past)

        for b in range(B):
            if random.random() >= self.aug_prob:
                continue  # this sample stays clean
            p = self._sample_aug_params(H, W)
            for t in frames_to_aug:
                for v in range(V):
                    # [H,W,3] uint8 -> [3,H,W] float[0,1]
                    x = out[b, v, t].permute(2, 0, 1).float() / 255.0
                    x = self._apply_aug(x, p)
                    out[b, v, t] = (x * 255.0).round().clamp_(0, 255).byte().permute(1, 2, 0)
        return out

    # ------------------------------------------------------------------ TRAINING FORWARD
    def _forward_libero(self, examples: List[dict] = None, **kwargs):
        """FutureOnly forward, but predictor-input frames (past/present) get a
        shared image-level augmentation while the L_pred target stays CLEAN.
        The augmented pixels feed BOTH DINO and the Qwen action tokens."""
        from PIL import Image

        device = self.cond_proj.weight.device

        # === Assemble videos [B, V, 3, H, W, 3] uint8, then augment inputs in-place
        videos = torch.from_numpy(np.stack([e["video"] for e in examples])).to(device)
        if self.training and self.aug_prob > 0.0:
            videos = self._augment_videos(videos)

        # === Encode all 3 frames with DINO (frames 0,1 possibly augmented; 2 clean)
        with torch.autocast("cuda", dtype=torch.float32):
            dino_all = self._encode_dino_per_frame(videos).float()   # [B, 3, N, D]

        s_past    = dino_all[:, 0]
        s_present = dino_all[:, 1]
        s_target  = dino_all[:, 2]                                    # CLEAN target
        s_in      = torch.stack([s_past, s_present], dim=1)          # [B, 2, N, D]

        # === Robot state at present
        states_all = torch.from_numpy(
            np.stack([e["state_full"] for e in examples])
        ).to(device, dtype=torch.float32)                             # [B, 3, 8]
        r_present = states_all[:, 1]                                  # [B, 8]

        # === Qwen action tokens — from the (possibly augmented) PRESENT frame
        batch_images = []
        for b, e in enumerate(examples):
            v = videos[b].cpu().numpy()                               # [V, 3, H, W, 3] uint8
            pil_views = [
                Image.fromarray(v[v_i, 1]).resize((self.dino_size, self.dino_size))
                for v_i in range(v.shape[0])
            ]
            batch_images.append(pil_views)
        instructions = [e["lang"] for e in examples]
        action_tokens = self._qwen_action_tokens(batch_images, instructions)

        # === Predictor: augmented (corrupted-view) inputs -> CLEAN future
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
