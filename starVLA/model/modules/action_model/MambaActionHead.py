# Copyright 2026 VLA-JEPA research. MIT License.
"""
MambaActionHead — Mamba-backbone flow-matching action decoder.

Designed as a drop-in replacement for FlowmatchingActionHead inside the
VLA_DINO_DualMamba framework. The world model (Mamba #1) already turns Qwen's
semantic intent into a future-latent prediction s_end; this head focuses
purely on the MOVEMENT: how to drive the robot from s_0 to s_end.

Sequence (no language tokens — intent is already absorbed by the world model):

    [ sigma(1)
    | LN(s_end - s_0) -> Linear(D->d) tokens (N tokens, no pooling)
    | robot_state(1)
    | noised_action(action_horizon) ]

A stack of `n_layer` Mamba blocks consumes this sequence; the last
`action_horizon` tokens decode the per-step velocity. Flow-matching loss is
MSE on velocity = (actions - noise), matching the original GR00T head's
training/inference contract so the framework code around it is unchanged.

Why s_diff (no pooling)
-----------------------
Most of the 256 DINO spatial tokens are static between s_0 and s_end (the
scene barely moves over a 7-frame chunk). The DIFFERENCE s_end - s_0 is
nearly zero on those tokens and only nonzero where action-relevant change
happens, so Mamba's learning signal naturally concentrates on the moving
regions. This is the soft analogue of explicit top-K change-token selection
but without any hard hyperparameter.

LayerNorm before the linear projection re-scales s_diff (which can be very
small in raw DINO units) so it is comparable to the other conditioning
tokens.

Reuses ``starVLA.model.modules.world_model.mamba_world_model.MambaBlock``
(pre-norm Mamba + MLP residual block, same primitive used for Mamba #1).
"""
import math
from typing import Optional

import torch
import torch.nn as nn

from starVLA.model.modules.world_model.mamba_world_model import MambaBlock, RMSNorm


class SinusoidalTimeEmbedding(nn.Module):
    """Standard sinusoidal embedding of a [0, 1] continuous timestep + 2-layer MLP."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: [B] in [0, 1] -> [B, 1, dim]
        if t.dim() == 0:
            t = t.unsqueeze(0)
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=t.device, dtype=torch.float32)
            / max(1, half - 1)
        )
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)  # [B, half]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # [B, dim or dim-1]
        if emb.shape[-1] < self.dim:
            emb = torch.nn.functional.pad(emb, (0, self.dim - emb.shape[-1]))
        return self.mlp(emb).unsqueeze(1)  # [B, 1, dim]


class MambaActionHead(nn.Module):
    """Flow-matching action decoder with a Mamba backbone.

    Forward / predict_action signatures mirror FlowmatchingActionHead so the
    framework can swap heads without touching its loss / inference paths.

    Conditioning convention
    -----------------------
    Instead of FlowmatchingActionHead's (vl_embs, state, actions) where
    vl_embs is the cross-attention key/value of [s_0, s_end] projected to
    Qwen dim, this head consumes (s_0, s_end, state, actions) directly and
    forms its own self-attention sequence. The framework passes raw DINO
    latents (no cond_proj) so the head owns the s_diff computation.
    """

    def __init__(
        self,
        latent_dim: int,              # DINO per-token dim (e.g. 768 for vitb14)
        action_dim: int,              # 7 for LIBERO
        action_horizon: int,          # H+1 = 8
        robot_state_dim: int = 8,
        embed_dim: int = 256,
        n_layer: int = 5,
        num_inference_timesteps: int = 10,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.embed_dim = embed_dim
        self.num_inference_timesteps = num_inference_timesteps

        # diffusion-timestep token (sigma in [0, 1])
        self.sigma_emb = SinusoidalTimeEmbedding(embed_dim)

        # LN + linear projection of s_diff = (s_end - s_0)  [B, N, latent_dim] -> [B, N, embed_dim]
        self.diff_norm = nn.LayerNorm(latent_dim)
        self.diff_proj = nn.Linear(latent_dim, embed_dim)

        # robot proprioception token
        self.state_emb = nn.Sequential(
            nn.Linear(robot_state_dim, embed_dim), nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.zero_state = nn.Parameter(torch.zeros(1, 1, embed_dim))  # used when state=None

        # noised action embedding (per-step)
        self.action_in = nn.Linear(action_dim, embed_dim)
        self.action_pos = nn.Parameter(torch.zeros(1, action_horizon, embed_dim))
        nn.init.trunc_normal_(self.action_pos, std=0.02)

        # Mamba backbone (causal; sigma & conditioning go first, actions last)
        self.blocks = nn.ModuleList([
            MambaBlock(embed_dim, d_state=d_state, d_conv=d_conv, expand=expand,
                       bidirectional=False)
            for _ in range(n_layer)
        ])
        self.norm_out = RMSNorm(embed_dim)

        # velocity decoder on the trailing action tokens
        self.action_out = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.GELU(),
            nn.Linear(embed_dim, action_dim),
        )

        # placeholder so external code that reads `head.device` keeps working
        self.register_buffer("_device_probe", torch.zeros(1), persistent=False)

    @property
    def device(self):
        return self._device_probe.device

    # ------------------------------------------------------------------ helpers
    def _make_state_token(self, state: Optional[torch.Tensor], batch_size: int) -> torch.Tensor:
        if state is None:
            return self.zero_state.expand(batch_size, -1, -1)
        return self.state_emb(state).unsqueeze(1)  # [B, 1, D]

    def _build_seq(self, s_0: torch.Tensor, s_end: torch.Tensor,
                   state: Optional[torch.Tensor], noised_actions: torch.Tensor,
                   sigma: torch.Tensor) -> torch.Tensor:
        """Assemble [sigma | s_diff | state | action] sequence."""
        B = s_0.shape[0]
        s_diff = self.diff_proj(self.diff_norm(s_end - s_0))                # [B, N, D]
        sigma_tok = self.sigma_emb(sigma).to(s_diff.dtype)                  # [B, 1, D]
        state_tok = self._make_state_token(state, B).to(s_diff.dtype)       # [B, 1, D]
        act_tok = self.action_in(noised_actions) + self.action_pos          # [B, A, D]
        act_tok = act_tok.to(s_diff.dtype)
        return torch.cat([sigma_tok, s_diff, state_tok, act_tok], dim=1)

    def _decode(self, seq: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            seq = blk(seq)
        seq = self.norm_out(seq)
        return self.action_out(seq[:, -self.action_horizon:])               # [B, A, action_dim]

    # ------------------------------------------------------------------ training
    def forward(self, s_0: torch.Tensor, s_end: torch.Tensor, actions: torch.Tensor,
                state: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Flow-matching velocity loss.

        s_0, s_end : [B, N, latent_dim]   DINO latents (current and predicted endpoint)
        actions    : [B, action_horizon, action_dim]   ground-truth actions
        state      : [B, robot_state_dim] or None
        returns    : scalar loss
        """
        device = s_0.device
        noise = torch.randn_like(actions)
        t = torch.rand(actions.shape[0], device=device, dtype=actions.dtype)
        t_b = t[:, None, None]
        noised = (1.0 - t_b) * noise + t_b * actions                        # interp toward GT
        velocity_target = actions - noise                                   # flow-matching target

        seq = self._build_seq(s_0, s_end, state, noised, t)
        velocity_pred = self._decode(seq)
        return ((velocity_pred - velocity_target) ** 2).mean()

    # ------------------------------------------------------------------ inference
    @torch.no_grad()
    def predict_action(self, s_0: torch.Tensor, s_end: torch.Tensor,
                       state: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Iterative Euler denoising from pure noise to a clean action chunk."""
        B = s_0.shape[0]
        device = s_0.device
        actions = torch.randn(B, self.action_horizon, self.action_dim, device=device)
        steps = self.num_inference_timesteps
        dt = 1.0 / steps
        for i in range(steps):
            t = torch.full((B,), i * dt, device=device)
            seq = self._build_seq(s_0, s_end, state, actions, t)
            velocity = self._decode(seq)
            actions = actions + dt * velocity
        return actions
