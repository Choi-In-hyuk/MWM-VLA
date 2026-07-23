"""Vector Quantizer with EMA codebook update — for world-model latent
manifold approximation.

Motivation
----------
The world-model predictor outputs `s_target_pred` in a continuous DINOv2 latent
space. Under observation noise (LIBERO-Plus perturbations) the input latents
drift off the clean training distribution, and this drift propagates through
the WM into `s_target_pred`, contaminating action decoding.

This module snaps `s_target_pred` to the nearest entry in a learned codebook.
The codebook is trained (via EMA of assigned latents) to concentrate on the
clean training distribution, so quantization approximates a discrete manifold
projection: contaminated predictions are re-routed to nearby clean prototypes
before reaching the action head.

Key design choices
------------------
- **Token-wise quantization**: each of the N=512 spatial tokens is quantized
  independently. Preserves spatial expressiveness (vs one code per frame).
- **EMA update** (Oord et al. 2017 VQ-VAE-2 style): codebook is not updated
  by gradient — instead, running averages of assigned latents and their counts
  are maintained. This is more stable than pure gradient updates and is the
  standard for large codebooks.
- **Straight-through estimator**: gradients from downstream flow through
  the quantized output back to `s_target_pred` as if it were the identity.
- **Dead-code reset**: codes never assigned to any token get replaced by
  perturbed active codes, preventing codebook collapse.
- **Perplexity metric**: exposed for monitoring effective codebook usage.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizerEMA(nn.Module):
    def __init__(
        self,
        num_codes: int = 1024,
        dim: int = 768,
        commit_beta: float = 0.25,
        ema_decay: float = 0.99,
        eps: float = 1e-5,
        dead_code_threshold: float = 1.0,
        dead_code_reset_every: int = 1000,
    ):
        super().__init__()
        self.num_codes = num_codes
        self.dim = dim
        self.commit_beta = commit_beta
        self.ema_decay = ema_decay
        self.eps = eps
        self.dead_code_threshold = dead_code_threshold
        self.dead_code_reset_every = dead_code_reset_every

        # codebook: [K, D]. Not a Parameter — updated via EMA, not gradient.
        codebook = torch.randn(num_codes, dim) * (1.0 / num_codes) ** 0.5
        self.register_buffer("codebook", codebook)
        # EMA running averages: sum of assigned latents and counts per code
        self.register_buffer("ema_cluster_size", torch.zeros(num_codes))
        self.register_buffer("ema_weight_sum", codebook.clone())
        self.register_buffer("_step", torch.zeros((), dtype=torch.long))

    @torch.no_grad()
    def _reset_dead_codes(self, flat_input: torch.Tensor):
        """Reset codes whose EMA cluster size dropped below the threshold.
        Replace with random active latents (from the current batch) + small
        noise, so the collapsed codes rejoin the training distribution."""
        dead = self.ema_cluster_size < self.dead_code_threshold  # [K]
        n_dead = int(dead.sum().item())
        if n_dead == 0:
            return 0
        # sample n_dead random rows from flat_input (with replacement is fine)
        n_in = flat_input.shape[0]
        idx = torch.randint(0, n_in, (n_dead,), device=flat_input.device)
        new_codes = flat_input[idx] + 0.01 * torch.randn_like(flat_input[idx])
        self.codebook[dead] = new_codes
        # re-seed EMA so these codes have a fair chance of survival
        self.ema_cluster_size[dead] = 1.0
        self.ema_weight_sum[dead] = new_codes.clone()
        return n_dead

    def forward(self, s: torch.Tensor):
        """Quantize per token.

        Args:
            s: [B, N, D] continuous latents.
        Returns:
            s_q: [B, N, D] quantized latents (straight-through gradient).
            info: dict with 'vq_loss', 'perplexity', 'codes_used', 'n_dead_reset'.
        """
        assert s.dim() == 3 and s.shape[-1] == self.dim, \
            f"expected [B,N,{self.dim}], got {tuple(s.shape)}"
        B, N, D = s.shape
        flat = s.reshape(-1, D)  # [B*N, D]

        # squared L2 distance to every code
        # ||a-b||^2 = ||a||^2 - 2 a.b + ||b||^2
        d = (
            flat.pow(2).sum(dim=1, keepdim=True)
            - 2.0 * flat @ self.codebook.t()
            + self.codebook.pow(2).sum(dim=1)
        )  # [B*N, K]

        indices = d.argmin(dim=1)  # [B*N]
        one_hot = F.one_hot(indices, num_classes=self.num_codes).to(flat.dtype)  # [B*N, K]
        s_q = one_hot @ self.codebook  # [B*N, D]
        s_q = s_q.view(B, N, D)

        # EMA update (train only)
        n_dead_reset = 0
        if self.training:
            with torch.no_grad():
                # cluster sizes: how many latents assigned to each code this batch
                cluster_size = one_hot.sum(dim=0)  # [K]
                weight_sum = one_hot.t() @ flat    # [K, D]

                # DDP: aggregate across ranks so codebook stays consistent
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    torch.distributed.all_reduce(cluster_size, op=torch.distributed.ReduceOp.SUM)
                    torch.distributed.all_reduce(weight_sum, op=torch.distributed.ReduceOp.SUM)

                self.ema_cluster_size.mul_(self.ema_decay).add_(cluster_size, alpha=1 - self.ema_decay)
                self.ema_weight_sum.mul_(self.ema_decay).add_(weight_sum, alpha=1 - self.ema_decay)

                # Laplace smoothing to avoid divide-by-zero and stabilize
                n = self.ema_cluster_size.sum()
                cluster_size_smoothed = (
                    (self.ema_cluster_size + self.eps) / (n + self.num_codes * self.eps) * n
                )
                self.codebook.copy_(self.ema_weight_sum / cluster_size_smoothed.unsqueeze(1))

                # periodic dead-code reset
                self._step += 1
                if int(self._step.item()) % self.dead_code_reset_every == 0:
                    n_dead_reset = self._reset_dead_codes(flat.detach())

        # commitment loss: pushes encoder output toward the chosen code
        commit_loss = F.mse_loss(s, s_q.detach())
        vq_loss = self.commit_beta * commit_loss

        # straight-through: forward = s_q, backward = identity to s
        s_q_st = s + (s_q - s).detach()

        # perplexity: effective number of codes actually used
        with torch.no_grad():
            avg_probs = one_hot.mean(dim=0)  # [K]
            perplexity = torch.exp(-(avg_probs * torch.log(avg_probs + 1e-10)).sum())
            codes_used = int((avg_probs > 0).sum().item())

        info = {
            "vq_loss": vq_loss,
            "perplexity": perplexity.detach(),
            "codes_used": codes_used,
            "n_dead_reset": n_dead_reset,
        }
        return s_q_st, info
