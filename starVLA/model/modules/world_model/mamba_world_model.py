# Copyright 2026 VLA-JEPA research. MIT License.
"""
Mamba-based latent world model for VLA-JEPA inference-time action generation.

Motivation
----------
In the original VLA-JEPA, the V-JEPA encoder + ViT predictor act only as a
training-time auxiliary loss; at inference actions come from the flow-matching
head and the world model is never run. This module makes the world model usable
*at inference* and fast enough for real-time control by replacing the heavy
V-JEPA video encoder / ViT predictor with light-weight Mamba (SSM) modules.

Design (single-observation, no tubelet)
---------------------------------------
At deployment only the *current* observation exists, so these modules never
consume video / tubelet stacks. V-JEPA's temporal axis is dropped; we distill
only its per-frame latent space and let Mamba own the time evolution:

    img_t                      --MambaStateEncoder-->  s_t        [B, 256, 2048]
    (s_t, qwen_action_tokens)  --MambaStatePredictor-> s_{t+1..H} [B, H, 256, 2048]
    (s_t, s_{t+1})             --InverseDynamicsHead-> a_t        [B, 7]

Targets during training (frozen V-JEPA as teacher):
    MambaStateEncoder   : s_t        ~= VJEPA_enc(frame_t)          (per-frame distill)
    MambaStatePredictor : s_{t+k}    ~= VJEPA_enc(frame_{t+k})      (latent forward model)
    InverseDynamicsHead : a_t        ~= ground-truth action          (inverse dynamics)

Measured interface (LIBERO ckpt): per-frame latent = [256 spatial tokens, 2048]
(V-JEPA vitl hidden 1024, 2 views concat = 2048); Qwen3-VL-2B hidden = 2048;
action chunk H = 7; action_dim = 7.
"""
from typing import Optional

import torch
import torch.nn as nn

from mamba_ssm import Mamba, Mamba2
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
from einops import rearrange


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


class MambaBlock(nn.Module):
    """Pre-norm Mamba block with residual. Optionally bidirectional.

    Bidirectional mode runs a second Mamba over the time-reversed sequence and
    sums the two passes -- appropriate for spatial (non-causal) token sets such
    as a single frame's patch tokens. Unidirectional (causal) mode is used for
    temporal roll-out where future must not leak into the past.
    """

    def __init__(self, dim: int, d_state: int = 16, d_conv: int = 4, expand: int = 2,
                 bidirectional: bool = False):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.bidirectional = bidirectional
        self.fwd = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.bwd = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand) \
            if bidirectional else None
        self.mlp = nn.Sequential(
            RMSNorm(dim), nn.Linear(dim, expand * dim), nn.GELU(), nn.Linear(expand * dim, dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        out = self.fwd(h)
        if self.bwd is not None:
            out = out + self.bwd(h.flip(1)).flip(1)
        x = x + out
        x = x + self.mlp(x)
        return x


class MambaStateEncoder(nn.Module):
    """Encode a single (multi-view) observation into a V-JEPA-style latent state.

    Replaces the frozen V-JEPA video encoder. Each view is patch-embedded
    (16x16) into `tokens_per_frame` tokens of `dim_per_view`, processed by
    bidirectional Mamba, then views are concatenated on the feature axis to
    match the V-JEPA multi-view latent [tokens_per_frame, num_views*dim_per_view].
    """

    def __init__(self, img_size: int = 256, patch_size: int = 16, num_views: int = 2,
                 dim_per_view: int = 1024, depth: int = 6, **mamba_kwargs):
        super().__init__()
        self.num_views = num_views
        self.dim_per_view = dim_per_view
        self.tokens_per_frame = (img_size // patch_size) ** 2
        self.state_dim = num_views * dim_per_view

        self.patch_embed = nn.Conv2d(3, dim_per_view, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.tokens_per_frame, dim_per_view))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.blocks = nn.ModuleList(
            [MambaBlock(dim_per_view, bidirectional=True, **mamba_kwargs) for _ in range(depth)]
        )
        self.norm = RMSNorm(dim_per_view)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """images: [B, V, 3, H, W] -> state: [B, tokens_per_frame, V*dim_per_view]."""
        B, V, C, H, W = images.shape
        assert V == self.num_views, f"expected {self.num_views} views, got {V}"
        x = images.reshape(B * V, C, H, W)
        x = self.patch_embed(x).flatten(2).transpose(1, 2)  # [B*V, tokens, dim]
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        x = x.reshape(B, V, self.tokens_per_frame, self.dim_per_view)
        x = x.permute(0, 2, 1, 3).reshape(B, self.tokens_per_frame, V * self.dim_per_view)
        return x  # [B, 256, 2048]


class MambaStatePredictor(nn.Module):
    """Action-conditioned latent forward model (replaces V-JEPA ViT predictor).

    Given the current latent state and Qwen action tokens, autoregressively
    rolls out H future latent states in a single causal Mamba pass. Construct a
    sequence [action_tokens | current_state | query_1 .. query_H] and read out
    the per-step query blocks; Mamba's causal structure ensures query_k only
    attends to action_tokens, the current state, and earlier queries.
    """

    def __init__(self, state_dim: int = 2048, action_token_dim: int = 2048,
                 tokens_per_frame: int = 256, horizon: int = 7, depth: int = 8,
                 **mamba_kwargs):
        super().__init__()
        self.tokens_per_frame = tokens_per_frame
        self.horizon = horizon
        self.state_dim = state_dim

        self.action_proj = nn.Linear(action_token_dim, state_dim)
        self.cur_pos = nn.Parameter(torch.zeros(1, tokens_per_frame, state_dim))
        # shared spatial query + per-step temporal embedding
        self.query = nn.Parameter(torch.zeros(1, tokens_per_frame, state_dim))
        self.step_embed = nn.Parameter(torch.zeros(1, horizon, 1, state_dim))
        for p in (self.cur_pos, self.query, self.step_embed):
            nn.init.trunc_normal_(p, std=0.02)

        self.blocks = nn.ModuleList(
            [MambaBlock(state_dim, bidirectional=False, **mamba_kwargs) for _ in range(depth)]
        )
        self.norm = RMSNorm(state_dim)
        self.out_proj = nn.Linear(state_dim, state_dim)

    def forward(self, state: torch.Tensor, action_tokens: torch.Tensor,
                vis_idx: Optional[torch.Tensor] = None) -> torch.Tensor:
        """state: [B, N, D], action_tokens: [B, Na, Da] -> [B, H, N, D].

        vis_idx: optional [B, Nv] long indices selecting which of the N current-state
        tokens are VISIBLE to the predictor (I-JEPA context masking). When given,
        `state` is the full [B, N, D] latent; only the Nv selected tokens (with their
        matching positional embeddings) enter the Mamba sequence, but all H*N future
        queries are still predicted. None -> full context (inference / stage2).
        """
        B, N, D = state.shape
        a = self.action_proj(action_tokens)                       # [B, Na, D]
        cur = state + self.cur_pos                                 # [B, N, D]
        if vis_idx is not None:                                    # keep only visible context tokens
            cur = torch.gather(cur, 1, vis_idx.unsqueeze(-1).expand(-1, -1, D))
        Nc = cur.shape[1]
        q = self.query + self.step_embed                          # [1, H, N, D]
        q = q.expand(B, -1, -1, -1).reshape(B, self.horizon * N, D)
        seq = torch.cat([a, cur, q], dim=1)                       # [B, Na+Nc+H*N, D]
        for blk in self.blocks:
            seq = blk(seq)
        seq = self.norm(seq)
        q_out = seq[:, a.shape[1] + Nc:, :]                       # [B, H*N, D]
        q_out = self.out_proj(q_out).reshape(B, self.horizon, N, D)
        return q_out


class MambaTemporalPredictor(nn.Module):
    """Endpoint predictor over a SEQUENCE of past latents (V3, temporal buffer).

    Unlike MambaStatePredictor (which sees a single current frame -> the Mamba is
    only a spatial mixer), this consumes a length-`context_len` buffer of past
    per-frame latents [s_{t-K+1}, ..., s_t] laid out as a TIME-then-space sequence,
    so Mamba's causal recurrence actually integrates motion across frames. It then
    predicts the single chunk-ENDPOINT latent s_{t+H} (one query block), keeping
    V2's large-change target (avoids V1's per-frame small-change problem).

    Sequence: [action_tokens | frame_0 (N) | ... | frame_{K-1} (N) | query (N)]
    Each past frame gets a per-frame temporal embedding; the query reads out s_end.
    """

    def __init__(self, state_dim: int = 768, action_token_dim: int = 2048,
                 tokens_per_frame: int = 512, context_len: int = 7, depth: int = 8,
                 **mamba_kwargs):
        super().__init__()
        self.tokens_per_frame = tokens_per_frame
        self.context_len = context_len
        self.state_dim = state_dim

        self.action_proj = nn.Linear(action_token_dim, state_dim)
        self.frame_pos = nn.Parameter(torch.zeros(1, context_len, 1, state_dim))  # per-frame time emb
        self.spatial_pos = nn.Parameter(torch.zeros(1, 1, tokens_per_frame, state_dim))
        self.query = nn.Parameter(torch.zeros(1, tokens_per_frame, state_dim))
        for p in (self.frame_pos, self.spatial_pos, self.query):
            nn.init.trunc_normal_(p, std=0.02)

        self.blocks = nn.ModuleList(
            [MambaBlock(state_dim, bidirectional=False, **mamba_kwargs) for _ in range(depth)]
        )
        self.norm = RMSNorm(state_dim)
        self.out_proj = nn.Linear(state_dim, state_dim)

    def forward(self, buffer: torch.Tensor, action_tokens: torch.Tensor) -> torch.Tensor:
        """buffer: [B, K, N, D] past latents (oldest..current), action_tokens: [B, Na, Da]
        -> s_end: [B, N, D] (the predicted chunk-endpoint latent)."""
        B, K, N, D = buffer.shape
        assert K == self.context_len, f"expected context_len={self.context_len}, got {K}"
        a = self.action_proj(action_tokens)                       # [B, Na, D]
        ctx = buffer + self.frame_pos + self.spatial_pos          # [B, K, N, D]
        ctx = ctx.reshape(B, K * N, D)                            # time-then-space
        q = self.query.expand(B, -1, -1)                          # [B, N, D]
        seq = torch.cat([a, ctx, q], dim=1)                      # [B, Na+K*N+N, D]
        for blk in self.blocks:
            seq = blk(seq)
        seq = self.norm(seq)
        s_end = self.out_proj(seq[:, a.shape[1] + K * N:, :])    # [B, N, D]
        return s_end


class InverseDynamicsHead(nn.Module):
    """Recover the action that drives one latent transition s_t -> s_{t+1}.

    Spatially pools each latent and feeds [pool(s_t), pool(s_{t+1}), their diff]
    plus an embedding of the current robot proprioceptive state to an MLP. The
    (single, current) robot state is broadcast across the whole action chunk —
    at inference only the current state is known; per-step change is carried by
    the latent transition.
    """

    def __init__(self, latent_dim: int = 2048, robot_state_dim: int = 8,
                 action_dim: int = 7, hidden_dim: int = 1024, state_embed_dim: int = 128):
        super().__init__()
        self.action_dim = action_dim
        self.robot_state_dim = robot_state_dim
        self.state_encoder = (
            nn.Sequential(nn.Linear(robot_state_dim, state_embed_dim), nn.GELU())
            if robot_state_dim else None
        )
        in_dim = 3 * latent_dim + (state_embed_dim if robot_state_dim else 0)
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def _encode_state(self, ref: torch.Tensor, state: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """ref: tensor whose dim0 is batch. state None -> zeros (eval without state)."""
        if self.state_encoder is None:
            return None
        if state is None:
            state = ref.new_zeros(ref.shape[0], self.robot_state_dim)
        return self.state_encoder(state)                          # [B, state_embed_dim]

    def _pair(self, s_t: torch.Tensor, s_tp1: torch.Tensor,
              state_emb: Optional[torch.Tensor]) -> torch.Tensor:
        p0, p1 = s_t.mean(dim=-2), s_tp1.mean(dim=-2)             # [..., D]
        feat = torch.cat([p0, p1, p1 - p0], dim=-1)
        if state_emb is not None:
            feat = torch.cat([feat, state_emb], dim=-1)
        return self.net(feat)

    def forward(self, states: torch.Tensor, state: Optional[torch.Tensor] = None) -> torch.Tensor:
        """states: [B, H+1, N, D], state: [B, robot_state_dim] -> [B, H, action_dim]."""
        s_t, s_tp1 = states[:, :-1], states[:, 1:]
        emb = self._encode_state(s_t, state)
        if emb is not None:
            emb = emb.unsqueeze(1).expand(-1, s_t.shape[1], -1)  # broadcast over chunk
        return self._pair(s_t, s_tp1, emb)

    def single(self, s_t: torch.Tensor, s_tp1: torch.Tensor,
               state: Optional[torch.Tensor] = None) -> torch.Tensor:
        """s_t, s_tp1: [B, N, D] -> [B, action_dim]."""
        return self._pair(s_t, s_tp1, self._encode_state(s_t, state))


class StreamingMambaBlock(nn.Module):
    """Causal pre-norm Mamba-2 block with explicit SSM-state carry.

    Built on `mamba_ssm.Mamba2`, but the forward pass calls
    `mamba_chunk_scan_combined` directly so that we can pass `initial_states`
    (the SSM state at the end of the previous chunk) and request
    `return_final_states`. Both flow through PyTorch autograd → gradient from
    chunk-N loss reaches chunk-N-1 parameters via the carried state.

    To keep the math exactly equivalent across "one long sequence" vs. "split
    into chunks with carried state" we set `d_conv=1` (no 1-D convolution).
    Mamba-2's SSM is strong enough on its own; conv state carry would
    otherwise be needed to be equivalent across the chunk boundary.

    API
    ---
    forward(x, initial_states=None) -> (y, final_state)
      x              : [B, L, D]
      initial_states : [B, nheads, headdim, d_state]  (or None for zero init)
      y              : [B, L, D]
      final_state    : [B, nheads, headdim, d_state]  (always returned)
    """

    def __init__(self, dim: int, d_state: int = 64, d_conv: int = 1,
                 expand: int = 2, headdim: int = 64, chunk_size: int = 64):
        super().__init__()
        self.norm = RMSNorm(dim)
        # d_conv=1 makes split-with-state mathematically equivalent to concat.
        self.mamba = Mamba2(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            headdim=headdim,
            chunk_size=chunk_size,
        )
        self.mlp = nn.Sequential(
            RMSNorm(dim), nn.Linear(dim, expand * dim), nn.GELU(), nn.Linear(expand * dim, dim)
        )
        self.d_inner = expand * dim
        self.d_state = d_state
        self.headdim = headdim
        self.nheads = self.d_inner // headdim
        self.chunk_size = chunk_size
        self.ngroups = self.mamba.ngroups
        self.d_ssm = self.mamba.d_ssm
        self.d_conv = d_conv

    def init_state(self, batch_size: int, device, dtype=None):
        """Zero initial SSM state, shape matching what mamba_chunk_scan_combined expects."""
        if dtype is None:
            dtype = self.norm.weight.dtype
        return torch.zeros(
            batch_size, self.nheads, self.headdim, self.d_state,
            device=device, dtype=dtype,
        )

    def _mamba_with_state(self, u: torch.Tensor, initial_states=None):
        """Run Mamba-2 over `u`, optionally starting from `initial_states`,
        and always returning the final SSM state. Re-implements
        `Mamba2.forward` so we can expose the state plumbing."""
        m = self.mamba
        # in_proj produces a single big tensor we split into z, xBC, dt
        zxbcdt = m.in_proj(u)
        A = -torch.exp(m.A_log.float())
        z, xBC, dt = torch.split(
            zxbcdt,
            [m.d_inner, m.d_inner + 2 * m.ngroups * m.d_state, m.nheads],
            dim=-1,
        )
        # d_conv == 1 path: just activation, no 1-D conv (no state to carry)
        if self.d_conv == 1:
            xBC_act = m.act(xBC)
        else:
            # Conv path: NB this breaks chunk-equivalence; we keep it only
            # in case someone wants to experiment with d_conv > 1 (then state
            # equivalence is not guaranteed across chunk boundary).
            xBC_act = m.act(
                m.conv1d(xBC.transpose(1, 2)).transpose(1, 2)[:, :-(self.d_conv - 1)]
            )
        x, B, C = torch.split(
            xBC_act,
            [m.d_ssm, m.ngroups * m.d_state, m.ngroups * m.d_state],
            dim=-1,
        )
        y, final_state = mamba_chunk_scan_combined(
            rearrange(x, "b l (h p) -> b l h p", p=m.headdim),
            dt, A,
            rearrange(B, "b l (g n) -> b l g n", g=m.ngroups),
            rearrange(C, "b l (g n) -> b l g n", g=m.ngroups),
            chunk_size=m.chunk_size,
            D=rearrange(m.D, "(h p) -> h p", p=m.headdim) if m.D_has_hdim else m.D,
            z=rearrange(z, "b l (h p) -> b l h p", p=m.headdim) if not m.rmsnorm else None,
            dt_bias=m.dt_bias,
            dt_softplus=True,
            initial_states=initial_states,
            return_final_states=True,
        )
        y = rearrange(y, "b l h p -> b l (h p)")
        if m.rmsnorm:
            y = m.norm(y, z)
        out = m.out_proj(y)
        return out, final_state

    def forward(self, x: torch.Tensor, initial_states=None):
        """Pre-norm Mamba-2 residual block with state carry.

        x              : [B, L, D]
        initial_states : per-block carried SSM state or None
        returns
            y          : [B, L, D]
            final_state: [B, nheads, headdim, d_state]
        """
        h = self.norm(x)
        out, final_state = self._mamba_with_state(h, initial_states=initial_states)
        x = x + out
        x = x + self.mlp(x)
        return x, final_state


class StreamingMambaPredictor(nn.Module):
    """Streaming Mamba world-model predictor.

    Conceptually: the model predicts the chunk-end latent s_end every H frames,
    but its hidden state observes ALL intermediate frames as they arrive. This
    gives the same chunk-rate prediction interface as the baseline predictor
    while carrying the in-chunk dynamics in the SSM state.

    Training path (`forward`)
    -------------------------
    The dataloader builds a *single concatenated sequence* of M consecutive
    chunks interleaved with mid-chunk stream frames, with one query block per
    chunk. A single parallel-scan Mamba forward processes the whole thing.

    Sequence layout for M=2 (per sample), where x means "DINO patch tokens
    of one frame", r means "robot_state token", a means "action_tokens":

        [ a_0 | r_0 | x_0 | Q_0           # chunk 0: predict s at t=+7
        | r_2 | x_2                        # mid-chunk stream (frame +2)
        | r_4 | x_4                        # mid-chunk stream (frame +4)
        | r_6 | x_6                        # mid-chunk stream (frame +6)
        | a_1 | r_7 | x_7 | Q_1 ]          # chunk 1: predict s at t=+14

    The model reads out per-chunk s_end predictions at the Q_i slices. Mamba's
    causal SSM ensures Q_1 sees everything that came before, so the mid-chunk
    frames really do influence the next chunk's prediction (and gradients flow
    back to teach the model to use them).

    Inference path (`forward_for_inference` + `step_stream`)
    --------------------------------------------------------
    Mathematically identical, but split because frames arrive one at a time.
    `forward_for_inference` runs the chunk-start tokens; `step_stream` runs
    the per-frame stream tokens one at a time using the recurrent `step` API.
    SSM state is carried across calls within an episode.

    Input dims
    ----------
        state_dim         : Mamba internal width (kept independent of DINO)
        action_token_dim  : Qwen hidden = 2048
        dino_dim          : 768 (vitb14)
        tokens_per_frame  : N = 512 (2 views × 256 patches)
        robot_state_dim   : 8 (joints 7 + gripper 1)
    """

    def __init__(self, state_dim: int = 1024, action_token_dim: int = 2048,
                 dino_dim: int = 768, tokens_per_frame: int = 512,
                 robot_state_dim: int = 0, depth: int = 12,
                 d_state: int = 64, d_conv: int = 1, expand: int = 2,
                 headdim: int = 64, chunk_size: int = 64):
        super().__init__()
        self.state_dim = state_dim
        self.dino_dim = dino_dim
        self.tokens_per_frame = tokens_per_frame
        self.depth = depth
        # robot_state is fed only to the action head, not to the predictor.
        # We keep the kwarg for API compat but ignore any non-zero value.
        self.robot_state_dim = 0

        # ---- projections into Mamba space
        self.action_proj = nn.Linear(action_token_dim, state_dim)
        self.dino_proj = nn.Linear(dino_dim, state_dim)

        # ---- role embeddings
        # Tells the SSM "this token belongs to ...". One vector per role,
        # broadcast over all matching tokens.
        self.role_emb = nn.ParameterDict({
            "action": nn.Parameter(torch.zeros(1, 1, state_dim)),
            "obs":    nn.Parameter(torch.zeros(1, 1, state_dim)),
            "query":  nn.Parameter(torch.zeros(1, 1, state_dim)),
        })
        for p in self.role_emb.values():
            nn.init.trunc_normal_(p, std=0.02)

        # Learnable query placeholder (N tokens; SSM reads out s_end here).
        self.query_tok = nn.Parameter(torch.zeros(1, tokens_per_frame, state_dim))
        nn.init.trunc_normal_(self.query_tok, std=0.02)

        # Positional embed for the patch axis (shared across frames). Helps
        # the SSM distinguish patches even though all observation tokens
        # share the same role embedding.
        self.patch_pos = nn.Parameter(torch.zeros(1, tokens_per_frame, state_dim))
        nn.init.trunc_normal_(self.patch_pos, std=0.02)

        # Per-input-frame time embedding. Two input frames per chunk: t-7 (idx 0)
        # and t (idx 1). Added to obs tokens so the SSM knows which frame each
        # patch came from. Shape [2, 1, D] broadcasts over patches.
        self.time_emb = nn.Parameter(torch.zeros(2, 1, state_dim))
        nn.init.trunc_normal_(self.time_emb, std=0.02)

        # ---- streaming Mamba-2 stack (d_conv=1 keeps split-with-state equivalent to concat)
        self.blocks = nn.ModuleList([
            StreamingMambaBlock(state_dim, d_state=d_state, d_conv=d_conv, expand=expand,
                                headdim=headdim, chunk_size=chunk_size)
            for _ in range(depth)
        ])
        self.norm_out = RMSNorm(state_dim)
        self.out_proj = nn.Linear(state_dim, state_dim)
        # back to DINO dim so downstream cond_proj / head sees s_end matching s_0
        self.to_dino = nn.Linear(state_dim, dino_dim)

    # ------------------------------------------------------------------ helpers
    def _token_action(self, action_tokens: torch.Tensor) -> torch.Tensor:
        """[B, Na, A_dim] -> [B, Na, D] with action role embedding."""
        return self.action_proj(action_tokens) + self.role_emb["action"]

    def _token_obs(self, s: torch.Tensor) -> torch.Tensor:
        """Two-frame obs.
        Input  : [B, 2, N, dino_dim]  (t-7 at idx 0, t at idx 1)
                 — accepts [B, N, dino_dim] as a single frame for back-compat.
        Output : [B, 2*N, D] with obs role + patch positional + time embed.
        """
        if s.dim() == 3:
            # legacy single-frame path (kept for tests / one-frame call sites)
            return self.dino_proj(s) + self.role_emb["obs"] + self.patch_pos
        # two-frame: project per frame, add patch_pos (shared) + time_emb (per frame)
        B, T, N, _ = s.shape
        x = self.dino_proj(s)                                  # [B, T, N, D]
        x = x + self.role_emb["obs"]                            # broadcast
        x = x + self.patch_pos.unsqueeze(0)                     # [1,1,N,D] broadcasts over T
        x = x + self.time_emb[:T].unsqueeze(0)                  # [1,T,1,D] broadcasts over N
        return x.reshape(B, T * N, x.shape[-1])

    def _token_query(self, B: int) -> torch.Tensor:
        """[B, N, D] learnable query tokens with query role embedding."""
        return (self.query_tok + self.role_emb["query"]).expand(B, -1, -1)

    # ------------------------------------------------------------------ chunk forward (train + inference)
    def init_states(self, batch_size: int, device, dtype=None):
        """Per-layer zero SSM state (used as initial_states for the FIRST chunk
        of a sample / episode)."""
        return [blk.init_state(batch_size, device, dtype) for blk in self.blocks]

    # nn.Module convention: forward is the entry point
    def forward(self, action_tokens, s_t, states=None):
        return self.forward_chunk(action_tokens, s_t, states)

    def forward_chunk(self, action_tokens: torch.Tensor,
                      s_t: torch.Tensor,
                      states=None):
        """Run ONE chunk forward.

        Sequence per chunk:
            [ a | s | query ]
              Na   T*N   N
        where T is the number of input frames (1 or 2). robot_state is NOT
        consumed here — it goes only to the downstream action head.

        Inputs:
          action_tokens : [B, Na, A_dim]
          s_t           : [B, N, dino_dim]  (single frame) or [B, 2, N, dino_dim]
          states        : list of per-layer SSM states, or None for zero init

        Returns:
          s_end_pred    : [B, N, dino_dim]
          new_states    : list of per-layer SSM states
        """
        B = action_tokens.shape[0]
        N = self.tokens_per_frame

        a_tok = self._token_action(action_tokens)             # [B, Na, D]
        o_tok = self._token_obs(s_t)                          # [B, T*N, D]
        q_tok = self._token_query(B)                          # [B, N, D]
        seq = torch.cat([a_tok, o_tok, q_tok], dim=1)         # [B, L, D]

        if states is None:
            states = [None] * len(self.blocks)

        h = seq
        new_states = []
        for blk, st in zip(self.blocks, states):
            h, fs = blk(h, initial_states=st)
            new_states.append(fs)
        h = self.norm_out(h)

        # Read out s_end at the query slice (last N tokens)
        q_out = self.out_proj(h[:, -N:, :])
        s_end_pred = self.to_dino(q_out)
        return s_end_pred, new_states


if __name__ == "__main__":
    dev = "cuda"
    B, V, N, D, H = 2, 2, 256, 2048, 7
    enc = MambaStateEncoder().to(dev)
    pred = MambaStatePredictor().to(dev)
    idm = InverseDynamicsHead().to(dev)

    imgs = torch.randn(B, V, 3, 256, 256, device=dev)
    action_tokens = torch.randn(B, 24, 2048, device=dev)
    robot_state = torch.randn(B, 8, device=dev)

    s_t = enc(imgs)
    print("encoder out:", tuple(s_t.shape))                       # [2, 256, 2048]
    future = pred(s_t, action_tokens)
    print("predictor out:", tuple(future.shape))                 # [2, 7, 256, 2048]
    traj = torch.cat([s_t.unsqueeze(1), future], dim=1)          # [2, 8, 256, 2048]
    actions = idm(traj, robot_state)
    print("inverse-dynamics out:", tuple(actions.shape))         # [2, 7, 7]
    print("idm without state (zeros):", tuple(idm(traj).shape))

    n = sum(p.numel() for p in list(enc.parameters()) + list(pred.parameters()) + list(idm.parameters()))
    print(f"total new params: {n/1e6:.1f}M")
