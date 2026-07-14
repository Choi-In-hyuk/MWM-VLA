#!/usr/bin/env python
# Copyright 2026 VLA-JEPA research. MIT License.
"""
Stage-2 trainer for VLA_DINO_StreamingMamba with TRUNCATED-BPTT episode
streaming.

Difference vs scripts/train_mamba_wm.py
---------------------------------------
This script uses `EpisodeStreamingDataset` (slot-based, per-slot episode/chunk
cursor) instead of a random-sampling dataloader. For each step:

  1. The dataset yields `num_slots` samples (one per slot). Each sample has
     `__episode_start__ : bool`.
  2. The trainer maintains a per-slot SSM hidden-state cache.
  3. Slots that just started a new episode get their state zeroed; all others
     receive their previous-step state (detached).
  4. The framework's `forward_stage2(examples, state_in, mask_reset)` runs the
     chunk forward + action loss; returns `state_out`.
  5. After backward+step, `state_out` is detached and stored for the next step.

Stage 2 only — predictor (fine-tune) + diffusion action head + Qwen LoRA all
train.

DDP: each rank owns its own slots and trajectory shard.
"""
import argparse
import math
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from starVLA.model.tools import read_mode_config
from starVLA.model.framework.share_tools import dict_to_namespace
from starVLA.model.framework import build_framework
from starVLA.dataloader.lerobot_datasets import get_vla_dataset
from starVLA.dataloader.episode_streaming import EpisodeStreamingDataset


# ---------------------------------------------------------------- DDP helpers
def ddp_setup():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", init_method="env://")
        return True, rank, world, local_rank
    return False, 0, 1, 0


def is_main(rank: int) -> bool:
    return rank == 0


def rprint(rank: int, *a, **kw):
    if is_main(rank):
        print(*a, **kw, flush=True)


def build_cfg(args):
    model_config, _ = read_mode_config(Path(args.backbone_ckpt))
    cfg = dict_to_namespace(model_config)
    cfg.framework.name = "VLA_DINO_StreamingMamba"
    cfg.trainer.pretrained_checkpoint = None
    cfg.datasets.vla_data.data_root_dir = args.data_root
    cfg.datasets.vla_data.data_mix = args.data_mix
    cfg.datasets.vla_data.with_state = True
    from omegaconf import OmegaConf
    cfg.framework.mamba_wm = OmegaConf.create({
        "dino_backbone": args.dino_backbone,
        "stream_state_dim": args.stream_state_dim,
        "stream_depth": args.stream_depth,
        "stream_d_state": args.stream_d_state,
        "stream_d_conv": args.stream_d_conv,
        "stream_headdim": args.stream_headdim,
        "stream_chunk_size": args.stream_chunk_size,
        "seq_len_M": 1,   # Stage 2 is single-chunk per step; state carried across steps
        "robot_state_dim": 8,
    })
    return cfg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone_ckpt", required=True,
                   help="VLA-JEPA backbone checkpoint (load Qwen/DINO/head from here)")
    p.add_argument("--resume_ckpt", required=True,
                   help="Stage-1 streaming predictor checkpoint (mamba weights)")
    p.add_argument("--data_root", default="/mnt/4TB_2/jamvla/datasets/LIBERO")
    p.add_argument("--data_mix", default="libero_10")
    p.add_argument("--dino_backbone", default="dinov2_vitb14")
    p.add_argument("--output_dir", default="results/stream_libero10_stage2")
    p.add_argument("--qwen_lora", action="store_true", default=True)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    # streaming Mamba-2 hyperparameters (must match Stage 1)
    p.add_argument("--stream_state_dim", type=int, default=1024)
    p.add_argument("--stream_depth", type=int, default=12)
    p.add_argument("--stream_d_state", type=int, default=64)
    p.add_argument("--stream_d_conv", type=int, default=1)
    p.add_argument("--stream_headdim", type=int, default=64)
    p.add_argument("--stream_chunk_size", type=int, default=64)
    # training
    p.add_argument("--max_steps", type=int, default=50000)
    p.add_argument("--warmup_steps", type=int, default=5000)
    p.add_argument("--num_slots", type=int, default=16,
                   help="per-rank batch size; each slot streams one episode at a time")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--save_every", type=int, default=5000)
    p.add_argument("--state_save_every", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    is_ddp, rank, world, local_rank = ddp_setup()
    device = torch.device(f"cuda:{local_rank}") if is_ddp else torch.device("cuda:0")
    if not is_ddp:
        torch.cuda.set_device(device)

    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    random.seed(args.seed + rank)

    out = Path(args.output_dir)
    if is_main(rank):
        (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    if is_ddp:
        dist.barrier()

    cfg = build_cfg(args)

    # ---- model
    model = build_framework(cfg)
    model.load_backbone(args.backbone_ckpt)
    if args.qwen_lora:
        model.apply_qwen_lora(r=args.lora_r, alpha=args.lora_alpha)
        cfg.framework.mamba_wm.qwen_lora = True
        cfg.framework.mamba_wm.lora_r = args.lora_r
        cfg.framework.mamba_wm.lora_alpha = args.lora_alpha
    # load Stage-1 weights (predictor + cond_proj + Qwen LoRA)
    sd = torch.load(args.resume_ckpt, map_location="cpu")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    rprint(rank, f"[trainer] resumed Stage-1 weights ({len(missing)} missing, {len(unexpected)} unexpected)")
    model = model.to(device)
    model.set_stage("stage2")
    model.eval()
    for _, mod in model.named_children():
        if any(pm.requires_grad for pm in mod.parameters()):
            mod.train()

    trainable = [pm for pm in model.parameters() if pm.requires_grad]
    n_tr = sum(pm.numel() for pm in trainable)
    rprint(rank, f"[trainer] stage=stage2 trunc-BPTT trainable: {n_tr/1e6:.1f}M | ddp={is_ddp} world={world} rank={rank}")

    # ---- persist config / norm stats (rank 0)
    from omegaconf import OmegaConf
    if is_main(rank):
        OmegaConf.save(cfg, out / "config.yaml")
        src_dir = Path(args.backbone_ckpt).parent.parent
        if (src_dir / "dataset_statistics.json").exists():
            shutil.copy(src_dir / "dataset_statistics.json", out / "dataset_statistics.json")

    # ---- data: streaming dataset
    mixture = get_vla_dataset(
        data_cfg=cfg.datasets.vla_data, action_horizon=model.horizon,
        video_horizon=model.horizon + 1, obs_indices=[0, model.horizon],
    )
    stream = EpisodeStreamingDataset(
        mixture=mixture, num_slots=args.num_slots, horizon=model.horizon,
        rank=rank, world=world, seed=args.seed, max_steps=args.max_steps + 10,
    )
    rprint(rank, f"[trainer] streaming dataset: {len(stream._catalog)} trajectories on rank {rank}, "
                 f"num_slots={args.num_slots}, horizon={model.horizon}")

    # ---- DDP wrap
    if is_ddp:
        model_engine = DDP(model, device_ids=[local_rank], output_device=local_rank,
                           find_unused_parameters=True, broadcast_buffers=False)
        model_root = model_engine.module
    else:
        model_engine = model
        model_root = model

    # ---- optim
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))

    def lr_at(step):
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        prog = (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))

    def _lean_sd():
        sd = model_root.state_dict()
        if hasattr(model_root, "lean_save_key"):
            return {k: v for k, v in sd.items() if model_root.lean_save_key(k)}
        # default: mamba_predictor + cond_proj + action_model + Qwen LoRA
        keep = ("mamba_predictor.", "cond_proj.", "action_model.")
        out = {k: v for k, v in sd.items() if k.startswith(keep)}
        out.update({k: v for k, v in sd.items()
                    if k.startswith("qwen_vl_interface.") and "lora_" in k})
        return out

    def save(tag, full=False):
        if not is_main(rank):
            return
        path = out / "checkpoints" / f"mamba_wm_{tag}.pt"
        sd = model_root.state_dict() if full else _lean_sd()
        torch.save(sd, path)
        print(f"[trainer] saved {path} ({'full' if full else 'lean'})", flush=True)

    def save_state(tag, step):
        if not is_main(rank):
            return
        path = out / "checkpoints" / f"training_state_{tag}.pt"
        torch.save({
            "step": step,
            "model": _lean_sd(),
            "optimizer": opt.state_dict(),
            "args": vars(args),
        }, path)
        print(f"[trainer] saved training state -> {path} (step={step})", flush=True)

    # ---- training loop with per-slot state cache
    state_cache = None   # list of per-layer tensors, or None on the very first step
    step = 0
    t0 = time.time()
    running = {}

    stream_iter = iter(stream)
    while step < args.max_steps:
        batch = next(stream_iter)   # list of length num_slots, each dict has __episode_start__

        # Build mask_reset and strip the flag from samples
        mask_reset = [bool(b.pop("__episode_start__")) for b in batch]

        # On the very first step, every slot is starting → state_cache=None means zero
        out_dict = model_engine.module.forward_stage2(
            batch, state_in=state_cache, mask_reset=mask_reset,
        ) if is_ddp else model_engine.forward_stage2(
            batch, state_in=state_cache, mask_reset=mask_reset,
        )

        loss = out_dict["loss"]
        (loss / args.grad_accum).backward()

        # accumulate logging
        for k in ("pred_loss", "action_loss", "pred_cos"):
            running[k] = running.get(k, 0.0) + float(out_dict[k])

        if (step + 1) % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
            for g in opt.param_groups:
                g["lr"] = args.lr * lr_at(step)
            opt.step()
            opt.zero_grad(set_to_none=True)

        # Carry state to next step (detach to truncate BPTT)
        state_cache = [s.detach() for s in out_dict["state_out"]]

        step += 1
        if step % args.log_every == 0 and is_main(rank):
            n = args.log_every
            dt = time.time() - t0
            sps = n / dt
            pct = 100.0 * step / args.max_steps
            eta_h = (args.max_steps - step) / max(sps, 1e-6) / 3600
            msg = (f"pred={running.get('pred_loss', 0.0)/n:.4f} "
                   f"action={running.get('action_loss', 0.0)/n:.4f} "
                   f"pred_cos={running.get('pred_cos', 0.0)/n:.4f}")
            print(f"[stage2-tbptt] {step}/{args.max_steps} ({pct:4.1f}%) | {msg} "
                  f"| lr={args.lr*lr_at(step):.1e} | {sps:.2f} it/s | ETA {eta_h:.1f}h",
                  flush=True)
            running, t0 = {}, time.time()
        if step % args.save_every == 0:
            save(f"step{step}")
        if step % args.state_save_every == 0:
            save_state("latest", step)

    save("final", full=True)
    save_state("final", step)
    rprint(rank, "[trainer] done.")
    if is_ddp:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
