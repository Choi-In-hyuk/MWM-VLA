#!/usr/bin/env python
"""Streaming Mamba predictor pretraining on SSv2 + Droid.

- No state, no action supervision (video + language only).
- Loss: L_pred (L1 + cosine) on the +H DINO latent.
- Trainable: Mamba predictor (from scratch) + Qwen LoRA (warm-started from
  the LIBERO-tuned backbone).
- DDP-ready. Reuses set_stage("predictor") to freeze everything except
  predictor + LoRA.

Usage (DDP x2):
  torchrun --standalone --nproc_per_node=2 scripts/pretrain_ssv2_droid.py \
    --backbone_ckpt /mnt/4TB_2/jamvla/models/VLA-JEPA-LIBERO/LIBERO/checkpoints/VLA-JEPA-LIBERO.pt \
    --ssv2_root /mnt/4TB_2/jamvla/datasets/ssv2 \
    --droid_root /mnt/4TB_2/jamvla/datasets/DroidLerobot \
    --output_dir results/pretrain_ssv2_droid \
    --max_steps 50000 --warmup_steps 5000 --batch_size 16 --num_workers 8
"""
from __future__ import annotations

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
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from starVLA.dataloader.pretrain_mixer import SSv2DroidMixed, pretrain_collate
from starVLA.model.framework import build_framework
from starVLA.model.tools import read_mode_config


# --------------------------------------------------------------------- DDP setup
def ddp_setup():
    """Initialize torch.distributed if launched via torchrun; else single-GPU.

    Some third-party packages (timm/transformers/peft/accelerate) auto-initialize
    the default process group at import time when RANK/WORLD_SIZE are present.
    Guard with `dist.is_initialized()` so we don't double-init.
    """
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return False, 0, 1, 0
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
    return True, rank, world, local_rank


def is_main(rank: int) -> bool:
    return rank == 0


def rprint(rank: int, *a, **kw):
    if is_main(rank):
        print(*a, **kw, flush=True)


def build_cfg(args) -> "OmegaConf":
    model_config, _ = read_mode_config(Path(args.backbone_ckpt))
    cfg = OmegaConf.create(model_config)
    cfg.framework.name = args.framework
    cfg.framework.mamba_wm = OmegaConf.merge(getattr(cfg.framework, "mamba_wm", {}), {
        "stream_state_dim": args.stream_state_dim,
        "stream_depth": args.stream_depth,
        "stream_d_state": args.stream_d_state,
        "stream_d_conv": args.stream_d_conv,
        "stream_headdim": args.stream_headdim,
        "stream_chunk_size": args.stream_chunk_size,
        "dino_backbone": args.dino_backbone,
        # predictor stage: robot_state_dim harmless (action head unused during pretrain)
        "robot_state_dim": 8,
    })
    return cfg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone_ckpt", required=True)
    p.add_argument("--framework", default="VLA_DINO_StreamingMamba")
    p.add_argument("--dino_backbone", default="dinov2_vitb14")
    p.add_argument("--qwen_lora", action="store_true", default=True)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)

    p.add_argument("--ssv2_root", default="/mnt/4TB_2/jamvla/datasets/ssv2")
    p.add_argument("--droid_root", default="/mnt/4TB_2/jamvla/datasets/DroidLerobot")
    p.add_argument("--obs_horizon", type=int, default=7)
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--ssv2_weight", type=float, default=1.0)
    p.add_argument("--droid_weight", type=float, default=1.0)

    p.add_argument("--output_dir", default="results/pretrain_ssv2_droid")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lora_lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--max_steps", type=int, default=50000)
    p.add_argument("--warmup_steps", type=int, default=5000)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--save_every", type=int, default=5000)
    p.add_argument("--state_save_every", type=int, default=5000)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--stream_state_dim", type=int, default=1024)
    p.add_argument("--stream_depth", type=int, default=12)
    p.add_argument("--stream_d_state", type=int, default=64)
    p.add_argument("--stream_d_conv", type=int, default=1)
    p.add_argument("--stream_headdim", type=int, default=64)
    p.add_argument("--stream_chunk_size", type=int, default=64)

    p.add_argument("--resume_state", default=None)
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
    model = model.to(device)
    model.set_stage("predictor")  # freezes everything except mamba_predictor + LoRA
    model.eval()
    for _, mod in model.named_children():
        if any(pm.requires_grad for pm in mod.parameters()):
            mod.train()

    # Separate LR groups: LoRA vs the rest (predictor).
    lora_params = [pm for n, pm in model.named_parameters()
                   if pm.requires_grad and "lora_" in n]
    rest_params = [pm for n, pm in model.named_parameters()
                   if pm.requires_grad and "lora_" not in n]
    n_lora = sum(pm.numel() for pm in lora_params)
    n_rest = sum(pm.numel() for pm in rest_params)
    rprint(rank, f"[trainer] trainable: predictor+cond {n_rest/1e6:.1f}M  "
                 f"lora {n_lora/1e6:.1f}M | ddp={is_ddp} world={world}")

    if is_main(rank):
        OmegaConf.save(cfg, out / "config.yaml")
        src_dir = Path(args.backbone_ckpt).parent.parent
        if (src_dir / "dataset_statistics.json").exists():
            shutil.copy(src_dir / "dataset_statistics.json", out / "dataset_statistics.json")

    # ---- data
    dataset = SSv2DroidMixed(
        ssv2_root=args.ssv2_root, droid_root=args.droid_root,
        obs_horizon=args.obs_horizon, image_size=args.image_size,
        ssv2_weight=args.ssv2_weight, droid_weight=args.droid_weight,
        seed=args.seed + rank,
    )
    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank,
                                 shuffle=True, drop_last=True) if is_ddp else None
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=(sampler is None),
        sampler=sampler, num_workers=args.num_workers, collate_fn=pretrain_collate,
        drop_last=True, pin_memory=False, persistent_workers=(args.num_workers > 0),
    )
    rprint(rank, f"[trainer] SSv2+Droid: nominal_len={len(dataset)}  "
                 f"steps/epoch={len(loader)} (per rank)")

    # ---- DDP wrap
    if is_ddp:
        model_engine = DDP(model, device_ids=[local_rank], output_device=local_rank,
                           find_unused_parameters=True, broadcast_buffers=False)
        model_root = model_engine.module
    else:
        model_engine = model
        model_root = model

    # ---- optim: separate LR for LoRA vs predictor
    param_groups = [
        {"params": rest_params, "lr": args.lr, "name": "predictor"},
        {"params": lora_params, "lr": args.lora_lr, "name": "lora"},
    ]
    opt = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    base_lrs = [args.lr, args.lora_lr]

    def lr_at(step):
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        prog = (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))

    # ---- lean save (predictor + LoRA)
    def _lean_sd():
        sd = model_root.state_dict()
        if hasattr(model_root, "lean_save_key"):
            return {k: v for k, v in sd.items() if model_root.lean_save_key(k)}
        return sd

    def save(tag, full=False):
        if not is_main(rank):
            return
        path = out / "checkpoints" / f"mamba_wm_{tag}.pt"
        sd = model_root.state_dict() if full else _lean_sd()
        torch.save(sd, path)
        print(f"[trainer] saved {path} ({'full' if full else 'lean'})", flush=True)

    def save_state(tag, step, epoch=0):
        if not is_main(rank):
            return
        path = out / "checkpoints" / f"training_state_{tag}.pt"
        tmp = path.with_suffix(".pt.tmp")
        torch.save({
            "step": step, "epoch": epoch,
            "model": _lean_sd(), "optimizer": opt.state_dict(),
            "args": vars(args),
            "rng": {"torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state_all(),
                    "numpy": np.random.get_state(),
                    "python": random.getstate()},
        }, tmp)
        # Atomic rename so a crash mid-save can't leave a corrupt latest.
        os.replace(tmp, path)
        print(f"[trainer] saved training state -> {path} (step={step}, epoch={epoch})", flush=True)

    # ---- resume (auto-detect training_state_latest.pt if --resume_state not given)
    start_step = 0
    resume_path = args.resume_state
    if resume_path is None:
        auto = out / "checkpoints" / "training_state_latest.pt"
        if auto.exists():
            resume_path = str(auto)
            rprint(rank, f"[trainer] AUTO-RESUME: found {auto}")
    if resume_path:
        ckpt = torch.load(resume_path, map_location="cpu")
        model_root.load_state_dict(ckpt["model"], strict=False)
        opt.load_state_dict(ckpt["optimizer"])
        start_step = int(ckpt["step"])
        # Restore RNG state so data ordering / dropout / init noise resume exactly.
        rng = ckpt.get("rng", {})
        if "torch" in rng:
            torch.set_rng_state(rng["torch"])
        if "cuda" in rng and torch.cuda.is_available():
            try:
                torch.cuda.set_rng_state_all(rng["cuda"])
            except Exception as e:
                rprint(rank, f"[trainer] WARN: cuda RNG restore failed: {e}")
        if "numpy" in rng:
            np.random.set_state(rng["numpy"])
        if "python" in rng:
            random.setstate(rng["python"])
        rprint(rank, f"[trainer] RESUMED from {resume_path} at step={start_step} "
                     f"(rng restored: {list(rng.keys())})")

    # ---- train loop
    steps_per_epoch = max(1, len(loader))
    running = {}
    step, t0 = start_step, time.time()
    # Prefer saved epoch (accurate under partial-epoch checkpoints); else derive.
    epoch = int(ckpt.get("epoch", step // steps_per_epoch)) if resume_path else 0
    opt.zero_grad(set_to_none=True)

    while step < args.max_steps:
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in loader:
            # DDP routes __call__ -> forward(); framework.forward dispatches to
            # forward_pretrain when it sees the pretrain-style batch dict.
            losses = model_engine(batch)
            (losses["loss"] / args.grad_accum).backward()
            for k, v in losses.items():
                running[k] = running.get(k, 0.0) + float(v.detach() if hasattr(v, "detach") else v)

            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [pm for g in param_groups for pm in g["params"]], args.max_grad_norm)
                scale = lr_at(step)
                for g, base_lr in zip(opt.param_groups, base_lrs):
                    g["lr"] = base_lr * scale
                opt.step()
                opt.zero_grad(set_to_none=True)

            step += 1
            if step % args.log_every == 0 and is_main(rank):
                n = args.log_every
                dt = time.time() - t0
                sps = n / dt
                pct = 100.0 * step / args.max_steps
                eta_h = (args.max_steps - step) / max(sps, 1e-6) / 3600
                msg = f"pred={running.get('pred_loss',0)/n:.4f} " \
                      f"cos={running.get('pred_cos',0)/n:.4f}"
                print(f"[pretrain] {step}/{args.max_steps} ({pct:4.1f}%) "
                      f"| {msg} | lr={args.lr*lr_at(step):.1e} "
                      f"lora_lr={args.lora_lr*lr_at(step):.1e} "
                      f"| {sps:.2f} it/s | ETA {eta_h:.1f}h",
                      flush=True)
                running, t0 = {}, time.time()
            if step % args.save_every == 0:
                save(f"step{step}")
            if step % args.state_save_every == 0:
                save_state("latest", step, epoch=epoch)
            if step >= args.max_steps:
                break
        epoch += 1

    save("final", full=True)
    save_state("final", step, epoch=epoch)
    rprint(rank, "[trainer] done.")
    if is_ddp:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
