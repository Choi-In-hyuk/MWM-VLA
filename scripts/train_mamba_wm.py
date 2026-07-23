#!/usr/bin/env python
# Copyright 2026 VLA-JEPA research. MIT License.
"""
Lean single-GPU trainer for the Mamba latent world model (VLA_JEPA_Mamba).

Only the three new Mamba modules are trained; the Qwen / V-JEPA / flow-matching
backbone is loaded frozen from a pretrained VLA_JEPA checkpoint. The output dir
mirrors the eval-harness layout (config.yaml + dataset_statistics.json +
checkpoints/*.pt) so the existing LIBERO server can evaluate it directly via
`predict_action` (which uses the WM+ID inference path, V-JEPA dropped).

Example:
  python scripts/train_mamba_wm.py \
    --backbone_ckpt /home/choi/data/checkpoints/VLA-JEPA/LIBERO/checkpoints/VLA-JEPA-LIBERO.pt \
    --data_root /home/choi/data/datasets/LIBERO --data_mix libero_10 \
    --output_dir results/mamba_wm_libero10 --batch_size 8 --max_steps 20000
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
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from starVLA.model.tools import read_mode_config
from starVLA.model.framework.share_tools import dict_to_namespace
from starVLA.model.framework import build_framework
from starVLA.dataloader.lerobot_datasets import get_vla_dataset, collate_fn


# ---------------------------------------------------------------- DDP helpers
def ddp_setup():
    """Initialize torch.distributed if launched via torchrun; else return single-GPU info.

    Some third-party packages (timm/accelerate/transformers/peft) auto-initialize
    the default process group as a side-effect of import when RANK/WORLD_SIZE
    are present. Skip our own init in that case so we don't double-initialize.

    Returns: (is_ddp, rank, world_size, local_rank)
    """
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
        print(*a, **kw)


def build_cfg(args):
    model_config, _ = read_mode_config(Path(args.backbone_ckpt))
    cfg = dict_to_namespace(model_config)
    cfg.framework.name = args.framework
    cfg.trainer.pretrained_checkpoint = None
    cfg.datasets.vla_data.data_root_dir = args.data_root
    cfg.datasets.vla_data.data_mix = args.data_mix
    # mamba world-model hyperparameters
    from omegaconf import OmegaConf
    cfg.framework.mamba_wm = OmegaConf.create({
        "encoder_depth": args.encoder_depth,
        "predictor_depth": args.predictor_depth,
        "idm_hidden": args.idm_hidden,
        "consist_weight": args.consist_weight,
        "dino_backbone": args.dino_backbone,
        "mask_ratio": args.mask_ratio,            # jepa: fraction of s_0 tokens hidden
        "ema_momentum": args.ema_momentum,        # jepa: EMA target-encoder momentum (->1.0)
        "train_dino": args.train_dino,            # A-b: unfreeze DINO in stage2
        "action_mamba_layers": args.action_mamba_layers,    # dual-mamba: Mamba #2 depth
        "action_embed_dim": args.action_embed_dim,          # dual-mamba: Mamba #2 width
        "action_inference_steps": args.action_inference_steps,
        # streaming Mamba-2 options (used by VLA_DINO_StreamingMamba)
        "stream_state_dim": args.stream_state_dim,
        "stream_depth": args.stream_depth,
        "stream_d_state": args.stream_d_state,
        "stream_d_conv": args.stream_d_conv,
        "stream_headdim": args.stream_headdim,
        "stream_chunk_size": args.stream_chunk_size,
        "seq_len_M": args.seq_len_M,
    })
    # Streaming framework needs per-frame proprioception.
    # startswith so subclasses (e.g. VLA_DINO_StreamingMamba_FutureOnly) also get it.
    if args.framework.startswith("VLA_DINO_StreamingMamba"):
        cfg.datasets.vla_data.with_state = True
    return cfg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone_ckpt", required=True)
    p.add_argument("--framework", default="VLA_DINO_Mamba",
                   help="framework name (VLA_DINO_Mamba: DINO encoder; VLA_JEPA_Mamba: V-JEPA distill)")
    p.add_argument("--dino_backbone", default="dinov2_vitb14",
                   help="DINOv2 variant for VLA_DINO_Mamba encoder")
    p.add_argument("--qwen_cache", default=None,
                   help="dir with precomputed Qwen action tokens (skips per-step Qwen forward)")
    p.add_argument("--qwen_lora", action="store_true",
                   help="fine-tune Qwen with LoRA (disables cache; Qwen runs live with grad)")
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--stage", choices=["encoder", "predictor", "id", "joint", "stage2", "jepa"], default="predictor",
                   help="predictor: train Mamba predictor (L_pred); id/stage2: train action head "
                        "(stage2 = VLA_DINO_Mamba_Diff: predictor fine-tune + diffusion head); "
                        "jepa: VLA_DINO_Mamba_JEPA stage-1 (online DINO + EMA target + masking, L_pred); joint: all")
    p.add_argument("--jepa_dino", default=None,
                   help="V3-b: checkpoint with JEPA-trained 'dino.*' weights to load into the encoder")
    p.add_argument("--train_dino", action="store_true",
                   help="A-b: unfreeze DINO in stage2 (action-anchored, detached target)")
    p.add_argument("--mask_ratio", type=float, default=0.5, help="jepa: fraction of s_0 tokens hidden")
    p.add_argument("--ema_momentum", type=float, default=0.996, help="jepa: EMA target momentum (cosine ->1.0)")
    p.add_argument("--resume_ckpt", default=None,
                   help="full VLA_JEPA_Mamba state_dict to continue from (e.g. stage-1 output for stage 2)")
    p.add_argument("--data_root", default="/home/choi/data/datasets/LIBERO")
    p.add_argument("--data_mix", default="libero_10")
    p.add_argument("--dataset_type", choices=["lerobot", "calvin"], default="lerobot",
                   help="lerobot=LIBERO/Droid LeRobot layout; calvin=CALVIN npz+lang layout")
    p.add_argument("--calvin_split", default="training",
                   help="calvin only: training|validation")
    p.add_argument("--output_dir", default="results/mamba_wm_libero10")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--max_steps", type=int, default=20000)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--save_every", type=int, default=2000)
    p.add_argument("--cuda", type=int, default=0)
    p.add_argument("--encoder_depth", type=int, default=6)
    p.add_argument("--predictor_depth", type=int, default=8)
    p.add_argument("--idm_hidden", type=int, default=1024)
    p.add_argument("--consist_weight", type=float, default=1.0)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    # dual-mamba action head (VLA_DINO_DualMamba) — Mamba #2 hparams
    p.add_argument("--action_mamba_layers", type=int, default=5,
                   help="dual-mamba: depth of the Mamba action decoder (Mamba #2)")
    p.add_argument("--action_embed_dim", type=int, default=256,
                   help="dual-mamba: hidden width of the Mamba action decoder")
    p.add_argument("--action_inference_steps", type=int, default=10,
                   help="dual-mamba: flow-matching Euler steps at inference")
    # streaming Mamba-2 (VLA_DINO_StreamingMamba) — predictor hparams
    p.add_argument("--stream_state_dim", type=int, default=1024,
                   help="streaming-mamba: internal Mamba-2 hidden dim")
    p.add_argument("--stream_depth", type=int, default=12,
                   help="streaming-mamba: number of Mamba-2 blocks")
    p.add_argument("--stream_d_state", type=int, default=64,
                   help="streaming-mamba: SSM state-space dim per head")
    p.add_argument("--stream_d_conv", type=int, default=1,
                   help="streaming-mamba: 1-D conv width (1 keeps split-with-state equivalent to concat)")
    p.add_argument("--stream_headdim", type=int, default=64,
                   help="streaming-mamba: SSM head dim")
    p.add_argument("--stream_chunk_size", type=int, default=64,
                   help="streaming-mamba: internal scan chunk size (Mamba-2 SSD)")
    p.add_argument("--seq_len_M", type=int, default=2,
                   help="streaming-mamba: number of consecutive chunks per training sample (BPTT length)")
    # ---- full-state resume (model + optimizer + scheduler + step + rng).
    # `--resume_ckpt` (already above) is the weights-only handoff between stages
    # (e.g. stage1 -> stage2). `--resume_state` continues the SAME run from a
    # crash / preemption and restores optimizer momentum, LR schedule progress,
    # and RNG so loss curves stay continuous.
    p.add_argument("--resume_state", default=None,
                   help="path to a training_state_*.pt to resume an interrupted run")
    p.add_argument("--state_save_every", type=int, default=5000,
                   help="how often (steps) to write a full resumable training_state.pt")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    is_ddp, rank, world, local_rank = ddp_setup()
    device = torch.device(f"cuda:{local_rank}") if is_ddp else torch.device(f"cuda:{args.cuda}")
    if not is_ddp:
        torch.cuda.set_device(device)

    # Seed each rank distinctly so dataloaders/noise differ across workers, but
    # restoring from a state checkpoint will override this.
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    random.seed(args.seed + rank)

    out = Path(args.output_dir)
    if is_main(rank):
        (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    if is_ddp:
        dist.barrier()

    cfg = build_cfg(args)

    # ---- model: frozen backbone + trainable mamba modules
    model = build_framework(cfg)
    model.load_backbone(args.backbone_ckpt)
    if args.qwen_lora:                              # apply LoRA AFTER backbone load (key match)
        model.apply_qwen_lora(r=args.lora_r, alpha=args.lora_alpha)
        # record in cfg (AFTER build) so eval/from_pretrained re-applies LoRA before loading
        cfg.framework.mamba_wm.qwen_lora = True
        cfg.framework.mamba_wm.lora_r = args.lora_r
        cfg.framework.mamba_wm.lora_alpha = args.lora_alpha
    if args.jepa_dino:                             # V3-b: load JEPA-trained DINO encoder weights
        model.load_jepa_dino(args.jepa_dino)
    if args.qwen_cache and not args.qwen_lora:
        model.load_qwen_cache(args.qwen_cache)
    elif args.qwen_cache and args.qwen_lora:
        rprint(rank, "[trainer] --qwen_lora set: ignoring --qwen_cache (Qwen trains live)")
    if args.resume_ckpt:                           # continue from a prior stage (loads trained mamba weights)
        missing, unexpected = model.load_state_dict(torch.load(args.resume_ckpt, map_location="cpu"), strict=False)
        rprint(rank, f"[trainer] resumed mamba weights from {args.resume_ckpt} "
                     f"(missing={len(missing)}, unexpected={len(unexpected)})")
    model = model.to(device)
    model.set_stage(args.stage)                    # sets requires_grad + loss weights for the stage

    model.eval()                                   # frozen backbone stays in eval
    # set train() on any top-level submodule that has trainable params (framework-agnostic)
    for _, mod in model.named_children():
        if any(pm.requires_grad for pm in mod.parameters()):
            mod.train()

    trainable = [pm for pm in model.parameters() if pm.requires_grad]
    n_tr = sum(pm.numel() for pm in trainable)
    rprint(rank, f"[trainer] stage={args.stage}, trainable params: {n_tr/1e6:.1f}M "
                 f"| ddp={is_ddp} world={world} rank={rank}")

    # ---- persist config + norm stats for the eval harness (rank 0 only)
    from omegaconf import OmegaConf
    if is_main(rank):
        OmegaConf.save(cfg, out / "config.yaml")
        src_dir = Path(args.backbone_ckpt).parent.parent
        if (src_dir / "dataset_statistics.json").exists():
            shutil.copy(src_dir / "dataset_statistics.json", out / "dataset_statistics.json")

    # ---- data
    # frameworks may request explicit obs delta indices (e.g. VLA_DINO_Mamba_Temporal's past
    # buffer [-(K-1)..0, horizon]); else default to obs_0..obs_H (H+1 frames).
    obs_indices = getattr(model, "obs_indices", None)
    # frameworks can also override video_horizon (number of consecutive frames the
    # loader fetches) for multi-chunk streaming windows.
    video_horizon = getattr(model, "video_horizon", model.horizon + 1)
    if args.dataset_type == "calvin":
        from starVLA.dataloader.calvin_dataset import CalvinDataset
        dataset = CalvinDataset(
            root=args.data_root, split=args.calvin_split,
            obs_horizon=model.horizon, action_horizon=model.horizon,
            image_size=cfg.datasets.vla_data.get("resolution_size", 256),
            seed=args.seed + rank,
        )
    else:
        dataset = get_vla_dataset(
            data_cfg=cfg.datasets.vla_data, action_horizon=model.horizon,
            video_horizon=video_horizon, obs_indices=obs_indices,
        )
    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True,
                                 drop_last=True) if is_ddp else None
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=(sampler is None),
        sampler=sampler, num_workers=args.num_workers, collate_fn=collate_fn,
        drop_last=True, pin_memory=False,
    )
    rprint(rank, f"[trainer] dataset {args.data_mix}: {len(dataset)} samples, "
                 f"{len(loader)} steps/epoch (per rank)")

    # ---- DDP wrap (after .to(device); find_unused_parameters=True because some
    # frameworks have sub-modules that don't participate in every forward, e.g.
    # the cond_proj layer is unused in stage='predictor', and the action head
    # only runs when loss_weights['action'] > 0).
    if is_ddp:
        model_engine = DDP(
            model, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=True,
            broadcast_buffers=False,
        )
        model_for_call = model_engine
        model_root = model_engine.module
    else:
        model_for_call = model
        model_root = model

    # ---- optim + cosine schedule with linear warmup
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))

    def lr_at(step):
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        prog = (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))

    # framework may override which params the lean intermediate save keeps (e.g. JEPA also
    # trains the online DINO encoder, so it must be saved to resume the next stage)
    mamba_prefixes = tuple(getattr(model_root, "save_prefixes",
                                   ("mamba_encoder", "mamba_predictor", "idm")))

    def _lean_sd():
        sd = model_root.state_dict()
        if hasattr(model_root, "lean_save_key"):
            return {k: v for k, v in sd.items() if model_root.lean_save_key(k)}
        return {k: v for k, v in sd.items() if k.startswith(mamba_prefixes)}

    def save(tag, full=False):
        """full=True -> entire model (eval-ready, ~8.7GB). Else only the Mamba
        modules (~1.9GB), enough to resume the next curriculum stage. rank-0 only."""
        if not is_main(rank):
            return
        path = out / "checkpoints" / f"mamba_wm_{tag}.pt"
        sd = model_root.state_dict() if full else _lean_sd()
        torch.save(sd, path)
        print(f"[trainer] saved {path} ({'full' if full else 'mamba-only'})", flush=True)

    def save_state(tag, step):
        """Full resumable training state: model (lean) + opt + step + rng. rank-0 only."""
        if not is_main(rank):
            return
        path = out / "checkpoints" / f"training_state_{tag}.pt"
        torch.save({
            "step": step,
            "model": _lean_sd(),
            "optimizer": opt.state_dict(),
            "args": vars(args),
            "rng": {
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all(),
                "numpy": np.random.get_state(),
                "python": random.getstate(),
            },
        }, path)
        print(f"[trainer] saved training state -> {path} (step={step})", flush=True)

    # ---- resume full training state (model + opt + step + rng), if provided
    start_step = 0
    if args.resume_state:
        ckpt = torch.load(args.resume_state, map_location="cpu")
        missing, unexpected = model_root.load_state_dict(ckpt["model"], strict=False)
        opt.load_state_dict(ckpt["optimizer"])
        start_step = int(ckpt["step"])
        rng = ckpt.get("rng", {})
        if "torch" in rng:
            torch.set_rng_state(rng["torch"])
        if "cuda" in rng:
            try:
                torch.cuda.set_rng_state_all(rng["cuda"])
            except Exception as e:
                rprint(rank, f"[trainer] warn: cuda rng restore skipped ({e})")
        if "numpy" in rng:
            np.random.set_state(rng["numpy"])
        if "python" in rng:
            random.setstate(rng["python"])
        rprint(rank, f"[trainer] RESUMED from {args.resume_state} at step={start_step} "
                     f"(missing={len(missing)}, unexpected={len(unexpected)})")

    steps_per_epoch = max(1, len(loader))
    # framework-agnostic: show the active (nonzero-weight) component losses + any
    # cosine-similarity metrics the framework reports (keys ending in "_cos").
    active = [k for k in model_root.loss_weights if model_root.loss_weights[k] > 0]
    show = [f"{k}_loss" for k in active]   # *_cos keys appended dynamically at log time

    step, t0 = start_step, time.time()
    epoch = step // max(1, steps_per_epoch)
    opt.zero_grad(set_to_none=True)
    running = {}
    while step < args.max_steps:
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in loader:
            # expose training progress to the framework (used by schedules like
            # change-mask ease-out). DDP-safe: writes to the unwrapped module.
            model_root._step = step
            model_root._max_steps = args.max_steps
            losses = model_for_call(batch)
            (losses["loss"] / args.grad_accum).backward()
            for k, v in losses.items():
                running[k] = running.get(k, 0.0) + float(v)

            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
                for g in opt.param_groups:
                    g["lr"] = args.lr * lr_at(step)
                opt.step()
                opt.zero_grad(set_to_none=True)
                if hasattr(model_root, "post_step"):  # e.g. JEPA EMA target-encoder update
                    model_root.post_step(step, args.max_steps)

            step += 1
            if step % args.log_every == 0 and is_main(rank):
                n = args.log_every
                dt = time.time() - t0
                sps = n / dt
                pct = 100.0 * step / args.max_steps
                ep = step / steps_per_epoch
                eta_h = (args.max_steps - step) / max(sps, 1e-6) / 3600
                extra = sorted(k for k in running if k.endswith("_cos") or k.endswith("_std"))
                msg = " ".join(f"{m.replace('_loss','')}={running.get(m, 0.0)/n:.4f}" for m in show + extra)
                print(f"[{args.stage}] {step}/{args.max_steps} ({pct:4.1f}%) ep{int(ep)} "
                      f"| {msg} | lr={args.lr*lr_at(step):.1e} | {sps:.2f} it/s | ETA {eta_h:.1f}h",
                      flush=True)
                running, t0 = {}, time.time()
            if step % args.save_every == 0:
                save(f"step{step}")
            if step % args.state_save_every == 0:
                save_state("latest", step)
            if step >= args.max_steps:
                break
        epoch += 1

    save("final", full=True)
    save_state("final", step)
    rprint(rank, "[trainer] done.")
    if is_ddp:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
