#!/usr/bin/env python
"""Per-task offline diagnostic of the world-model's future-latent prediction.

Measures, per LIBERO task index, on training-distribution samples:

  pred_cos   : cosine(s_end_pred, s_end_gt), per-patch mean
  pred_l1    : L1(s_end_pred, s_end_gt)
  s0_cos     : cosine(s_0, s_end_gt) — "static baseline" (no prediction).
               pred_cos must beat this for the model to be doing real work.

Action head is NOT touched — this isolates the world model.

Usage:
  python scripts/diag_pred_per_task.py \
    --ckpt results/dino_mamba_diff_lora_ddp_libero_10/stage2/checkpoints/mamba_wm_final.pt \
    --data_root /mnt/4TB_2/jamvla/datasets/LIBERO --data_mix libero_10 \
    --samples_per_task 50
"""
import argparse, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from starVLA.model.tools import read_mode_config
from starVLA.model.framework.share_tools import dict_to_namespace
from starVLA.model.framework import build_framework
from starVLA.dataloader.lerobot_datasets import get_vla_dataset, collate_fn


def build_cfg(args, model_config):
    cfg = dict_to_namespace(model_config)
    cfg.framework.name = args.framework
    cfg.trainer.pretrained_checkpoint = None
    cfg.datasets.vla_data.data_root_dir = args.data_root
    cfg.datasets.vla_data.data_mix = args.data_mix
    from omegaconf import OmegaConf
    cfg.framework.mamba_wm = OmegaConf.create({
        "encoder_depth": 6, "predictor_depth": 8, "idm_hidden": 1024,
        "consist_weight": 1.0, "dino_backbone": args.dino_backbone,
        "mask_ratio": 0.5, "ema_momentum": 0.996, "train_dino": False,
        "action_mamba_layers": 5, "action_embed_dim": 256, "action_inference_steps": 10,
        "qwen_lora": args.qwen_lora, "lora_r": args.lora_r, "lora_alpha": args.lora_alpha,
    })
    return cfg


@torch.no_grad()
def pred_only_step(model, batch, device):
    """Re-do the predictor-only path of forward() but without computing action loss.
    Returns per-sample (pred_cos, pred_l1, s0_cos)."""
    examples = batch  # collate_fn just returns the list back for this framework
    batch_images = [e["image"] for e in examples]
    instructions = [e["lang"] for e in examples]
    videos = torch.from_numpy(np.stack([e["video"] for e in examples])).to(device)
    frames = videos[:, :, [0, model.endpoint]]

    with torch.autocast("cuda", dtype=torch.float32):
        s = model._dino_latents(frames).float()
        s_0, s_end_gt = s[:, 0], s[:, 1]
    action_tokens = model._qwen_action_tokens(batch_images, instructions)
    s_end_pred = model.mamba_predictor(s_0, action_tokens)[:, 0]

    pred_cos = F.cosine_similarity(s_end_pred, s_end_gt, dim=-1).mean(dim=-1)  # [B]
    pred_l1 = (s_end_pred - s_end_gt).abs().mean(dim=(-1, -2))                  # [B]
    s0_cos = F.cosine_similarity(s_0, s_end_gt, dim=-1).mean(dim=-1)            # [B]
    return pred_cos.cpu().numpy(), pred_l1.cpu().numpy(), s0_cos.cpu().numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--framework", default="VLA_DINO_Mamba_Diff")
    p.add_argument("--dino_backbone", default="dinov2_vitb14")
    p.add_argument("--qwen_lora", action="store_true", default=True)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--data_root", default="/mnt/4TB_2/jamvla/datasets/LIBERO")
    p.add_argument("--data_mix", default="libero_10")
    p.add_argument("--samples_per_task", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--cuda", type=int, default=0)
    p.add_argument("--out_json", default="results/eval/diag_pred_per_task.json")
    args = p.parse_args()

    device = torch.device(f"cuda:{args.cuda}")
    torch.cuda.set_device(device)

    model_config, _ = read_mode_config(Path(args.ckpt))
    cfg = build_cfg(args, model_config)
    print(f"[diag] building framework {args.framework}...")
    model = build_framework(cfg)
    model.load_backbone(args.ckpt)
    if args.qwen_lora:
        model.apply_qwen_lora(r=args.lora_r, alpha=args.lora_alpha)
    sd = torch.load(args.ckpt, map_location="cpu")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[diag] loaded ckpt (missing={len(missing)}, unexpected={len(unexpected)})")
    model = model.to(device).eval()

    print(f"[diag] building dataset {args.data_mix}...")
    obs_indices = getattr(model, "obs_indices", None)
    dataset = get_vla_dataset(
        data_cfg=cfg.datasets.vla_data,
        action_horizon=model.horizon,
        video_horizon=model.horizon + 1,
        obs_indices=obs_indices,
    )
    n = len(dataset)
    print(f"[diag] {n} samples")

    probe = dataset[0]
    if "task_index" in probe:
        key_fn = lambda s: int(s["task_index"])
    elif "lang" in probe:
        key_fn = lambda s: s["lang"][0] if isinstance(s["lang"], (list, tuple)) else s["lang"]
    else:
        raise RuntimeError("can't find a task identifier in dataset samples")

    print(f"[diag] scanning to collect {args.samples_per_task} samples per task...")
    rng = np.random.default_rng(0)
    order = rng.permutation(n)
    task_to_indices = defaultdict(list)
    target = args.samples_per_task
    for i, idx in enumerate(order):
        s = dataset[int(idx)]
        k = key_fn(s)
        if len(task_to_indices[k]) < target:
            task_to_indices[k].append(int(idx))
        if len(task_to_indices) >= 10 and all(len(v) >= target for v in task_to_indices.values()):
            break
        if i % 2000 == 0:
            mn = min((len(v) for v in task_to_indices.values()), default=0)
            print(f"  scan {i}/{n} | tasks seen: {len(task_to_indices)} | min/task: {mn}")
    print(f"[diag] collected tasks: {len(task_to_indices)}")

    results = {}
    BS = args.batch_size
    for tkey, idxs in task_to_indices.items():
        idxs = idxs[:target]
        pcs, pls, s0s = [], [], []
        for i0 in range(0, len(idxs), BS):
            batch = [dataset[j] for j in idxs[i0:i0 + BS]]
            pc, pl, s0c = pred_only_step(model, batch, device)
            pcs.append(pc); pls.append(pl); s0s.append(s0c)
        pcs = np.concatenate(pcs); pls = np.concatenate(pls); s0s = np.concatenate(s0s)
        results[str(tkey)] = {
            "n": int(pcs.size),
            "pred_cos_mean": float(pcs.mean()),
            "pred_cos_std":  float(pcs.std()),
            "pred_l1_mean":  float(pls.mean()),
            "s0_cos_mean":   float(s0s.mean()),
            "delta_cos":     float((pcs - s0s).mean()),  # uplift over static baseline
        }
        r = results[str(tkey)]
        print(f"  task={str(tkey)[:55]:<55} | pred_cos={r['pred_cos_mean']:.4f}±{r['pred_cos_std']:.3f}"
              f" | s0_cos={r['s0_cos_mean']:.4f} | Δ={r['delta_cos']:+.4f} | pred_l1={r['pred_l1_mean']:.4f}")

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[diag] saved -> {out_path}")

    print("\n=== summary (sorted by delta_cos asc → weakest world-model first) ===")
    rows = sorted(results.items(), key=lambda kv: kv[1]["delta_cos"])
    print(f"{'task':<55} {'pred_cos':>9} {'s0_cos':>8} {'Δcos':>7} {'pred_l1':>8}")
    for k, v in rows:
        print(f"{str(k)[:54]:<55} {v['pred_cos_mean']:>9.4f} "
              f"{v['s0_cos_mean']:>8.4f} {v['delta_cos']:>+7.4f} {v['pred_l1_mean']:>8.4f}")


if __name__ == "__main__":
    main()
