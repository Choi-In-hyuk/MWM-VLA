"""CALVIN dataset loader for streaming Mamba VLA.

Each sample = one language-annotated segment sampled at a random base timestep,
returning 3 frames (t-H, t, t+H) x 2 views + the 7-step relative action chunk
starting at t + language instruction + 8-D robot state.

Mirrors starVLA/dataloader/lerobot_datasets.py conventions so the same model
consumes CALVIN and LIBERO uniformly:
  - obs indices: [-H, 0, +H]  (H = obs_horizon, default 7)
  - action indices: [0 .. H-1] absolute from base (7 rel_actions)
  - image size: 256 x 256 (resize from CALVIN native 200 static / 84 gripper)
  - robot state: 8-D  = [EE pos (3), EE ori euler (3), gripper width (1), gripper action (1)]
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


@dataclass
class CalvinAnn:
    """One language-annotated segment."""
    frame_start: int
    frame_end: int
    instruction: str
    task: str


def _load_annotations(root: str, split: str) -> list[CalvinAnn]:
    """Load lang_annotations/auto_lang_ann.npy for a split ('training'|'validation')."""
    ann_path = os.path.join(root, split, "lang_annotations", "auto_lang_ann.npy")
    d = np.load(ann_path, allow_pickle=True).item()
    anns_txt = d["language"]["ann"]
    tasks = d["language"]["task"]
    idxs = d["info"]["indx"]
    out = []
    for i, (s, e) in enumerate(idxs):
        out.append(CalvinAnn(
            frame_start=int(s), frame_end=int(e),
            instruction=str(anns_txt[i]),
            task=str(tasks[i]),
        ))
    return out


def _extract_robot_state_8d(robot_obs_15d: np.ndarray) -> np.ndarray:
    """CALVIN robot_obs (15D) -> LIBERO-like 8D state.
    CALVIN layout: [0:3 EE pos | 3:6 EE ori euler | 6 gripper width | 7:14 joints | 14 gripper action]
    We take: EE pos (3) + EE ori (3) + gripper width (1) + gripper action (1) = 8D.
    """
    return np.concatenate([
        robot_obs_15d[0:6],       # EE pos + ori
        robot_obs_15d[6:7],       # gripper width
        robot_obs_15d[14:15],     # gripper action
    ]).astype(np.float32)          # shape (8,)


class CalvinDataset(Dataset):
    """CALVIN language-annotated sampler.

    Each __getitem__ returns:
      video:       np.uint8 [V, 3, H, W, 3]  (V=2 views: static, gripper; 3 frames past/present/target)
      state_full:  np.float32 [3, 8]         robot state at [past, present, target]
      action:      np.float32 [action_horizon, 7]   rel_actions[base : base+H]
      lang:        str
    """

    def __init__(
        self,
        root: str,
        split: str = "training",
        obs_horizon: int = 7,
        action_horizon: int = 7,
        image_size: int = 256,
        seed: int = 0,
    ):
        self.root = root
        self.split = split
        self.H = obs_horizon
        self.action_horizon = action_horizon
        self.image_size = image_size
        self._rng = random.Random(seed)

        self._anns = _load_annotations(root, split)
        if not self._anns:
            raise RuntimeError(f"No CALVIN annotations under {root}/{split}")
        self._split_dir = os.path.join(root, split)

    def __len__(self) -> int:
        return len(self._anns)

    def _episode_path(self, frame_idx: int) -> str:
        return os.path.join(self._split_dir, f"episode_{frame_idx:07d}.npz")

    def _load_frame(self, frame_idx: int) -> dict:
        return dict(np.load(self._episode_path(frame_idx), allow_pickle=True))

    def _resize_view(self, img: np.ndarray) -> np.ndarray:
        pil = Image.fromarray(img)
        if pil.size != (self.image_size, self.image_size):
            pil = pil.resize((self.image_size, self.image_size), Image.BILINEAR)
        return np.asarray(pil, dtype=np.uint8)

    def __getitem__(self, idx: int) -> dict:
        ann = self._anns[idx]
        # Random base timestep in [frame_start, frame_end - action_horizon]
        low = ann.frame_start
        high = max(low + 1, ann.frame_end - self.action_horizon)
        base = self._rng.randint(low, high)

        # frame indices: past, present, target (front-pad if past < start of annotation)
        past = max(low, base - self.H)
        present = base
        target = min(ann.frame_end - 1, base + self.H)
        frame_idxs = [past, present, target]

        # Load 3 frames
        frames = [self._load_frame(fi) for fi in frame_idxs]

        # video: [V=2, 3, H, W, 3]
        static = np.stack([self._resize_view(f["rgb_static"]) for f in frames], axis=0)   # [3,H,W,3]
        gripper = np.stack([self._resize_view(f["rgb_gripper"]) for f in frames], axis=0)  # [3,H,W,3]
        video = np.stack([static, gripper], axis=0)                                       # [2,3,H,W,3]

        # state_full: [3, 8]
        state_full = np.stack([_extract_robot_state_8d(f["robot_obs"]) for f in frames], axis=0)

        # action: [action_horizon, 7]  — rel_actions from base
        actions = []
        for k in range(self.action_horizon):
            fi = min(ann.frame_end - 1, base + k)
            actions.append(np.load(self._episode_path(fi), allow_pickle=True)["rel_actions"].astype(np.float32))
        actions = np.stack(actions, axis=0)  # [H, 7]

        return {
            "video": video,
            "state_full": state_full,
            "action": actions,
            "lang": ann.instruction,
            "task": ann.task,
        }
