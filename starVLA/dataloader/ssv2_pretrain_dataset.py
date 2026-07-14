"""SSv2 (20BN Something-Something-v2) minimal dataset for Streaming Mamba pretraining.

Loads 3 frames (t-H, t, t+H) from a single webm view + label as language instruction.
Wrist view is faked by duplicating the exterior view (SSv2 is single-view).
"""

from __future__ import annotations

import gc
import json
import os
import random
import time
from dataclasses import dataclass
from typing import Optional

import av
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


SSV2_FPS = 12  # SSv2 webm avg frame rate
SSV2_VIDEO_SUBDIR = "20bn-something-something-v2"


@dataclass
class SSv2Sample:
    video_id: str
    label: str  # natural-language "template + placeholders" filled sentence


def _load_split(root: str, split: str) -> list[SSv2Sample]:
    """Load train/validation json into a flat SSv2Sample list."""
    fname = {"train": "train.json", "val": "validation.json", "validation": "validation.json"}[split]
    with open(os.path.join(root, fname), "r") as f:
        data = json.load(f)
    out = []
    for item in data:
        vid = str(item["id"])
        lab = item.get("label", "").strip()
        if not lab:
            continue
        out.append(SSv2Sample(video_id=vid, label=lab))
    return out


class SSv2PretrainDataset(Dataset):
    """Minimal SSv2 loader for predictor pretraining.

    Each sample returns:
      images:      list[list[PIL.Image]], outer length 3 (t-H, t, t+H),
                   inner length = 2 (single view duplicated to match Droid's [exterior, wrist]).
      instruction: str (label like "holding potato next to vicks vaporub bottle")
      dataset:     "ssv2"
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        obs_horizon: int = 7,
        image_size: int = 256,
        views_out: int = 2,   # duplicate single-view to reach this count
        seed: int = 0,
    ):
        self.root = root
        self.H = obs_horizon
        self.image_size = image_size
        self.views_out = views_out
        self._rng = random.Random(seed)

        self._samples = _load_split(root, split)
        if not self._samples:
            raise RuntimeError(f"No SSv2 {split} samples found under {root}")

        self._video_dir = os.path.join(root, "videos", SSV2_VIDEO_SUBDIR)

    def __len__(self) -> int:
        return len(self._samples)

    def _video_path(self, video_id: str) -> str:
        return os.path.join(self._video_dir, f"{video_id}.webm")

    def _decode_frames(self, path: str, want_idxs: list[int], ep_len: int) -> list[Image.Image]:
        """Decode the whole webm (short, ~50-60 frames each) and pick the requested indices."""
        container = av.open(path)
        try:
            stream = container.streams.video[0]
            arrs: list[np.ndarray] = []
            for f in container.decode(stream):
                arrs.append(f.to_ndarray(format="rgb24"))
        finally:
            container.close()

        if not arrs:
            raise ValueError(f"No frames decoded from {path}")

        n = len(arrs)
        pil = []
        for f_idx in want_idxs:
            i = max(0, min(n - 1, f_idx))
            img = Image.fromarray(arrs[i], mode="RGB")
            if img.size != (self.image_size, self.image_size):
                img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
            pil.append(img)
        return pil

    def _probe_length(self, path: str) -> int:
        """Cheap length probe: decode all pts (avoid full pixel decode).

        SSv2 webms are tiny; full decode is ~20ms so we just full-decode
        inside `_decode_frames` and use its returned count implicitly.
        This method is kept only for the _frame_indices choice — we take
        a fixed guess (~50 frames typical) and clamp inside decode.
        """
        return 60  # nominal; actual clamp done in _decode_frames using len(arrs)

    def _sample_indices(self) -> list[int]:
        """Pick a random base t and return [t-H, t, t+H], clamped by caller."""
        # Nominal episode ~60 frames; base ~ [H, ep-H) picks the "central" range.
        # Clamping to real length happens inside _decode_frames.
        base = self._rng.randint(self.H, max(self.H + 1, 60 - self.H))
        return [base - self.H, base, base + self.H]

    def _try_sample(self, idx: int) -> dict:
        s = self._samples[idx]
        path = self._video_path(s.video_id)

        want_idxs = self._sample_indices()
        frames = self._decode_frames(path, want_idxs, ep_len=None)

        # Duplicate the single view to match Droid's [exterior, wrist] layout.
        images = [[frames[t]] * self.views_out for t in range(3)]

        return {
            "images": images,
            "instruction": s.label,
            "dataset": "ssv2",
            "video_id": s.video_id,
        }

    def __getitem__(self, idx: int) -> dict:
        """Fetch one sample with retry + backoff (see droid_pretrain_dataset)."""
        max_retries = 5
        last_err: Optional[Exception] = None
        for attempt in range(max_retries):
            try:
                return self._try_sample(idx)
            except (MemoryError, OSError, av.error.FFmpegError, ValueError) as e:
                last_err = e
                gc.collect()
                wait = 2.0 * (attempt + 1)
                print(f"[ssv2_dataset] retry {attempt+1}/{max_retries} "
                      f"(idx={idx}, err={type(e).__name__}: {e}); "
                      f"sleeping {wait:.1f}s", flush=True)
                time.sleep(wait)
        print(f"[ssv2_dataset] falling back to random idx after {max_retries} "
              f"failures on idx={idx} (last err: {last_err})", flush=True)
        for _ in range(8):
            alt = self._rng.randint(0, len(self._samples) - 1)
            try:
                return self._try_sample(alt)
            except Exception:
                continue
        raise RuntimeError(
            f"SSv2 dataset: could not fetch any sample after retries "
            f"(last err on idx={idx}: {last_err})"
        )
