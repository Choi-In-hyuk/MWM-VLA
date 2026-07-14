"""SSv2 + Droid mixed dataset for Streaming Mamba predictor pretraining.

Presents a unified interface producing dicts with:
  images:      list[list[PIL]] of length 3 (t-H, t, t+H), each inner len = n_views
  instruction: str
  dataset:     "ssv2" or "droid"
"""

from __future__ import annotations

import random
from typing import Optional

from torch.utils.data import Dataset

from starVLA.dataloader.droid_pretrain_dataset import DroidPretrainDataset
from starVLA.dataloader.ssv2_pretrain_dataset import SSv2PretrainDataset


class SSv2DroidMixed(Dataset):
    """Sample-level 1:1 (or arbitrary weight) interleaving of SSv2 and Droid.

    Length = ssv2_weight * len(ssv2) + droid_weight * len(droid), normalized so
    default 1:1 means "one epoch = one pass over the larger dataset, and the
    smaller is oversampled (with replacement) to match."
    """

    def __init__(
        self,
        ssv2_root: str,
        droid_root: str,
        obs_horizon: int = 7,
        image_size: int = 256,
        ssv2_weight: float = 1.0,
        droid_weight: float = 1.0,
        seed: int = 0,
    ):
        self.ssv2 = SSv2PretrainDataset(
            root=ssv2_root, split="train",
            obs_horizon=obs_horizon, image_size=image_size, seed=seed,
        )
        self.droid = DroidPretrainDataset(
            root=droid_root,
            obs_horizon=obs_horizon, image_size=image_size, seed=seed + 1,
        )
        assert ssv2_weight > 0 and droid_weight > 0
        self._w = (ssv2_weight, droid_weight)
        # Nominal epoch length: sum of the two "effective" lengths.
        # With 1:1 weights we produce max(len_ssv2, len_droid) * 2 samples per
        # nominal epoch (each side hit equally often on average).
        n_max = max(len(self.ssv2), len(self.droid))
        self._nominal_len = int(n_max * (ssv2_weight + droid_weight))
        self._rng = random.Random(seed + 2)

    def __len__(self) -> int:
        return self._nominal_len

    def __getitem__(self, idx: int) -> dict:
        # Pick dataset by weighted coin flip. `idx` is only used as a shuffle
        # seed hint via rng.random() — this is an IterableDataset-style random
        # sampler wrapped in a map-style interface so DDP + num_workers work.
        r = self._rng.random()
        w_sum = self._w[0] + self._w[1]
        if r < self._w[0] / w_sum:
            j = self._rng.randint(0, len(self.ssv2) - 1)
            return self.ssv2[j]
        else:
            j = self._rng.randint(0, len(self.droid) - 1)
            return self.droid[j]


def pretrain_collate(batch: list[dict]) -> dict:
    """Collate for the mixed dataset.

    Returns:
      images_past:    list[list[PIL]] length B, each inner list has V views
      images_present: same
      images_target:  same
      instructions:   list[str]
      datasets:       list[str] ("ssv2" or "droid")
    """
    past    = [item["images"][0] for item in batch]
    present = [item["images"][1] for item in batch]
    target  = [item["images"][2] for item in batch]
    return {
        "images_past":    past,
        "images_present": present,
        "images_target":  target,
        "instructions":   [item["instruction"] for item in batch],
        "datasets":       [item["dataset"] for item in batch],
    }
