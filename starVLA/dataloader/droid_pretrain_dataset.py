"""Droid v3 minimal dataset for Streaming Mamba pretraining.

Loads 3 frames per sample (t-H, t, t+H) from 2 views + language instruction.
State/action are intentionally ignored.
"""

from __future__ import annotations

import gc
import os
import random
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchcodec.decoders import VideoDecoder


DROID_FPS = 15
DROID_VIEWS = ("observation.images.exterior_1_left", "observation.images.wrist_left")

# Worker-local LRU cache of torchcodec VideoDecoder objects. Each decoder holds
# a lightweight index into the mp4 (no giant mmap of the full file). We still
# cache to avoid re-parsing metadata for every sample, but the memory footprint
# is far smaller than PyAV containers, so no vm.max_map_count pressure.
_DECODER_CACHE_SIZE = 8
_decoder_cache: "OrderedDict[str, VideoDecoder]" = OrderedDict()


def _get_decoder(path: str) -> VideoDecoder:
    d = _decoder_cache.get(path)
    if d is not None:
        _decoder_cache.move_to_end(path)
        return d
    while len(_decoder_cache) >= _DECODER_CACHE_SIZE:
        _decoder_cache.popitem(last=False)  # torchcodec releases on GC
    # `approximate` seek_mode is O(1) — no full file scan on open.
    d = VideoDecoder(path, seek_mode="approximate")
    _decoder_cache[path] = d
    return d


def _evict_decoder(path: str):
    _decoder_cache.pop(path, None)


@dataclass
class DroidEpisode:
    episode_index: int
    length: int
    instruction: str
    data_chunk: int
    data_file: int
    dataset_from: int  # global row idx in data parquet
    dataset_to: int
    video_chunk: dict  # view_key -> chunk_idx
    video_file: dict   # view_key -> file_idx
    video_from_ts: dict  # view_key -> from_timestamp (sec)
    video_to_ts: dict    # view_key -> to_timestamp (sec)


def _load_episode_index(root: str) -> list[DroidEpisode]:
    """Scan meta/episodes/chunk-*/file-*.parquet and build a flat episode list.

    Filters out episodes with empty language_instruction or length < 3H+1.
    """
    meta_dir = os.path.join(root, "meta", "episodes")
    episodes: list[DroidEpisode] = []
    chunks = sorted(os.listdir(meta_dir))
    for chunk in chunks:
        chunk_dir = os.path.join(meta_dir, chunk)
        for fname in sorted(os.listdir(chunk_dir)):
            path = os.path.join(chunk_dir, fname)
            cols = [
                "episode_index",
                "tasks",
                "length",
                "data/chunk_index",
                "data/file_index",
                "dataset_from_index",
                "dataset_to_index",
            ]
            for v in DROID_VIEWS:
                cols += [
                    f"videos/{v}/chunk_index",
                    f"videos/{v}/file_index",
                    f"videos/{v}/from_timestamp",
                    f"videos/{v}/to_timestamp",
                ]
            df = pq.read_table(path, columns=cols).to_pandas()
            for _, row in df.iterrows():
                tasks = row["tasks"]
                # `tasks` may be list[str] or single string in this schema.
                if isinstance(tasks, (list, np.ndarray)):
                    instr = tasks[0] if len(tasks) > 0 else ""
                else:
                    instr = str(tasks) if tasks is not None else ""
                if not instr or not str(instr).strip():
                    continue
                episodes.append(
                    DroidEpisode(
                        episode_index=int(row["episode_index"]),
                        length=int(row["length"]),
                        instruction=str(instr).strip(),
                        data_chunk=int(row["data/chunk_index"]),
                        data_file=int(row["data/file_index"]),
                        dataset_from=int(row["dataset_from_index"]),
                        dataset_to=int(row["dataset_to_index"]),
                        video_chunk={v: int(row[f"videos/{v}/chunk_index"]) for v in DROID_VIEWS},
                        video_file={v: int(row[f"videos/{v}/file_index"]) for v in DROID_VIEWS},
                        video_from_ts={v: float(row[f"videos/{v}/from_timestamp"]) for v in DROID_VIEWS},
                        video_to_ts={v: float(row[f"videos/{v}/to_timestamp"]) for v in DROID_VIEWS},
                    )
                )
    return episodes


class DroidPretrainDataset(Dataset):
    """Minimal Droid loader: video-only + language, no state/action.

    Each sample returns:
      images:      list[list[PIL.Image]], outer length 3 (t-H, t, t+H),
                   inner length = len(views) (default 2: exterior_1_left, wrist_left).
      instruction: str
      dataset:     "droid"
    """

    def __init__(
        self,
        root: str,
        obs_horizon: int = 7,
        image_size: int = 256,
        views: tuple[str, ...] = DROID_VIEWS,
        video_backend: str = "pyav",
        seed: int = 0,
    ):
        self.root = root
        self.H = obs_horizon
        self.image_size = image_size
        self.views = views
        self.video_backend = video_backend
        self._rng = random.Random(seed)

        # Build episode index (filter empty language, length >= 1).
        self._episodes = _load_episode_index(root)
        if not self._episodes:
            raise RuntimeError(f"No valid Droid episodes found under {root}")

    def __len__(self) -> int:
        return len(self._episodes)

    def _video_path(self, view: str, chunk: int, file_idx: int) -> str:
        return os.path.join(
            self.root, "videos", view,
            f"chunk-{chunk:03d}", f"file-{file_idx:03d}.mp4",
        )

    def _sample_base(self, ep_length: int) -> int:
        """Random base frame index within the episode.

        Front-padding is applied for base < H (past = frame 0).
        Back-padding is applied for base + H >= ep_length (target = last frame).
        """
        # allow any base in [0, ep_length - 1]; boundaries handled by clamp.
        return self._rng.randint(0, max(0, ep_length - 1))

    def _frame_indices(self, base: int, ep_length: int) -> list[int]:
        past = max(0, base - self.H)
        target = min(ep_length - 1, base + self.H)
        return [past, base, target]

    def _decode_frames(self, view: str, ep: DroidEpisode, frame_idxs: list[int]) -> list[Image.Image]:
        """torchcodec seek + decode. Returns one PIL per requested episode-frame index.

        Droid mp4 files concatenate many episodes (~500MB / 6000s / 96k frames per file).
        torchcodec's VideoDecoder does keyframe-aware seeks under the hood without
        opening the whole file into RAM, so we can index by timestamp cheaply.
        """
        path = self._video_path(view, ep.video_chunk[view], ep.video_file[view])
        base_ts = ep.video_from_ts[view]
        want_ts = [base_ts + f / DROID_FPS for f in frame_idxs]

        decoder = _get_decoder(path)
        try:
            frames_uint8 = [decoder.get_frame_played_at(ts).data for ts in want_ts]
            # torchcodec returns tensors as [3, H, W] uint8; convert to PIL.
            arrs = [f.permute(1, 2, 0).contiguous().numpy() for f in frames_uint8]
        except Exception:
            _evict_decoder(path)
            raise

        pil = []
        for arr in arrs:
            img = Image.fromarray(arr, mode="RGB")
            if img.size != (self.image_size, self.image_size):
                img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
            pil.append(img)
        return pil

    def _try_sample(self, idx: int) -> dict:
        ep = self._episodes[idx]
        base = self._sample_base(ep.length)
        f_idxs = self._frame_indices(base, ep.length)

        # For each of the 3 time slots, gather one PIL per view.
        per_view_frames = {v: self._decode_frames(v, ep, f_idxs) for v in self.views}
        images = []
        for t in range(3):
            images.append([per_view_frames[v][t] for v in self.views])

        return {
            "images": images,               # list[list[PIL]], [3][V]
            "instruction": ep.instruction,   # str
            "dataset": "droid",
            "episode_index": ep.episode_index,
            "base_frame": base,
        }

    def __getitem__(self, idx: int) -> dict:
        """Fetch one sample with retry + backoff.

        On transient decoder failures (ENOMEM, corrupt frame, IO hiccup) we:
          1. flush the container cache + gc so FFmpeg buffers actually free
          2. sleep with linear backoff (2s, 4s, 6s, 8s)
          3. retry the SAME episode
          4. after `max_retries` on same idx, fall back to a random other idx
        The DataLoader worker never dies — it either returns a sample or
        keeps trying, so training just stalls briefly instead of crashing.
        """
        max_retries = 5
        last_err: Optional[Exception] = None
        for attempt in range(max_retries):
            try:
                return self._try_sample(idx)
            except (MemoryError, OSError, RuntimeError, ValueError) as e:
                last_err = e
                # Free everything we can before waiting.
                for path in list(_decoder_cache.keys()):
                    _evict_decoder(path)
                gc.collect()
                wait = 2.0 * (attempt + 1)
                print(f"[droid_dataset] retry {attempt+1}/{max_retries} "
                      f"(idx={idx}, err={type(e).__name__}: {e}); "
                      f"sleeping {wait:.1f}s", flush=True)
                time.sleep(wait)
        # Same-idx retries exhausted. Try random fallback a few times.
        print(f"[droid_dataset] falling back to random idx after {max_retries} "
              f"failures on idx={idx} (last err: {last_err})", flush=True)
        for _ in range(8):
            alt = self._rng.randint(0, len(self._episodes) - 1)
            try:
                return self._try_sample(alt)
            except Exception:
                continue
        raise RuntimeError(
            f"Droid dataset: could not fetch any sample after retries "
            f"(last err on idx={idx}: {last_err})"
        )


def droid_pretrain_collate(batch: list[dict]) -> dict:
    """Collate that keeps `images` as nested lists (matches predict_action interface).

    Returns:
      batch_images: list[list[PIL]] of length B, each inner list has V views for the
                    reference frame. We DO NOT stack across time here; instead we
                    expose three parallel keys (past/present/target) so the framework
                    can DINO-embed them independently.
    """
    past   = [item["images"][0] for item in batch]   # list of [V PIL], length B
    present = [item["images"][1] for item in batch]
    target = [item["images"][2] for item in batch]
    return {
        "images_past": past,
        "images_present": present,
        "images_target": target,
        "instructions": [item["instruction"] for item in batch],
        "dataset": [item["dataset"] for item in batch],
    }
