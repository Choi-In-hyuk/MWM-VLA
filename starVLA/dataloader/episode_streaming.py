# Copyright 2026 VLA-JEPA research. MIT License.
"""
Episode-streaming dataset for truncated-BPTT training of stateful world
models (Stage-2 of VLA_DINO_StreamingMamba).

Idea
----
A standard random-sampling dataloader gives the model independent windows
each step. For a stateful Mamba predictor we want CONSECUTIVE chunks of the
SAME episode to flow through the model so its SSM hidden state can be carried
across steps — and for the LOSS to backpropagate through that carried state
within a step (truncated BPTT).

We implement this as N "slots". Each slot is a stateful cursor over one
episode at a time: it yields the chunks of that episode in order, then
advances to a new episode. The trainer:

  * uses a batch of size N (one sample per slot per step)
  * keeps a per-slot SSM-state cache; passes state_in -> framework, gets
    state_out back, then `detach()`s and stores it for the next step
  * sees an `episode_start` flag per slot and zeros that slot's state when set

The model sees the same conditional distribution at training time as at
inference time (state evolved chunk-by-chunk within an episode).

Layout per chunk
----------------
Stage-2 single-chunk per step:
    indices = [c*H, (c+1)*H]  (chunk-start frame + GT target frame)
    where H = chunk horizon (7 for LIBERO).

The mixture dataset's `get_explicit(dataset_idx, trajectory_id, base_index)`
is used so we can drive (dataset, trajectory, step) explicitly.

Distribution / DDP
------------------
Each DDP rank owns N_per_rank slots = batch_size. Trajectories are sharded
across ranks (modulo) so different ranks see different episodes; within a
rank, slots also iterate through different trajectories. Episode end on any
slot just rotates that slot to the next un-assigned trajectory.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import IterableDataset


@dataclass
class _SlotState:
    """Per-slot iteration state."""
    dataset_idx: int = 0
    trajectory_id: int = 0
    chunk_cursor: int = 0          # which chunk in the trajectory we're on
    num_chunks: int = 0            # total chunks in the current trajectory
    just_started: bool = True      # set when a new episode begins on this slot


class EpisodeStreamingDataset(IterableDataset):
    """Slot-based stream of episode chunks for truncated-BPTT training.

    Each `next(iter(self))` yields a list of length `num_slots` of per-slot
    sample dicts. Each dict has an extra key `__episode_start__: bool` that
    the trainer uses to know when to reset that slot's SSM state.

    Parameters
    ----------
    mixture        : `LeRobotMixtureDataset` instance.
    num_slots      : per-rank batch size (one sample per slot per step).
    horizon        : chunk length in frames (LIBERO: 7).
    rank, world    : DDP rank / world size — used to shard trajectories.
    seed           : RNG seed for trajectory assignment.
    max_steps      : total steps to iterate before the dataset signals done.
                     (IterableDataset doesn't have __len__; we just stop here.)
    """

    def __init__(self, mixture, num_slots: int, horizon: int = 7,
                 rank: int = 0, world: int = 1, seed: int = 0,
                 max_steps: int = 10_000_000):
        super().__init__()
        self.mixture = mixture
        self.num_slots = num_slots
        self.horizon = horizon
        self.rank = rank
        self.world = world
        self.seed = seed
        self.max_steps = max_steps

        # Build a flat (dataset_idx, trajectory_id, num_chunks) catalog,
        # sharded across DDP ranks.
        self._catalog: List[Tuple[int, int, int]] = []
        for ds_i, ds in enumerate(mixture.datasets):
            for traj_id, traj_len in zip(ds.trajectory_ids, ds.trajectory_lengths):
                num_chunks = max(1, int(traj_len) // horizon)   # whole chunks only
                if num_chunks < 2:
                    continue   # need at least 2 frames (chunk start + target)
                self._catalog.append((ds_i, int(traj_id), num_chunks))

        # Deterministic per-rank shard
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(self._catalog))
        self._catalog = [self._catalog[i] for i in order]
        # rank takes every `world`-th entry
        self._catalog = [c for i, c in enumerate(self._catalog) if i % world == rank]
        if len(self._catalog) == 0:
            raise RuntimeError("EpisodeStreamingDataset: empty catalog for this rank")

        # Round-robin pointer through catalog (advances each time a slot
        # finishes its current trajectory).
        self._catalog_ptr = 0

    # ------------------------------------------------------------------ slot init
    def _assign_new_episode(self, slot: _SlotState):
        ds_i, traj_id, num_chunks = self._catalog[self._catalog_ptr % len(self._catalog)]
        self._catalog_ptr += 1
        slot.dataset_idx = ds_i
        slot.trajectory_id = traj_id
        slot.num_chunks = num_chunks
        slot.chunk_cursor = 0
        slot.just_started = True

    # ------------------------------------------------------------------ iterate
    def __iter__(self):
        # Init all slots
        slots = [_SlotState() for _ in range(self.num_slots)]
        for s in slots:
            self._assign_new_episode(s)

        for step in range(self.max_steps):
            batch = []
            for s in slots:
                # If the slot has consumed all chunks in its episode (one full
                # pass: chunk 0, 1, ..., num_chunks-1), rotate to a new episode.
                # The condition is `>= num_chunks` so the LAST chunk is also
                # produced — `chunk_cursor == num_chunks - 1` still yields one
                # more sample, then on the next step the rotation triggers.
                if s.chunk_cursor >= s.num_chunks:
                    self._assign_new_episode(s)

                # Build the sample at frame = chunk_cursor * H
                base_index = s.chunk_cursor * self.horizon
                sample = self.mixture.get_explicit(s.dataset_idx, s.trajectory_id, base_index)
                sample["__episode_start__"] = s.just_started
                batch.append(sample)
                s.just_started = False
                s.chunk_cursor += 1

            yield batch


def episode_streaming_collate_fn(batch):
    """The dataset already yields a list of length `num_slots`. The DataLoader
    is configured with `batch_size=None` so this collate is a no-op."""
    return batch
