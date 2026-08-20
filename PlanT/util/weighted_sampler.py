"""Weighted sampling of frames, split across DDP ranks.

PlanTDataset expands every route into one sample per frame, and the loader
draws them uniformly. For a speed-limit sign that is the wrong distribution:
the braking transient the scene was built around lasts 7-10 frames out of
several hundred, so ~6% of the frames carry the sign->speed signal and the
rest teach that speed does not depend on the sign. Detour scenes do not have
this problem (the cones are visible in 95% of frames), which is why the weight
is per sample rather than per family.
"""
from __future__ import annotations

import torch
from torch.utils.data import Sampler


class DistributedWeightedSampler(Sampler):
    """Draw indices with replacement, proportional to per-sample weights.

    Every rank draws the SAME multinomial sample from the SAME generator and
    then keeps its own stride of it. That keeps the ranks disjoint without any
    communication, and makes an epoch reproducible from (seed, epoch) alone.
    Torch's own WeightedRandomSampler is not usable here: Lightning wraps a
    plain sampler in a DistributedSampler, which would shard the *weights*
    rather than the draw, and every rank would sample from a different pool.
    """

    def __init__(self, weights, num_replicas: int = 1, rank: int = 0,
                 num_samples: int | None = None, seed: int = 0):
        if num_replicas < 1:
            raise ValueError(f"num_replicas must be >= 1, got {num_replicas}")
        if not 0 <= rank < num_replicas:
            raise ValueError(f"rank {rank} out of range for {num_replicas} replicas")

        self.weights = torch.as_tensor(weights, dtype=torch.double)
        if self.weights.ndim != 1 or self.weights.numel() == 0:
            raise ValueError("weights must be a non-empty 1-D sequence")
        if torch.any(self.weights < 0) or float(self.weights.sum()) <= 0:
            raise ValueError("weights must be non-negative with a positive sum")

        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.epoch = 0

        n = int(num_samples if num_samples is not None else self.weights.numel())
        # Round up so every rank gets the same count: an uneven split makes DDP
        # hang at the end of an epoch, waiting for a rank that already stopped.
        self.num_samples = -(-n // self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        idx = torch.multinomial(self.weights, self.total_size, replacement=True, generator=g)
        return iter(idx[self.rank::self.num_replicas].tolist())

    def __len__(self) -> int:
        # Per-rank length. Callers sizing an LR schedule must NOT divide this by
        # world_size again -- unlike the DistributedSampler Lightning attaches
        # on its own, this one is already sharded.
        return self.num_samples
