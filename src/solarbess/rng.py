"""Seed plumbing for reproducible, variance-reducing paired runs.

The harness uses three logical seed streams per episode:
- world seed: drives TruthGenerator
- forecast seed: drives ForecastGenerator (same across policies for variance reduction)
- policy seed: drives any policy-internal randomness (e.g. tie-breaking)

`derive_seeds(base_seed, episode_idx)` produces a stable triple.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SeedTriple:
    world: int
    forecast: int
    policy: int


def derive_seeds(base_seed: int, episode_idx: int) -> SeedTriple:
    """Spawn three independent seeds from a base seed and episode index.

    Uses numpy's SeedSequence to guarantee statistically independent streams.
    """
    ss = np.random.SeedSequence([int(base_seed), int(episode_idx)])
    a, b, c = ss.spawn(3)
    return SeedTriple(
        world=int(a.generate_state(1)[0]),
        forecast=int(b.generate_state(1)[0]),
        policy=int(c.generate_state(1)[0]),
    )


def rng_from_seed(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)
