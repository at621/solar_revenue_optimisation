"""Paired-seed N-episode evaluation harness.

Variance reduction: same world seed used for every policy in an episode index,
so all policies face the same truth. The forecast-noise seed is also shared,
so all policies see identical InfoSets (only the policy logic varies).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Callable

import numpy as np
import pandas as pd

from solarbess.config import AppConfig
from solarbess.clock import DSTTransitionError
from solarbess.eval.stats import ComparisonResult, compare, empirical_cvar
from solarbess.forecast.scenarios import ForecastGenerator
from solarbess.rng import derive_seeds
from solarbess.simulator.episode import EpisodeResult, run_episode
from solarbess.world.truth import TruthGenerator, TruthPath

PolicyFactory = Callable[[TruthPath], "object"]


@dataclass
class EvaluationReport:
    results_by_policy: dict[str, list[EpisodeResult]]
    comparisons: list[ComparisonResult]
    cvar_by_policy: dict[str, float] = field(default_factory=dict)
    cvar_alpha: float = 0.95

    def profit_array(self, policy: str) -> np.ndarray:
        return np.array([r.pnl.total for r in self.results_by_policy[policy]])

    def summary_table(self) -> pd.DataFrame:
        rows = []
        for policy, runs in self.results_by_policy.items():
            profits = self.profit_array(policy)
            rows.append({
                "policy": policy,
                "n_episodes": len(profits),
                "mean_profit_eur": float(profits.mean()),
                "median_profit_eur": float(np.median(profits)),
                "std_profit_eur": float(profits.std(ddof=1)) if len(profits) > 1 else 0.0,
                f"cvar_{int(self.cvar_alpha * 100)}_loss_eur": self.cvar_by_policy.get(policy, float("nan")),
            })
        return pd.DataFrame(rows)


def run_paired_evaluation(
    policy_factories: dict[str, PolicyFactory],
    cfg: AppConfig,
    n_episodes: int,
    base_seed: int,
    start_date: date,
    skip_dst: bool = True,
    cvar_alpha: float = 0.95,
    progress: bool = False,
) -> EvaluationReport:
    truth_gen = TruthGenerator(cfg)
    fc_gen = ForecastGenerator(cfg.forecast)
    results: dict[str, list[EpisodeResult]] = {name: [] for name in policy_factories}

    completed = 0
    k = 0
    while completed < n_episodes and k < n_episodes * 4:
        seeds = derive_seeds(base_seed, k)
        d = start_date + timedelta(days=k)
        try:
            truth = truth_gen.sample(d, seed=seeds.world)
        except DSTTransitionError:
            if skip_dst:
                k += 1
                continue
            raise
        for name, factory in policy_factories.items():
            policy = factory(truth)
            r = run_episode(truth, cfg, policy, fc_gen, forecast_seed=seeds.forecast)
            results[name].append(r)
        completed += 1
        if progress:
            print(f"[eval] {completed}/{n_episodes}: {d}", flush=True)
        k += 1

    # Pairwise comparisons (all unordered pairs)
    comparisons: list[ComparisonResult] = []
    names = list(policy_factories.keys())
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            profits_a = np.array([r.pnl.total for r in results[a]])
            profits_b = np.array([r.pnl.total for r in results[b]])
            comparisons.append(compare(a, profits_a, b, profits_b))

    cvar_by_policy = {
        name: empirical_cvar(np.array([r.pnl.total for r in runs]), alpha=cvar_alpha)
        for name, runs in results.items()
    }
    return EvaluationReport(
        results_by_policy=results,
        comparisons=comparisons,
        cvar_by_policy=cvar_by_policy,
        cvar_alpha=cvar_alpha,
    )
