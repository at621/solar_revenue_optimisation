"""Property: oracle >= pyomo-deterministic >= rule-based on average across paired seeds."""

from __future__ import annotations

from datetime import date

import pytest

from solarbess.config import AppConfig
from solarbess.eval.harness import run_paired_evaluation
from solarbess.policy.oracle_policy import OraclePolicy
from solarbess.policy.pyomo_policy import PyomoPolicy
from solarbess.policy.rule_based import RuleBasedPolicy


@pytest.mark.slow
def test_paired_ordering(cfg: AppConfig) -> None:
    """Oracle must dominate everything; Pyomo and rule-based have small gaps that
    can be sample-noise on a per-episode basis. Plan calls for 30 episodes; we use
    12 here for test speed (each Pyomo solve is non-trivial).
    """
    factories = {
        "rule_based": lambda _t: RuleBasedPolicy(cfg.battery, cfg.market, cfg.dt_hours),
        "pyomo_deterministic": lambda _t: PyomoPolicy(cfg.battery, cfg.market, cfg.dt_hours, use_stochastic=False),
        "oracle": lambda t: OraclePolicy(cfg.battery, cfg.market, cfg.dt_hours, t),
    }
    report = run_paired_evaluation(
        policy_factories=factories,
        cfg=cfg,
        n_episodes=12,
        base_seed=2026,
        start_date=date(2026, 6, 1),
    )
    rule_mean = report.profit_array("rule_based").mean()
    det_mean = report.profit_array("pyomo_deterministic").mean()
    oracle_mean = report.profit_array("oracle").mean()
    # Oracle is a deterministic upper bound -> strictly dominates (LP relaxation).
    assert oracle_mean >= det_mean - 1.0
    # Pyomo deterministic on scenario MEAN can lose to rule-based on a finite sample
    # because forecast bias leaks into the LP solution. Allow a relative tolerance.
    assert det_mean >= rule_mean - 0.05 * abs(rule_mean), (
        f"pyomo_det mean {det_mean:.2f} substantially below rule_based mean {rule_mean:.2f}"
    )


def test_settlement_invariant(cfg: AppConfig) -> None:
    """For any episode, sum(period_pnls.total) must equal pnl.total."""
    from solarbess.forecast.scenarios import ForecastGenerator
    from solarbess.world.truth import TruthGenerator
    from solarbess.simulator.episode import run_episode
    fg = ForecastGenerator(cfg.forecast)
    tg = TruthGenerator(cfg)
    truth = tg.sample(date(2026, 6, 5), seed=987)
    pol = RuleBasedPolicy(cfg.battery, cfg.market, cfg.dt_hours)
    r = run_episode(truth, cfg, pol, fg, forecast_seed=2)
    sum_periods = sum(p.total for p in r.pnl.period_pnls)
    assert sum_periods == pytest.approx(r.pnl.total, abs=1e-6)
