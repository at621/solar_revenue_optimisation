"""End-to-end episode runs for each policy."""

from __future__ import annotations

import math

import numpy as np

from solarbess.config import AppConfig
from solarbess.forecast.scenarios import ForecastGenerator
from solarbess.policy.oracle_policy import OraclePolicy
from solarbess.policy.pyomo_policy import PyomoPolicy
from solarbess.policy.rule_based import RuleBasedPolicy
from solarbess.simulator.episode import run_episode


def test_rule_based_completes(cfg: AppConfig, truth) -> None:
    fg = ForecastGenerator(cfg.forecast)
    pol = RuleBasedPolicy(cfg.battery, cfg.market, cfg.dt_hours)
    r = run_episode(truth, cfg, pol, fg, forecast_seed=1)
    assert math.isfinite(r.pnl.total)
    # SoC stays inside bounds
    assert (r.timeseries.soc_mwh >= cfg.battery.soc_min_mwh - 1e-6).all()
    assert (r.timeseries.soc_mwh <= cfg.battery.soc_max_mwh + 1e-6).all()


def test_pyomo_deterministic_completes(cfg: AppConfig, truth) -> None:
    fg = ForecastGenerator(cfg.forecast)
    pol = PyomoPolicy(cfg.battery, cfg.market, cfg.dt_hours, use_stochastic=False)
    r = run_episode(truth, cfg, pol, fg, forecast_seed=1)
    assert math.isfinite(r.pnl.total)


def test_oracle_completes_and_dominates_rule(cfg: AppConfig, truth) -> None:
    fg = ForecastGenerator(cfg.forecast)
    rule = RuleBasedPolicy(cfg.battery, cfg.market, cfg.dt_hours)
    oracle = OraclePolicy(cfg.battery, cfg.market, cfg.dt_hours, truth)
    r_rule = run_episode(truth, cfg, rule, fg, forecast_seed=1)
    r_oracle = run_episode(truth, cfg, oracle, fg, forecast_seed=1)
    # Oracle uses TruthPath; given LP relaxation, must score >= rule-based.
    assert r_oracle.pnl.total >= r_rule.pnl.total - 1e-3
