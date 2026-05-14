"""Run a single episode with one policy and write an HTML report.

Usage:
    uv run python scripts/run_episode.py --policy rule_based --seed 42
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from solarbess.config import AppConfig
from solarbess.forecast.scenarios import ForecastGenerator
from solarbess.policy.oracle_policy import OraclePolicy
from solarbess.policy.pyomo_policy import PyomoPolicy
from solarbess.policy.rule_based import RuleBasedPolicy
from solarbess.reporting.episode_report import render_episode_report
from solarbess.rng import derive_seeds
from solarbess.simulator.episode import run_episode
from solarbess.world.truth import TruthGenerator


def build_policy(name: str, cfg: AppConfig, truth):
    if name == "rule_based":
        return RuleBasedPolicy(cfg.battery, cfg.market, cfg.dt_hours)
    if name == "pyomo_deterministic":
        return PyomoPolicy(cfg.battery, cfg.market, cfg.dt_hours, use_stochastic=False)
    if name == "pyomo_stochastic":
        return PyomoPolicy(cfg.battery, cfg.market, cfg.dt_hours, use_stochastic=True)
    if name == "pyomo_cvar":
        return PyomoPolicy(
            cfg.battery,
            cfg.market,
            cfg.dt_hours,
            use_stochastic=True,
            lambda_cvar=cfg.cvar.lambda_cvar if cfg.cvar.lambda_cvar > 0 else 0.5,
            cvar_alpha=cfg.cvar.alpha,
            lambda_imbalance=cfg.cvar.lambda_imbalance,
        )
    if name == "oracle":
        return OraclePolicy(cfg.battery, cfg.market, cfg.dt_hours, truth)
    raise SystemExit(f"unknown policy {name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--policy", default="rule_based",
                    choices=["rule_based", "pyomo_deterministic", "pyomo_stochastic", "pyomo_cvar", "oracle"])
    ap.add_argument("--date", default="2026-06-01")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="reports/episode.html")
    args = ap.parse_args()

    cfg = AppConfig.from_yaml(args.config)
    tg = TruthGenerator(cfg)
    fg = ForecastGenerator(cfg.forecast)
    delivery_date = date.fromisoformat(args.date)
    seeds = derive_seeds(args.seed, 0)
    truth = tg.sample(delivery_date, seed=seeds.world)
    policy = build_policy(args.policy, cfg, truth)
    result = run_episode(truth, cfg, policy, fg, forecast_seed=seeds.forecast)
    out = render_episode_report(result, Path(args.out))
    pnl = result.pnl
    print(f"Policy: {result.policy_name}")
    print(f"Date:   {result.delivery_date}")
    print(f"PnL:    {pnl.total:>10.2f} EUR  (DA {pnl.da_revenue:.0f} ID {pnl.id_revenue:.0f} "
          f"Imb {pnl.imbalance_revenue:.0f} Degr -{pnl.degradation_cost:.0f} Fees -{pnl.fees:.0f})")
    print(f"Report: {out}")


if __name__ == "__main__":
    main()
