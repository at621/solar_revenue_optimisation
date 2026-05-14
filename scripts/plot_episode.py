"""Convenience: render every policy for a single seed/day into individual HTML files."""

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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--date", default="2026-06-01")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="reports/episodes")
    args = ap.parse_args()
    cfg = AppConfig.from_yaml(args.config)
    tg = TruthGenerator(cfg)
    fg = ForecastGenerator(cfg.forecast)
    delivery_date = date.fromisoformat(args.date)
    seeds = derive_seeds(args.seed, 0)
    truth = tg.sample(delivery_date, seed=seeds.world)
    policies = {
        "rule_based": RuleBasedPolicy(cfg.battery, cfg.market, cfg.dt_hours),
        "pyomo_deterministic": PyomoPolicy(cfg.battery, cfg.market, cfg.dt_hours, use_stochastic=False),
        "pyomo_stochastic": PyomoPolicy(cfg.battery, cfg.market, cfg.dt_hours, use_stochastic=True),
        "oracle": OraclePolicy(cfg.battery, cfg.market, cfg.dt_hours, truth),
    }
    out_dir = Path(args.out_dir)
    for name, policy in policies.items():
        r = run_episode(truth, cfg, policy, fg, forecast_seed=seeds.forecast)
        out = render_episode_report(r, out_dir / f"{name}.html")
        print(f"{name:<22} PnL={r.pnl.total:>10.2f}  {out}")


if __name__ == "__main__":
    main()
