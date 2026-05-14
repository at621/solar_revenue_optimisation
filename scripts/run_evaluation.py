"""Run N paired-seed episodes across all policies; write comparison report.

Usage:
    uv run python scripts/run_evaluation.py --n-episodes 50 --base-seed 100
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from solarbess.config import AppConfig
from solarbess.eval.harness import run_paired_evaluation
from solarbess.policy.oracle_policy import OraclePolicy
from solarbess.policy.pyomo_policy import PyomoPolicy
from solarbess.policy.rule_based import RuleBasedPolicy
from solarbess.reporting.comparison_report import render_comparison_report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--n-episodes", type=int, default=10)
    ap.add_argument("--base-seed", type=int, default=100)
    ap.add_argument("--start-date", default="2026-06-01")
    ap.add_argument("--out", default="reports/comparison.html")
    ap.add_argument("--include-stochastic", action="store_true", default=True)
    ap.add_argument("--include-cvar", action="store_true", default=False)
    args = ap.parse_args()

    cfg = AppConfig.from_yaml(args.config)
    factories = {
        "rule_based": lambda _t: RuleBasedPolicy(cfg.battery, cfg.market, cfg.dt_hours),
        "pyomo_deterministic": lambda _t: PyomoPolicy(
            cfg.battery, cfg.market, cfg.dt_hours, use_stochastic=False
        ),
        "oracle": lambda t: OraclePolicy(cfg.battery, cfg.market, cfg.dt_hours, t),
    }
    if args.include_stochastic:
        factories["pyomo_stochastic"] = lambda _t: PyomoPolicy(
            cfg.battery, cfg.market, cfg.dt_hours, use_stochastic=True
        )
    if args.include_cvar:
        factories["pyomo_cvar"] = lambda _t: PyomoPolicy(
            cfg.battery,
            cfg.market,
            cfg.dt_hours,
            use_stochastic=True,
            lambda_cvar=0.5,
            cvar_alpha=cfg.cvar.alpha,
        )

    start = date.fromisoformat(args.start_date)
    report = run_paired_evaluation(
        policy_factories=factories,
        cfg=cfg,
        n_episodes=args.n_episodes,
        base_seed=args.base_seed,
        start_date=start,
        cvar_alpha=cfg.cvar.alpha,
        progress=True,
    )
    print()
    print(report.summary_table().to_string(index=False))
    print()
    for c in report.comparisons:
        print(
            f"  {c.label_a} vs {c.label_b}: mean diff {c.mean_diff:>8.1f}  "
            f"t-p {c.paired_t_pvalue:.4g}  Wilcoxon-p {c.wilcoxon_pvalue:.4g}"
        )
    out = render_comparison_report(report, Path(args.out))
    print(f"\nReport: {out}")


if __name__ == "__main__":
    main()
