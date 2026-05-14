"""Sample N truth paths and render a world-marginals + correlation report."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from solarbess.config import AppConfig
from solarbess.reporting.world_validation import render_world_validation_report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--n-days", type=int, default=200)
    ap.add_argument("--base-seed", type=int, default=10_000)
    ap.add_argument("--start-date", default="2026-05-01")
    ap.add_argument("--out", default="reports/world_validation.html")
    args = ap.parse_args()
    cfg = AppConfig.from_yaml(args.config)
    out = render_world_validation_report(
        cfg=cfg,
        n_days=args.n_days,
        base_seed=args.base_seed,
        out_path=Path(args.out),
        start_date=date.fromisoformat(args.start_date),
    )
    print(f"Report: {out}")


if __name__ == "__main__":
    main()
