"""World validation: marginals + solar-DA correlation across N truth samples."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from solarbess.clock import DSTTransitionError
from solarbess.config import AppConfig
from solarbess.world.truth import TruthGenerator


def render_world_validation_report(
    cfg: AppConfig, n_days: int, base_seed: int, out_path: Path, start_date: date | None = None
) -> Path:
    start = start_date or date(2026, 5, 1)
    tg = TruthGenerator(cfg)
    solars: list[np.ndarray] = []
    das: list[np.ndarray] = []
    imb_surplus: list[np.ndarray] = []
    imb_short: list[np.ndarray] = []
    k = 0
    collected = 0
    while collected < n_days and k < n_days * 4:
        try:
            t = tg.sample(start + timedelta(days=k), seed=base_seed + k)
            solars.append(t.solar_mw)
            das.append(t.da_price_mtu)
            imb_surplus.append(t.imb_surplus_price)
            imb_short.append(t.imb_shortfall_price)
            collected += 1
        except DSTTransitionError:
            pass
        k += 1
    solar_arr = np.stack(solars)
    da_arr = np.stack(das)

    mtus_per_hour = cfg.mtus_per_hour
    midday = slice(11 * mtus_per_hour, 14 * mtus_per_hour)
    midday_solar = solar_arr[:, midday].mean(axis=1)
    midday_da = da_arr[:, midday].mean(axis=1)
    corr = float(np.corrcoef(midday_solar, midday_da)[0, 1])

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            f"Solar MW histogram (all MTUs, n={collected} days)",
            f"DA price histogram (n={collected})",
            f"Mean diurnal shape: solar (MW) and DA price",
            f"Midday solar vs DA (corr={corr:.3f})",
        ),
    )
    fig.add_trace(go.Histogram(x=solar_arr.flatten(), nbinsx=60, name="solar"), row=1, col=1)
    fig.add_trace(go.Histogram(x=da_arr.flatten(), nbinsx=60, name="DA"), row=1, col=2)
    fig.add_trace(go.Scatter(y=solar_arr.mean(axis=0), name="solar mean MW"), row=2, col=1)
    fig.add_trace(
        go.Scatter(y=da_arr.mean(axis=0), name="DA mean EUR/MWh", yaxis="y2"),
        row=2, col=1,
    )
    fig.add_trace(
        go.Scatter(x=midday_solar, y=midday_da, mode="markers", name="midday"), row=2, col=2
    )

    fig.update_layout(height=1100, title=f"World validation - {collected} days from {start}")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_path)
    return out_path
