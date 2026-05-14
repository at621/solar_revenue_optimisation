"""Per-episode HTML report: PnL stack, position vs physical, SoC, price overlays."""

from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from solarbess.simulator.episode import EpisodeResult


def render_episode_report(result: EpisodeResult, out_path: Path) -> Path:
    df = result.timeseries
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        subplot_titles=(
            "Schedule vs Physical (MW)",
            "Realized solar / Battery dispatch (MW)",
            "State of Charge (MWh)",
            "Prices (EUR/MWh)",
        ),
        vertical_spacing=0.05,
        row_heights=[0.28, 0.24, 0.18, 0.30],
    )
    fig.add_trace(go.Scatter(x=df.timestamp, y=df.committed_mw, name="committed"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.timestamp, y=df.physical_mw, name="physical"), row=1, col=1)

    fig.add_trace(go.Scatter(x=df.timestamp, y=df.realized_solar_mw, name="solar"), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.timestamp, y=df.charge_mw, name="charge"), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.timestamp, y=df.discharge_mw, name="discharge"), row=2, col=1)

    fig.add_trace(go.Scatter(x=df.timestamp, y=df.soc_mwh, name="soc"), row=3, col=1)

    fig.add_trace(go.Scatter(x=df.timestamp, y=df.da_price, name="DA price"), row=4, col=1)
    fig.add_trace(
        go.Scatter(x=df.timestamp, y=df.imb_surplus_price, name="imb surplus"), row=4, col=1
    )
    fig.add_trace(
        go.Scatter(x=df.timestamp, y=df.imb_shortfall_price, name="imb shortfall"), row=4, col=1
    )

    pnl = result.pnl
    title = (
        f"Episode {result.delivery_date} | policy={result.policy_name} | "
        f"Total PnL: {pnl.total:.1f} EUR | "
        f"DA={pnl.da_revenue:.0f} ID={pnl.id_revenue:.0f} Imb={pnl.imbalance_revenue:.0f} "
        f"Degr=-{pnl.degradation_cost:.0f} Fees=-{pnl.fees:.0f}"
    )
    fig.update_layout(height=1200, title=title, hovermode="x unified")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_path)
    return out_path
