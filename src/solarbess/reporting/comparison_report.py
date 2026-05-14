"""Cross-policy comparison HTML report."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from solarbess.eval.harness import EvaluationReport


def render_comparison_report(report: EvaluationReport, out_path: Path) -> Path:
    summary = report.summary_table()
    profits_by_policy = {
        name: np.array([r.pnl.total for r in runs])
        for name, runs in report.results_by_policy.items()
    }

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Profit distribution (per policy)",
            "Profit time series (paired episodes)",
            "Summary table",
            "Comparisons (paired tests)",
        ),
        specs=[
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "table"}, {"type": "table"}],
        ],
        vertical_spacing=0.12,
    )

    for name, profits in profits_by_policy.items():
        fig.add_trace(go.Box(y=profits, name=name, boxmean=True), row=1, col=1)
        fig.add_trace(
            go.Scatter(y=profits, mode="lines+markers", name=name), row=1, col=2
        )

    fig.add_trace(
        go.Table(
            header=dict(values=list(summary.columns)),
            cells=dict(
                values=[summary[col].astype(str).tolist() for col in summary.columns]
            ),
        ),
        row=2,
        col=1,
    )

    if report.comparisons:
        rows = [
            [
                c.label_a,
                c.label_b,
                f"{c.mean_diff:.2f}",
                f"{c.median_diff:.2f}",
                f"{c.paired_t_pvalue:.4g}",
                f"{c.wilcoxon_pvalue:.4g}",
                str(c.n),
            ]
            for c in report.comparisons
        ]
        fig.add_trace(
            go.Table(
                header=dict(
                    values=["a", "b", "mean(a-b)", "median(a-b)", "t p", "Wilcoxon p", "n"]
                ),
                cells=dict(values=list(zip(*rows, strict=False))),
            ),
            row=2,
            col=2,
        )

    fig.update_layout(height=1400, title="Policy comparison report")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_path)
    return out_path
