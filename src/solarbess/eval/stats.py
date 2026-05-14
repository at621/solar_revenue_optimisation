"""Significance tests + empirical CVaR."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.stats as stats


@dataclass(frozen=True)
class ComparisonResult:
    label_a: str
    label_b: str
    mean_diff: float
    median_diff: float
    paired_t_statistic: float
    paired_t_pvalue: float
    wilcoxon_statistic: float
    wilcoxon_pvalue: float
    n: int


def paired_t_test(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    res = stats.ttest_rel(a, b)
    return float(res.statistic), float(res.pvalue)


def wilcoxon_signed_rank(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    diff = np.asarray(a) - np.asarray(b)
    if np.all(diff == 0):
        return 0.0, 1.0
    res = stats.wilcoxon(diff, zero_method="wilcox")
    return float(res.statistic), float(res.pvalue)


def compare(label_a: str, a: np.ndarray, label_b: str, b: np.ndarray) -> ComparisonResult:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("samples must align")
    t_stat, t_p = paired_t_test(a, b)
    w_stat, w_p = wilcoxon_signed_rank(a, b)
    return ComparisonResult(
        label_a=label_a,
        label_b=label_b,
        mean_diff=float(np.mean(a - b)),
        median_diff=float(np.median(a - b)),
        paired_t_statistic=t_stat,
        paired_t_pvalue=t_p,
        wilcoxon_statistic=w_stat,
        wilcoxon_pvalue=w_p,
        n=len(a),
    )


def empirical_cvar(profits: np.ndarray, alpha: float = 0.95) -> float:
    """Empirical CVaR of the LOSS distribution. losses = -profits.

    CVaR_alpha = mean of the worst (1-alpha) fraction of losses.
    Higher CVaR = worse downside risk.
    """
    losses = -np.asarray(profits, dtype=float)
    if losses.size == 0:
        return float("nan")
    quantile = np.quantile(losses, alpha)
    tail = losses[losses >= quantile]
    if tail.size == 0:
        return float(quantile)
    return float(tail.mean())
