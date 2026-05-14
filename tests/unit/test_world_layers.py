"""World L2/L3 coupling: midday solar negatively correlated with midday DA price."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pytest

from solarbess.clock import DSTTransitionError
from solarbess.config import AppConfig
from solarbess.world.truth import TruthGenerator


@pytest.mark.slow
def test_midday_solar_da_correlation_negative(cfg: AppConfig) -> None:
    tg = TruthGenerator(cfg)
    solars: list[float] = []
    prices: list[float] = []
    collected = 0
    k = 0
    while collected < 150 and k < 600:
        try:
            t = tg.sample(date(2026, 5, 1) + timedelta(days=k), seed=20_000 + k)
            # Clock.mtu_index() starts at local midnight (= prior day 21:00 UTC).
            # So MTU 48 = +12h = local noon = solar peak.
            # Peak production window in MTU index: 48-60 = local 12:00-15:00.
            mid = slice(48, 60)
            solars.append(t.solar_mw[mid].mean())
            prices.append(t.da_price_mtu[mid].mean())
            collected += 1
        except DSTTransitionError:
            pass
        k += 1
    corr = float(np.corrcoef(np.array(solars), np.array(prices))[0, 1])
    # The PDF design target is -0.3 at the population level; with 150 samples
    # and Student-t shocks the estimate has ~+/-0.07 noise. -0.20 is the
    # sample-robust threshold that still proves the channel works.
    assert corr < -0.20, f"midday solar-DA correlation too weak: {corr:.3f}"


def test_truth_shapes(truth) -> None:
    n = len(truth.timestamps)
    assert truth.solar_mw.shape == (n,)
    assert truth.da_price_mtu.shape == (n,)
    assert truth.id_price_mtu.ndim == 2 and truth.id_price_mtu.shape[0] == n
    assert truth.imb_surplus_price.shape == (n,)
    assert truth.imb_shortfall_price.shape == (n,)
    assert (truth.imb_shortfall_price >= truth.imb_surplus_price).all()


def test_solar_non_negative(truth) -> None:
    assert (truth.solar_mw >= 0).all()
