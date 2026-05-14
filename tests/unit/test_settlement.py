"""Settlement: hand-calc one period of surplus, shortfall, and balanced.

This is the Phase-1 acceptance gate per the PDF.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from solarbess.config import BatteryParams, MarketParams
from solarbess.market.interface import Position
from solarbess.market.settlement import ProfitAccountant
from solarbess.world.truth import TruthPath


@pytest.fixture
def battery() -> BatteryParams:
    return BatteryParams(
        capacity_mwh=10.0,
        p_charge_max_mw=2.0,
        p_discharge_max_mw=2.0,
        eta_charge=0.9,
        eta_discharge=0.9,
        soc_min_mwh=0.5,
        soc_max_mwh=9.5,
        soc_initial_mwh=5.0,
        soc_terminal_min_mwh=1.0,
        degradation_cost_eur_per_mwh=4.0,
    )


@pytest.fixture
def market() -> MarketParams:
    return MarketParams(export_limit_mw=20.0, import_limit_mw=0.0, fees_eur_per_mwh=0.1)


def _toy_truth() -> TruthPath:
    # 4 MTUs of 15 min each = 1 h
    ts = pd.date_range("2026-06-01T00:00:00Z", periods=4, freq="15min")
    return TruthPath(
        delivery_date=date(2026, 6, 1),
        timestamps=ts,
        solar_mw=np.array([10.0, 10.0, 10.0, 10.0]),
        da_price_hourly=np.array([50.0]),
        da_price_mtu=np.array([50.0, 50.0, 50.0, 50.0]),
        id_price_mtu=np.array([[48.0], [48.0], [48.0], [48.0]]),
        imb_surplus_price=np.array([40.0, 40.0, 40.0, 40.0]),
        imb_shortfall_price=np.array([80.0, 80.0, 80.0, 80.0]),
        latent={},
    )


def test_balanced_period_pnl(battery: BatteryParams, market: MarketParams) -> None:
    truth = _toy_truth()
    pa = ProfitAccountant(market, battery, dt_hours=0.25)
    pos = Position(n_mtu=4, n_rounds=1)
    pos.set_day_ahead(np.array([5.0, 5.0, 5.0, 5.0]))
    # physical exactly matches schedule -> no imbalance
    pnl = pa.settle_period(period_idx=0, position=pos, physical_mw=5.0, truth=truth,
                            charge_mw=0.0, discharge_mw=0.0, realized_solar_mw=10.0)
    # da_rev = 5 * 50 * 0.25 = 62.5
    assert pnl.da_revenue == pytest.approx(62.5)
    assert pnl.id_revenue == pytest.approx(0.0)
    assert pnl.imbalance_revenue == pytest.approx(0.0)
    assert pnl.degradation_cost == pytest.approx(0.0)
    assert pnl.fees == pytest.approx(0.1 * 5 * 0.25)  # 0.125


def test_surplus_period_pnl(battery: BatteryParams, market: MarketParams) -> None:
    truth = _toy_truth()
    pa = ProfitAccountant(market, battery, dt_hours=0.25)
    pos = Position(n_mtu=4, n_rounds=1)
    pos.set_day_ahead(np.array([3.0, 0.0, 0.0, 0.0]))
    # physical = 6 (more than committed 3) -> surplus of 3 MW * 0.25 h = 0.75 MWh @ 40 EUR
    pnl = pa.settle_period(period_idx=0, position=pos, physical_mw=6.0, truth=truth,
                            charge_mw=0.0, discharge_mw=0.0, realized_solar_mw=6.0)
    assert pnl.da_revenue == pytest.approx(3.0 * 50.0 * 0.25)
    assert pnl.imbalance_revenue == pytest.approx(40.0 * 0.75)
    assert pnl.imb_pos_mwh == pytest.approx(0.75)
    assert pnl.imb_neg_mwh == pytest.approx(0.0)


def test_shortfall_period_pnl(battery: BatteryParams, market: MarketParams) -> None:
    truth = _toy_truth()
    pa = ProfitAccountant(market, battery, dt_hours=0.25)
    pos = Position(n_mtu=4, n_rounds=1)
    pos.set_day_ahead(np.array([8.0, 0.0, 0.0, 0.0]))
    pnl = pa.settle_period(period_idx=0, position=pos, physical_mw=5.0, truth=truth,
                            charge_mw=0.0, discharge_mw=0.0, realized_solar_mw=5.0)
    # shortfall 3 MW * 0.25 h = 0.75 MWh @ 80 EUR
    assert pnl.imbalance_revenue == pytest.approx(-80.0 * 0.75)
    assert pnl.imb_neg_mwh == pytest.approx(0.75)


def test_intraday_revenue_signs(battery: BatteryParams, market: MarketParams) -> None:
    truth = _toy_truth()
    pa = ProfitAccountant(market, battery, dt_hours=0.25)
    pos = Position(n_mtu=4, n_rounds=1)
    pos.set_day_ahead(np.array([3.0, 0.0, 0.0, 0.0]))
    pos.add_intraday(np.array([2.0, 0.0, 0.0, 0.0]), 0)  # sold extra 2 MW intraday @ 48
    pnl = pa.settle_period(period_idx=0, position=pos, physical_mw=5.0, truth=truth,
                            charge_mw=0.0, discharge_mw=0.0, realized_solar_mw=5.0)
    assert pnl.id_revenue == pytest.approx(2.0 * 48.0 * 0.25)
    assert pnl.imbalance_revenue == pytest.approx(0.0)
