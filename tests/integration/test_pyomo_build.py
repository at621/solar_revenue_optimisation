"""Pyomo builders + HiGHS solve smoke test."""

from __future__ import annotations

import numpy as np
import pandas as pd

from solarbess.config import BatteryParams, MarketParams
from solarbess.forecast.scenarios import ScenarioForecast
from solarbess.optim.builder import build_deterministic_model, build_stochastic_model
from solarbess.optim.solve import solve_highs
from solarbess.world.oracle import DecisionTime


def test_deterministic_solves() -> None:
    times = pd.date_range("2026-06-01T00:00:00Z", periods=4, freq="15min")
    bp = BatteryParams(
        capacity_mwh=10, p_charge_max_mw=2, p_discharge_max_mw=2,
        eta_charge=0.95, eta_discharge=0.95,
        soc_min_mwh=1, soc_max_mwh=9, soc_initial_mwh=5,
        soc_terminal_min_mwh=1, degradation_cost_eur_per_mwh=2,
    )
    mp = MarketParams(export_limit_mw=20, import_limit_mw=0)
    m = build_deterministic_model(
        times=times,
        solar_mw=np.array([0.0, 5.0, 5.0, 0.0]),
        da_price_mtu=np.array([40.0, 30.0, 80.0, 60.0]),
        id_price_mtu=np.array([40.0, 30.0, 80.0, 60.0]),
        imb_surplus=np.array([20.0, 15.0, 60.0, 40.0]),
        imb_shortfall=np.array([60.0, 50.0, 100.0, 80.0]),
        battery=bp,
        market=mp,
        dt_hours=0.25,
        soc_initial_mwh=5.0,
    )
    res = solve_highs(m, time_limit_s=5.0)
    assert np.isfinite(res.objective)


def test_stochastic_two_scenarios() -> None:
    times = pd.date_range("2026-06-01T00:00:00Z", periods=4, freq="15min")
    bp = BatteryParams(
        capacity_mwh=10, p_charge_max_mw=2, p_discharge_max_mw=2,
        eta_charge=0.95, eta_discharge=0.95,
        soc_min_mwh=1, soc_max_mwh=9, soc_initial_mwh=5,
        soc_terminal_min_mwh=1, degradation_cost_eur_per_mwh=2,
    )
    mp = MarketParams(export_limit_mw=20, import_limit_mw=0)
    scen = ScenarioForecast(
        scenario_ids=np.array([0, 1]),
        probabilities=np.array([0.5, 0.5]),
        timestamps=times,
        solar_mw=np.array([[0, 5, 5, 0], [0, 3, 3, 0]], dtype=float),
        da_price=np.array([[40, 30, 80, 60], [50, 40, 90, 70]], dtype=float),
        id_price=np.array([[40, 30, 80, 60], [50, 40, 90, 70]], dtype=float),
        imb_surplus=np.array([[20, 15, 60, 40], [25, 20, 70, 50]], dtype=float),
        imb_shortfall=np.array([[60, 50, 100, 80], [70, 60, 110, 90]], dtype=float),
        decision_time=DecisionTime.DA_GATE,
    )
    m = build_stochastic_model(
        times=times, scenarios=scen, battery=bp, market=mp,
        dt_hours=0.25, soc_initial_mwh=5.0,
    )
    res = solve_highs(m, time_limit_s=5.0)
    assert np.isfinite(res.objective)
