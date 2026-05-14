"""Phase 4: MILP no-simultaneous-charge/discharge enforcement."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyomo.environ as pyo

from solarbess.config import BatteryParams, MarketParams
from solarbess.forecast.scenarios import ForecastGenerator
from solarbess.optim.builder import build_deterministic_model
from solarbess.optim.solve import solve_highs
from solarbess.policy.pyomo_policy import PyomoPolicy
from solarbess.simulator.episode import run_episode


def test_milp_solves_small_case() -> None:
    """Toy MILP solves to optimality and respects the binary constraint."""
    times = pd.date_range("2026-06-01T00:00:00Z", periods=4, freq="15min")
    bp = BatteryParams(
        capacity_mwh=10, p_charge_max_mw=2, p_discharge_max_mw=2,
        eta_charge=0.95, eta_discharge=0.95,
        soc_min_mwh=1, soc_max_mwh=9, soc_initial_mwh=5,
        soc_terminal_min_mwh=1, degradation_cost_eur_per_mwh=2.0,
    )
    mp = MarketParams(export_limit_mw=20, import_limit_mw=0)
    m = build_deterministic_model(
        times=times,
        solar_mw=np.array([0.0, 5.0, 5.0, 0.0]),
        da_price_mtu=np.array([40.0, 30.0, 80.0, 60.0]),
        id_price_mtu=np.array([40.0, 30.0, 80.0, 60.0]),
        imb_surplus=np.array([20.0, 15.0, 60.0, 40.0]),
        imb_shortfall=np.array([60.0, 50.0, 100.0, 80.0]),
        battery=bp, market=mp, dt_hours=0.25, soc_initial_mwh=5.0,
        binary_battery_mode=True,
    )
    res = solve_highs(m, time_limit_s=5.0)
    assert np.isfinite(res.objective)
    # y_charge is binary
    for t in m.T:
        y = pyo.value(m.y_charge[t])
        assert abs(y - round(y)) < 1e-4, f"y_charge[{t}] not integer: {y}"
        # And the structural property
        ch = pyo.value(m.charge[t])
        dis = pyo.value(m.discharge[t])
        assert ch < 1e-4 or dis < 1e-4, f"both legs open at {t}: ch={ch} dis={dis}"


def test_milp_full_episode_no_simultaneous(cfg, truth) -> None:
    """End-to-end MILP episode: in every period the binary constraint must hold."""
    fg = ForecastGenerator(cfg.forecast)
    pol = PyomoPolicy(cfg.battery, cfg.market, cfg.dt_hours,
                      use_stochastic=False, binary_battery_mode=True)
    r = run_episode(truth, cfg, pol, fg, forecast_seed=1)
    df = r.timeseries
    both = (df["charge_mw"] > 1e-6) & (df["discharge_mw"] > 1e-6)
    assert not both.any(), f"{both.sum()} periods have both legs open under MILP"


def test_milp_policy_runs_full_episode(cfg, truth) -> None:
    fg = ForecastGenerator(cfg.forecast)
    pol = PyomoPolicy(
        cfg.battery, cfg.market, cfg.dt_hours,
        use_stochastic=False, binary_battery_mode=True,
    )
    r = run_episode(truth, cfg, pol, fg, forecast_seed=1)
    assert np.isfinite(r.pnl.total)
    assert pol.name == "pyomo_deterministic_milp"


def test_lp_and_milp_close(cfg, truth) -> None:
    """With nonzero degradation cost the LP rarely opens both legs; MILP
    must therefore score within ~1% of LP on a typical day.
    """
    fg = ForecastGenerator(cfg.forecast)
    lp = PyomoPolicy(cfg.battery, cfg.market, cfg.dt_hours, use_stochastic=False)
    milp = PyomoPolicy(cfg.battery, cfg.market, cfg.dt_hours,
                       use_stochastic=False, binary_battery_mode=True)
    r_lp = run_episode(truth, cfg, lp, fg, forecast_seed=1)
    r_milp = run_episode(truth, cfg, milp, fg, forecast_seed=1)
    rel = abs(r_milp.pnl.total - r_lp.pnl.total) / max(abs(r_lp.pnl.total), 1.0)
    assert rel < 0.05, f"LP/MILP diverge: lp={r_lp.pnl.total:.2f} milp={r_milp.pnl.total:.2f} rel={rel:.4f}"
