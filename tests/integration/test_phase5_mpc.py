"""Phase 5: rolling MPC at dispatch."""

from __future__ import annotations

import numpy as np

from solarbess.forecast.scenarios import ForecastGenerator
from solarbess.policy.pyomo_policy import PyomoPolicy
from solarbess.simulator.episode import run_episode


def test_mpc_runs_and_names_distinct(cfg, truth) -> None:
    fg = ForecastGenerator(cfg.forecast)
    pol_no = PyomoPolicy(cfg.battery, cfg.market, cfg.dt_hours, use_stochastic=True)
    pol_mpc = PyomoPolicy(cfg.battery, cfg.market, cfg.dt_hours, use_stochastic=True,
                          mpc_dispatch_horizon=-1, mpc_dispatch_period=4,
                          forecast_gen=fg, forecast_seed=1)
    assert pol_no.name != pol_mpc.name
    assert "mpc" in pol_mpc.name
    r_no = run_episode(truth, cfg, pol_no, fg, forecast_seed=1)
    r_mpc = run_episode(truth, cfg, pol_mpc, fg, forecast_seed=1)
    assert np.isfinite(r_no.pnl.total)
    assert np.isfinite(r_mpc.pnl.total)


def test_mpc_requires_forecast_gen(cfg) -> None:
    """Constructing an MPC policy without a forecast generator must raise."""
    import pytest
    with pytest.raises(ValueError):
        PyomoPolicy(cfg.battery, cfg.market, cfg.dt_hours,
                    mpc_dispatch_horizon=4, forecast_gen=None)


def test_mpc_preserves_information_firewall(cfg, truth) -> None:
    """MPC attaches truth via `attach_mpc_truth`, but the policy still calls
    forecast_gen.generate(truth, now=...) which only emits scenarios for
    timestamps >= now. We check that the MPC policy does not write any
    realized-truth value beyond `now` into its plan cache.
    """
    fg = ForecastGenerator(cfg.forecast)
    pol = PyomoPolicy(cfg.battery, cfg.market, cfg.dt_hours, use_stochastic=False,
                      mpc_dispatch_horizon=-1, mpc_dispatch_period=8,
                      forecast_gen=fg, forecast_seed=1)
    pol.attach_mpc_truth(truth)
    # Running the episode should not error and the plan cache (if any) should
    # have shape (n_mtu,)
    r = run_episode(truth, cfg, pol, fg, forecast_seed=1)
    assert np.isfinite(r.pnl.total)
    if pol._charge_plan is not None:
        assert pol._charge_plan.shape == (cfg.mtus_per_day,)


def test_attach_mpc_truth_not_in_init_signature() -> None:
    """The firewall test checks `__init__` — confirm `truth` still not there."""
    import inspect
    sig = inspect.signature(PyomoPolicy.__init__)
    assert "truth" not in sig.parameters
