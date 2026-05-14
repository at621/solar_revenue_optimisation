"""Phase 6: multi-stage scenario tree with non-anticipativity."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pyomo.environ as pyo

from solarbess.config import BatteryParams, MarketParams
from solarbess.forecast.scenarios import ForecastGenerator, ScenarioForecast
from solarbess.forecast.tree import ScenarioTree, build_tree_from_scenarios
from solarbess.optim.solve import solve_highs
from solarbess.optim.tree_builder import build_tree_stochastic_model
from solarbess.policy.tree_policy import TreePolicy
from solarbess.simulator.episode import run_episode
from solarbess.world.oracle import DecisionTime


def _toy_forecast() -> ScenarioForecast:
    times = pd.date_range("2026-06-01T00:00:00Z", periods=4, freq="15min")
    return ScenarioForecast(
        scenario_ids=np.arange(8),
        probabilities=np.ones(8) / 8,
        timestamps=times,
        solar_mw=np.tile(np.array([[0, 5, 5, 0]], dtype=float), (8, 1)),
        da_price=np.array([
            [30, 25, 70, 55], [35, 28, 75, 58],
            [40, 32, 80, 60], [45, 38, 85, 62],
            [50, 42, 90, 65], [55, 48, 95, 68],
            [60, 52, 100, 70], [65, 58, 110, 75],
        ], dtype=float),
        id_price=np.tile(np.array([[40, 30, 80, 60]], dtype=float), (8, 1)),
        imb_surplus=np.tile(np.array([[20, 15, 60, 40]], dtype=float), (8, 1)),
        imb_shortfall=np.tile(np.array([[60, 50, 100, 80]], dtype=float), (8, 1)),
        decision_time=DecisionTime.DA_GATE,
    )


def test_tree_clustering_partitions_leaves() -> None:
    sc = _toy_forecast()
    tree = build_tree_from_scenarios(sc, n_ida_nodes=4)
    assert isinstance(tree, ScenarioTree)
    assert tree.solar_mw.shape == sc.solar_mw.shape
    # Sort-and-split = roughly equal bucket sizes
    sizes = [int((tree.leaf_to_ida == k).sum()) for k in tree.ida_node_ids]
    assert max(sizes) - min(sizes) <= 1
    assert math.isclose(tree.leaf_probabilities.sum(), 1.0, abs_tol=1e-9)
    assert math.isclose(tree.ida_probabilities.sum(), 1.0, abs_tol=1e-9)


def test_tree_model_solves() -> None:
    sc = _toy_forecast()
    tree = build_tree_from_scenarios(sc, n_ida_nodes=3)
    bp = BatteryParams(
        capacity_mwh=10, p_charge_max_mw=2, p_discharge_max_mw=2,
        eta_charge=0.95, eta_discharge=0.95,
        soc_min_mwh=1, soc_max_mwh=9, soc_initial_mwh=5,
        soc_terminal_min_mwh=1, degradation_cost_eur_per_mwh=2,
    )
    mp = MarketParams(export_limit_mw=20, import_limit_mw=0)
    m = build_tree_stochastic_model(
        times=sc.timestamps, tree=tree, battery=bp, market=mp,
        dt_hours=0.25, soc_initial_mwh=5.0,
    )
    res = solve_highs(m, time_limit_s=10.0)
    assert np.isfinite(res.objective)


def test_tree_enforces_non_anticipativity() -> None:
    """q_id_ida is indexed by IDA node only -> all leaves in the same IDA branch
    see the same q_id_ida[ida, t]. This is non-anticipativity by construction.
    """
    sc = _toy_forecast()
    tree = build_tree_from_scenarios(sc, n_ida_nodes=3)
    bp = BatteryParams(
        capacity_mwh=10, p_charge_max_mw=2, p_discharge_max_mw=2,
        eta_charge=0.95, eta_discharge=0.95,
        soc_min_mwh=1, soc_max_mwh=9, soc_initial_mwh=5,
        soc_terminal_min_mwh=1, degradation_cost_eur_per_mwh=2,
    )
    mp = MarketParams(export_limit_mw=20, import_limit_mw=0)
    m = build_tree_stochastic_model(
        times=sc.timestamps, tree=tree, battery=bp, market=mp,
        dt_hours=0.25, soc_initial_mwh=5.0,
    )
    solve_highs(m, time_limit_s=10.0)
    # Verify: q_id_ida exists, indexed by (ida, t); not by leaf.
    assert hasattr(m, "q_id_ida")
    # Look at one (ida, t) tuple and check it resolves to a single Var component.
    # Then check that two leaves in the same IDA branch yield identical schedule
    # contributions from q_id_ida.
    leaves_in_ida0 = [int(tree.leaf_ids[i]) for i in range(len(tree.leaf_ids))
                      if tree.leaf_to_ida[i] == 0]
    if len(leaves_in_ida0) >= 2:
        l1, l2 = leaves_in_ida0[0], leaves_in_ida0[1]
        for t in sc.timestamps:
            s1 = pyo.value(m.schedule[l1, t])
            s2 = pyo.value(m.schedule[l2, t])
            assert abs(s1 - s2) < 1e-6, (
                f"non-anticipativity broken: leaves {l1},{l2} share IDA branch "
                f"but schedule diverges at {t}: {s1} vs {s2}"
            )


def test_tree_policy_runs_episode(cfg, truth) -> None:
    fg = ForecastGenerator(cfg.forecast)
    pol = TreePolicy(cfg.battery, cfg.market, cfg.dt_hours, n_ida_nodes=3)
    r = run_episode(truth, cfg, pol, fg, forecast_seed=1)
    assert np.isfinite(r.pnl.total)
    assert pol.name == "tree_ida3"


def test_tree_policy_no_truth_in_init() -> None:
    """Firewall: TreePolicy must not accept truth in __init__."""
    import inspect
    sig = inspect.signature(TreePolicy.__init__)
    assert "truth" not in sig.parameters
