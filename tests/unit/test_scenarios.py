"""ScenarioForecast: shapes, probability normalisation, lead-time variance monotone."""

from __future__ import annotations

import numpy as np
import pandas as pd

from solarbess.forecast.scenarios import ForecastGenerator
from solarbess.world.oracle import DecisionTime


def test_probabilities_sum_to_one(cfg, truth) -> None:
    fg = ForecastGenerator(cfg.forecast)
    sc = fg.generate(truth, now=truth.timestamps[0], decision_time=DecisionTime.DA_GATE, n_scenarios=20, seed=7)
    assert sc.probabilities.shape == (20,)
    assert abs(sc.probabilities.sum() - 1.0) < 1e-9


def test_shape_consistency(cfg, truth) -> None:
    fg = ForecastGenerator(cfg.forecast)
    sc = fg.generate(truth, now=truth.timestamps[0], decision_time=DecisionTime.DA_GATE, n_scenarios=30, seed=11)
    T = len(sc.timestamps)
    for arr in (sc.solar_mw, sc.da_price, sc.id_price, sc.imb_surplus, sc.imb_shortfall):
        assert arr.shape == (30, T)
    assert (sc.solar_mw >= 0).all()
    assert (sc.imb_shortfall > sc.imb_surplus).all()


def test_lead_time_variance_shrinks(cfg, truth) -> None:
    """Forecast variance at a fixed delivery period must shrink as decision time
    approaches it.
    """
    fg = ForecastGenerator(cfg.forecast)
    delivery_t = truth.timestamps[60]  # late in the day
    early_now = truth.timestamps[0] - pd.Timedelta(hours=12)
    late_now = truth.timestamps[40]  # 5 hours before delivery_t

    sc_early = fg.generate(truth, now=early_now, decision_time=DecisionTime.DA_GATE, n_scenarios=200, seed=99)
    sc_late = fg.generate(truth, now=late_now, decision_time=DecisionTime.IDA3, n_scenarios=200, seed=99)

    j_early = pd.Index(sc_early.timestamps).get_loc(delivery_t)
    j_late = pd.Index(sc_late.timestamps).get_loc(delivery_t)
    var_early = float(np.var(sc_early.solar_mw[:, j_early]))
    var_late = float(np.var(sc_late.solar_mw[:, j_late]))
    assert var_late < var_early, f"expected variance to shrink: early={var_early:.2f} late={var_late:.2f}"
