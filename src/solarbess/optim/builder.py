"""Two Pyomo model builders: deterministic (Phase 1) and stochastic (Phase 2/3).

Both share `optim.helpers` for SoC balance, solar-only charging, export/import
limits, curtailment, and imbalance balance. The stochastic builder additionally
adds the CVaR layer through `optim.cvar.add_cvar_layer`.

Sign convention (mirrors `market/settlement.py`):
- positive q = export (sell)
- imb_pos paid at price_imb_surplus
- imb_neg charged at price_imb_shortfall
- physical = solar - curtail + discharge - charge
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyomo.environ as pyo

from solarbess.config import BatteryParams, MarketParams
from solarbess.forecast.scenarios import ScenarioForecast
from solarbess.optim.cvar import add_cvar_layer
from solarbess.optim.helpers import (
    add_battery_vars,
    add_curtail_limit,
    add_export_import_limits,
    add_imbalance_balance,
    add_no_simultaneous_charge_discharge,
    add_solar_only_charging,
    add_soc_balance,
    add_terminal_soc,
)


def build_deterministic_model(
    times: pd.DatetimeIndex,
    solar_mw: np.ndarray,
    da_price_mtu: np.ndarray,
    id_price_mtu: np.ndarray,
    imb_surplus: np.ndarray,
    imb_shortfall: np.ndarray,
    battery: BatteryParams,
    market: MarketParams,
    dt_hours: float,
    soc_initial_mwh: float,
    fixed_q_da: np.ndarray | None = None,
    enforce_terminal_soc: bool = True,
    allow_grid_charging: bool = False,
    binary_battery_mode: bool = False,
) -> pyo.ConcreteModel:
    """Single-scenario LP. `fixed_q_da` switches to intraday mode."""
    n = len(times)
    if solar_mw.shape != (n,) or da_price_mtu.shape != (n,):
        raise ValueError("input arrays must align with times")

    m = pyo.ConcreteModel()
    t_list = list(times)
    m.T = pyo.Set(initialize=t_list, ordered=True)

    m.solar = pyo.Param(m.T, initialize={t: float(solar_mw[i]) for i, t in enumerate(t_list)})
    m.price_da = pyo.Param(m.T, initialize={t: float(da_price_mtu[i]) for i, t in enumerate(t_list)})
    m.price_id = pyo.Param(m.T, initialize={t: float(id_price_mtu[i]) for i, t in enumerate(t_list)})
    m.price_imb_surplus = pyo.Param(
        m.T, initialize={t: float(imb_surplus[i]) for i, t in enumerate(t_list)}
    )
    m.price_imb_shortfall = pyo.Param(
        m.T, initialize={t: float(imb_shortfall[i]) for i, t in enumerate(t_list)}
    )

    if fixed_q_da is None:
        m.q_da = pyo.Var(m.T, bounds=(-market.import_limit_mw, market.export_limit_mw))
    else:
        if fixed_q_da.shape != (n,):
            raise ValueError("fixed_q_da shape mismatch")
        m.q_da = pyo.Param(
            m.T, initialize={t: float(fixed_q_da[i]) for i, t in enumerate(t_list)}, mutable=True
        )
    m.q_id = pyo.Var(m.T, bounds=(-market.import_limit_mw, market.export_limit_mw))

    add_battery_vars(m, None, battery)

    def _physical(m, t):
        return m.solar[t] - m.curtail[t] + m.discharge[t] - m.charge[t]
    m.physical = pyo.Expression(m.T, rule=_physical)

    def _schedule(m, t):
        return m.q_da[t] + m.q_id[t]
    m.schedule = pyo.Expression(m.T, rule=_schedule)

    add_curtail_limit(m, None)
    if not allow_grid_charging:
        add_solar_only_charging(m, None)
    add_soc_balance(m, None, t_list, battery, soc_initial_mwh, dt_hours)
    add_export_import_limits(m, None, market)
    add_imbalance_balance(m, None)
    if enforce_terminal_soc:
        add_terminal_soc(m, None, t_list, battery.soc_terminal_min_mwh)
    if binary_battery_mode:
        add_no_simultaneous_charge_discharge(m, None, battery)

    degr = battery.degradation_cost_eur_per_mwh
    # Trading fees are |q|-proportional and would require auxiliary positive/negative
    # split variables to keep the model linear. MVP charges fees in settlement only;
    # the optimiser is fee-insensitive. This is acceptable while fees are small
    # relative to expected profit (per default config: 0.05 EUR/MWh).

    profit = sum(
        dt_hours * (
            m.price_da[t] * m.q_da[t]
            + m.price_id[t] * m.q_id[t]
            + m.price_imb_surplus[t] * m.imb_pos[t]
            - m.price_imb_shortfall[t] * m.imb_neg[t]
            - degr * (m.charge[t] + m.discharge[t])
        )
        for t in m.T
    )

    m.obj = pyo.Objective(expr=profit, sense=pyo.maximize)
    return m


def build_stochastic_model(
    times: pd.DatetimeIndex,
    scenarios: ScenarioForecast,
    battery: BatteryParams,
    market: MarketParams,
    dt_hours: float,
    soc_initial_mwh: float,
    fixed_q_da: np.ndarray | None = None,
    cvar_alpha: float = 0.95,
    lambda_cvar: float = 0.0,
    lambda_imbalance: float = 0.0,
    enforce_terminal_soc: bool = True,
    allow_grid_charging: bool = False,
    binary_battery_mode: bool = False,
) -> pyo.ConcreteModel:
    """Multi-scenario extensive-form LP. q_DA is the here-and-now decision."""
    n = len(times)
    if scenarios.timestamps.shape[0] != n:
        # Allow forecast subset (intraday): align by re-indexing scenarios onto `times`.
        raise ValueError(
            f"scenarios.timestamps len {len(scenarios.timestamps)} != times len {n}; "
            "caller must align before invoking the builder."
        )

    m = pyo.ConcreteModel()
    t_list = list(times)
    m.T = pyo.Set(initialize=t_list, ordered=True)
    omega_ids = list(scenarios.scenario_ids.tolist())
    m.Omega = pyo.Set(initialize=omega_ids)

    probs = scenarios.probabilities / scenarios.probabilities.sum()
    m.prob = pyo.Param(m.Omega, initialize={o: float(probs[i]) for i, o in enumerate(omega_ids)})

    def _array_param(arr: np.ndarray) -> dict:
        return {(omega_ids[i], t_list[j]): float(arr[i, j]) for i in range(len(omega_ids)) for j in range(n)}

    m.solar = pyo.Param(m.Omega, m.T, initialize=_array_param(scenarios.solar_mw))
    m.price_da = pyo.Param(m.Omega, m.T, initialize=_array_param(scenarios.da_price))
    m.price_id = pyo.Param(m.Omega, m.T, initialize=_array_param(scenarios.id_price))
    m.price_imb_surplus = pyo.Param(m.Omega, m.T, initialize=_array_param(scenarios.imb_surplus))
    m.price_imb_shortfall = pyo.Param(m.Omega, m.T, initialize=_array_param(scenarios.imb_shortfall))

    # First-stage DA decision is shared across scenarios (here-and-now)
    if fixed_q_da is None:
        m.q_da = pyo.Var(m.T, bounds=(-market.import_limit_mw, market.export_limit_mw))
    else:
        if fixed_q_da.shape != (n,):
            raise ValueError("fixed_q_da shape mismatch")
        m.q_da = pyo.Param(
            m.T, initialize={t: float(fixed_q_da[i]) for i, t in enumerate(t_list)}, mutable=True
        )
    m.q_id = pyo.Var(m.Omega, m.T, bounds=(-market.import_limit_mw, market.export_limit_mw))

    add_battery_vars(m, m.Omega, battery)

    def _physical(m, o, t):
        return m.solar[o, t] - m.curtail[o, t] + m.discharge[o, t] - m.charge[o, t]
    m.physical = pyo.Expression(m.Omega, m.T, rule=_physical)

    def _schedule(m, o, t):
        return m.q_da[t] + m.q_id[o, t]
    m.schedule = pyo.Expression(m.Omega, m.T, rule=_schedule)

    add_curtail_limit(m, m.Omega)
    if not allow_grid_charging:
        add_solar_only_charging(m, m.Omega)
    add_soc_balance(m, m.Omega, t_list, battery, soc_initial_mwh, dt_hours)
    add_export_import_limits(m, m.Omega, market)
    add_imbalance_balance(m, m.Omega)
    if enforce_terminal_soc:
        add_terminal_soc(m, m.Omega, t_list, battery.soc_terminal_min_mwh)
    if binary_battery_mode:
        add_no_simultaneous_charge_discharge(m, m.Omega, battery)

    degr = battery.degradation_cost_eur_per_mwh

    def _profit(m, o):
        return sum(
            dt_hours * (
                m.price_da[o, t] * m.q_da[t]
                + m.price_id[o, t] * m.q_id[o, t]
                + m.price_imb_surplus[o, t] * m.imb_pos[o, t]
                - m.price_imb_shortfall[o, t] * m.imb_neg[o, t]
                - degr * (m.charge[o, t] + m.discharge[o, t])
            )
            for t in m.T
        )
    m.scenario_profit = pyo.Expression(m.Omega, rule=_profit)

    m.expected_profit = pyo.Expression(
        expr=sum(m.prob[o] * m.scenario_profit[o] for o in m.Omega)
    )

    obj_expr = m.expected_profit
    cvar_loss = add_cvar_layer(m, cvar_alpha, lambda_cvar, m.Omega, m.prob, m.scenario_profit)
    if cvar_loss is not None:
        obj_expr = obj_expr - lambda_cvar * cvar_loss
    if lambda_imbalance > 0:
        m.imbalance_penalty = pyo.Expression(
            expr=sum(
                m.prob[o] * dt_hours * (m.imb_pos[o, t] + m.imb_neg[o, t])
                for o in m.Omega for t in m.T
            )
        )
        obj_expr = obj_expr - lambda_imbalance * m.imbalance_penalty

    m.obj = pyo.Objective(expr=obj_expr, sense=pyo.maximize)
    return m

