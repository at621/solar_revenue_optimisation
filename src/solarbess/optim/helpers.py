"""Shared Pyomo helpers used by both deterministic and stochastic builders.

The key fix vs the skeleton in docs/plan.md: SoC ordering uses a precomputed
index map instead of `T_list.index(t)` inside the rule (which is O(T) per call).
"""

from __future__ import annotations

import pyomo.environ as pyo

from solarbess.config import BatteryParams, MarketParams


def index_map(t_list: list) -> dict:
    return {t: i for i, t in enumerate(t_list)}


def add_battery_vars(
    m: pyo.ConcreteModel,
    scenario_set: pyo.Set | None,
    battery: BatteryParams,
) -> None:
    """Add charge, discharge, soc, curtail, imb_pos, imb_neg variables."""
    if scenario_set is not None:
        m.charge = pyo.Var(scenario_set, m.T, bounds=(0, battery.p_charge_max_mw))
        m.discharge = pyo.Var(scenario_set, m.T, bounds=(0, battery.p_discharge_max_mw))
        m.soc = pyo.Var(scenario_set, m.T, bounds=(battery.soc_min_mwh, battery.soc_max_mwh))
        m.curtail = pyo.Var(scenario_set, m.T, within=pyo.NonNegativeReals)
        m.imb_pos = pyo.Var(scenario_set, m.T, within=pyo.NonNegativeReals)
        m.imb_neg = pyo.Var(scenario_set, m.T, within=pyo.NonNegativeReals)
    else:
        m.charge = pyo.Var(m.T, bounds=(0, battery.p_charge_max_mw))
        m.discharge = pyo.Var(m.T, bounds=(0, battery.p_discharge_max_mw))
        m.soc = pyo.Var(m.T, bounds=(battery.soc_min_mwh, battery.soc_max_mwh))
        m.curtail = pyo.Var(m.T, within=pyo.NonNegativeReals)
        m.imb_pos = pyo.Var(m.T, within=pyo.NonNegativeReals)
        m.imb_neg = pyo.Var(m.T, within=pyo.NonNegativeReals)


def add_soc_balance(
    m: pyo.ConcreteModel,
    scenario_set: pyo.Set | None,
    t_list: list,
    battery: BatteryParams,
    soc_initial_mwh: float,
    dt_hours: float,
) -> None:
    idx_map = index_map(t_list)
    eta_ch = battery.eta_charge
    eta_dis = battery.eta_discharge

    if scenario_set is not None:
        def rule(m, o, t):
            i = idx_map[t]
            prev = soc_initial_mwh if i == 0 else m.soc[o, t_list[i - 1]]
            return m.soc[o, t] == prev + eta_ch * m.charge[o, t] * dt_hours - m.discharge[o, t] * dt_hours / eta_dis
        m.soc_balance = pyo.Constraint(scenario_set, m.T, rule=rule)
    else:
        def rule(m, t):
            i = idx_map[t]
            prev = soc_initial_mwh if i == 0 else m.soc[t_list[i - 1]]
            return m.soc[t] == prev + eta_ch * m.charge[t] * dt_hours - m.discharge[t] * dt_hours / eta_dis
        m.soc_balance = pyo.Constraint(m.T, rule=rule)


def add_solar_only_charging(
    m: pyo.ConcreteModel,
    scenario_set: pyo.Set | None,
) -> None:
    """charge <= solar - curtail (uncurtailed solar bounds charge)."""
    if scenario_set is not None:
        def rule(m, o, t):
            return m.charge[o, t] <= m.solar[o, t] - m.curtail[o, t]
        m.solar_only_charging = pyo.Constraint(scenario_set, m.T, rule=rule)
    else:
        def rule(m, t):
            return m.charge[t] <= m.solar[t] - m.curtail[t]
        m.solar_only_charging = pyo.Constraint(m.T, rule=rule)


def add_curtail_limit(
    m: pyo.ConcreteModel,
    scenario_set: pyo.Set | None,
) -> None:
    """curtail <= solar."""
    if scenario_set is not None:
        def rule(m, o, t):
            return m.curtail[o, t] <= m.solar[o, t]
        m.curtail_limit = pyo.Constraint(scenario_set, m.T, rule=rule)
    else:
        def rule(m, t):
            return m.curtail[t] <= m.solar[t]
        m.curtail_limit = pyo.Constraint(m.T, rule=rule)


def add_export_import_limits(
    m: pyo.ConcreteModel,
    scenario_set: pyo.Set | None,
    market: MarketParams,
) -> None:
    if scenario_set is not None:
        def export_rule(m, o, t):
            return m.physical[o, t] <= market.export_limit_mw

        def import_rule(m, o, t):
            return m.physical[o, t] >= -market.import_limit_mw

        m.export_limit_c = pyo.Constraint(scenario_set, m.T, rule=export_rule)
        m.import_limit_c = pyo.Constraint(scenario_set, m.T, rule=import_rule)
    else:
        def export_rule_det(m, t):
            return m.physical[t] <= market.export_limit_mw

        def import_rule_det(m, t):
            return m.physical[t] >= -market.import_limit_mw

        m.export_limit_c = pyo.Constraint(m.T, rule=export_rule_det)
        m.import_limit_c = pyo.Constraint(m.T, rule=import_rule_det)


def add_terminal_soc(
    m: pyo.ConcreteModel,
    scenario_set: pyo.Set | None,
    t_list: list,
    soc_terminal_min: float,
) -> None:
    final_t = t_list[-1]
    if scenario_set is not None:
        def rule(m, o):
            return m.soc[o, final_t] >= soc_terminal_min
        m.terminal_soc = pyo.Constraint(scenario_set, rule=rule)
    else:
        m.terminal_soc = pyo.Constraint(expr=m.soc[final_t] >= soc_terminal_min)


def add_imbalance_balance(
    m: pyo.ConcreteModel,
    scenario_set: pyo.Set | None,
) -> None:
    """physical - schedule == imb_pos - imb_neg."""
    if scenario_set is not None:
        def rule(m, o, t):
            return m.physical[o, t] - m.schedule[o, t] == m.imb_pos[o, t] - m.imb_neg[o, t]
        m.imbalance_balance = pyo.Constraint(scenario_set, m.T, rule=rule)
    else:
        def rule(m, t):
            return m.physical[t] - m.schedule[t] == m.imb_pos[t] - m.imb_neg[t]
        m.imbalance_balance = pyo.Constraint(m.T, rule=rule)


def add_no_simultaneous_charge_discharge(
    m: pyo.ConcreteModel,
    scenario_set: pyo.Set | None,
    battery: BatteryParams,
) -> None:
    """Phase 4: binary y_charge[t] forbids charge and discharge in the same MTU.

    charge    <= p_charge_max    * y_charge
    discharge <= p_discharge_max * (1 - y_charge)

    With degradation cost > 0 the LP relaxation rarely opens both legs, but it
    can happen under exotic price patterns. MILP eliminates the artefact.
    """
    p_ch = battery.p_charge_max_mw
    p_dis = battery.p_discharge_max_mw
    if scenario_set is not None:
        m.y_charge = pyo.Var(scenario_set, m.T, within=pyo.Binary)

        def ch_rule(m, o, t):
            return m.charge[o, t] <= p_ch * m.y_charge[o, t]

        def dis_rule(m, o, t):
            return m.discharge[o, t] <= p_dis * (1 - m.y_charge[o, t])

        m.binary_charge = pyo.Constraint(scenario_set, m.T, rule=ch_rule)
        m.binary_discharge = pyo.Constraint(scenario_set, m.T, rule=dis_rule)
    else:
        m.y_charge = pyo.Var(m.T, within=pyo.Binary)

        def ch_rule_det(m, t):
            return m.charge[t] <= p_ch * m.y_charge[t]

        def dis_rule_det(m, t):
            return m.discharge[t] <= p_dis * (1 - m.y_charge[t])

        m.binary_charge = pyo.Constraint(m.T, rule=ch_rule_det)
        m.binary_discharge = pyo.Constraint(m.T, rule=dis_rule_det)
