"""Phase 6: multi-stage stochastic Pyomo model built over a ScenarioTree.

Non-anticipativity is enforced by construction:
- q_da is indexed by t only (root decision, shared by all leaves).
- q_id is indexed by (ida_node, t) only (shared by all leaves in that IDA branch).
- charge, discharge, soc, imb_pos/neg, curtail are indexed by (leaf, t).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyomo.environ as pyo

from solarbess.config import BatteryParams, MarketParams
from solarbess.forecast.tree import ScenarioTree
from solarbess.optim.cvar import add_cvar_layer


def build_tree_stochastic_model(
    times: pd.DatetimeIndex,
    tree: ScenarioTree,
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
    n = len(times)
    if tree.timestamps.shape[0] != n:
        raise ValueError("tree timestamps must align with `times`")

    m = pyo.ConcreteModel()
    t_list = list(times)
    m.T = pyo.Set(initialize=t_list, ordered=True)

    leaf_ids = list(tree.leaf_ids.tolist())
    ida_ids = list(tree.ida_node_ids.tolist())
    m.LEAVES = pyo.Set(initialize=leaf_ids)
    m.IDA = pyo.Set(initialize=ida_ids)

    leaf_prob_norm = tree.leaf_probabilities / tree.leaf_probabilities.sum()
    m.leaf_prob = pyo.Param(
        m.LEAVES, initialize={leaf_ids[i]: float(leaf_prob_norm[i]) for i in range(len(leaf_ids))}
    )
    leaf_to_ida = {leaf_ids[i]: ida_ids[tree.leaf_to_ida[i]] for i in range(len(leaf_ids))}

    def _per_leaf(arr: np.ndarray) -> dict:
        return {(leaf_ids[i], t_list[j]): float(arr[i, j])
                for i in range(len(leaf_ids)) for j in range(n)}

    m.solar = pyo.Param(m.LEAVES, m.T, initialize=_per_leaf(tree.solar_mw))
    m.price_da = pyo.Param(m.LEAVES, m.T, initialize=_per_leaf(tree.da_price))
    m.price_id = pyo.Param(m.LEAVES, m.T, initialize=_per_leaf(tree.id_price))
    m.price_imb_surplus = pyo.Param(m.LEAVES, m.T, initialize=_per_leaf(tree.imb_surplus))
    m.price_imb_shortfall = pyo.Param(m.LEAVES, m.T, initialize=_per_leaf(tree.imb_shortfall))

    # Root decision
    if fixed_q_da is None:
        m.q_da = pyo.Var(m.T, bounds=(-market.import_limit_mw, market.export_limit_mw))
    else:
        if fixed_q_da.shape != (n,):
            raise ValueError("fixed_q_da shape mismatch")
        m.q_da = pyo.Param(
            m.T, initialize={t: float(fixed_q_da[i]) for i, t in enumerate(t_list)}, mutable=True
        )

    # IDA-stage decision: shared by descending leaves
    m.q_id_ida = pyo.Var(
        m.IDA, m.T, bounds=(-market.import_limit_mw, market.export_limit_mw)
    )

    # Leaf-level dispatch
    m.charge = pyo.Var(m.LEAVES, m.T, bounds=(0, battery.p_charge_max_mw))
    m.discharge = pyo.Var(m.LEAVES, m.T, bounds=(0, battery.p_discharge_max_mw))
    m.soc = pyo.Var(m.LEAVES, m.T, bounds=(battery.soc_min_mwh, battery.soc_max_mwh))
    m.curtail = pyo.Var(m.LEAVES, m.T, within=pyo.NonNegativeReals)
    m.imb_pos = pyo.Var(m.LEAVES, m.T, within=pyo.NonNegativeReals)
    m.imb_neg = pyo.Var(m.LEAVES, m.T, within=pyo.NonNegativeReals)

    if binary_battery_mode:
        m.y_charge = pyo.Var(m.LEAVES, m.T, within=pyo.Binary)

        def ch_bin(m, l, t):
            return m.charge[l, t] <= battery.p_charge_max_mw * m.y_charge[l, t]

        def dis_bin(m, l, t):
            return m.discharge[l, t] <= battery.p_discharge_max_mw * (1 - m.y_charge[l, t])

        m.binary_charge = pyo.Constraint(m.LEAVES, m.T, rule=ch_bin)
        m.binary_discharge = pyo.Constraint(m.LEAVES, m.T, rule=dis_bin)

    def _physical(m, leaf, t):
        return m.solar[leaf, t] - m.curtail[leaf, t] + m.discharge[leaf, t] - m.charge[leaf, t]
    m.physical = pyo.Expression(m.LEAVES, m.T, rule=_physical)

    def _schedule(m, leaf, t):
        ida = leaf_to_ida[leaf]
        return m.q_da[t] + m.q_id_ida[ida, t]
    m.schedule = pyo.Expression(m.LEAVES, m.T, rule=_schedule)

    def curtail_rule(m, l, t):
        return m.curtail[l, t] <= m.solar[l, t]
    m.curtail_limit = pyo.Constraint(m.LEAVES, m.T, rule=curtail_rule)

    if not allow_grid_charging:
        def solar_only(m, l, t):
            return m.charge[l, t] <= m.solar[l, t] - m.curtail[l, t]
        m.solar_only_charging = pyo.Constraint(m.LEAVES, m.T, rule=solar_only)

    def imbalance_rule(m, l, t):
        return m.physical[l, t] - m.schedule[l, t] == m.imb_pos[l, t] - m.imb_neg[l, t]
    m.imbalance_balance = pyo.Constraint(m.LEAVES, m.T, rule=imbalance_rule)

    def export_rule(m, l, t):
        return m.physical[l, t] <= market.export_limit_mw

    def import_rule(m, l, t):
        return m.physical[l, t] >= -market.import_limit_mw
    m.export_limit_c = pyo.Constraint(m.LEAVES, m.T, rule=export_rule)
    m.import_limit_c = pyo.Constraint(m.LEAVES, m.T, rule=import_rule)

    # SoC balance per leaf
    idx_map = {t: i for i, t in enumerate(t_list)}
    eta_ch = battery.eta_charge
    eta_dis = battery.eta_discharge

    def soc_rule(m, l, t):
        i = idx_map[t]
        prev = soc_initial_mwh if i == 0 else m.soc[l, t_list[i - 1]]
        return m.soc[l, t] == prev + eta_ch * m.charge[l, t] * dt_hours - m.discharge[l, t] * dt_hours / eta_dis
    m.soc_balance = pyo.Constraint(m.LEAVES, m.T, rule=soc_rule)

    if enforce_terminal_soc:
        final_t = t_list[-1]

        def terminal_rule(m, l):
            return m.soc[l, final_t] >= battery.soc_terminal_min_mwh
        m.terminal_soc = pyo.Constraint(m.LEAVES, rule=terminal_rule)

    degr = battery.degradation_cost_eur_per_mwh

    def leaf_profit_rule(m, l):
        return sum(
            dt_hours * (
                m.price_da[l, t] * m.q_da[t]
                + m.price_id[l, t] * m.q_id_ida[leaf_to_ida[l], t]
                + m.price_imb_surplus[l, t] * m.imb_pos[l, t]
                - m.price_imb_shortfall[l, t] * m.imb_neg[l, t]
                - degr * (m.charge[l, t] + m.discharge[l, t])
            )
            for t in m.T
        )
    m.leaf_profit = pyo.Expression(m.LEAVES, rule=leaf_profit_rule)
    m.expected_profit = pyo.Expression(
        expr=sum(m.leaf_prob[l] * m.leaf_profit[l] for l in m.LEAVES)
    )

    obj_expr = m.expected_profit
    cvar_loss = add_cvar_layer(m, cvar_alpha, lambda_cvar, m.LEAVES, m.leaf_prob, m.leaf_profit)
    if cvar_loss is not None:
        obj_expr = obj_expr - lambda_cvar * cvar_loss
    if lambda_imbalance > 0:
        m.imbalance_penalty = pyo.Expression(
            expr=sum(
                m.leaf_prob[l] * dt_hours * (m.imb_pos[l, t] + m.imb_neg[l, t])
                for l in m.LEAVES for t in m.T
            )
        )
        obj_expr = obj_expr - lambda_imbalance * m.imbalance_penalty

    m.obj = pyo.Objective(expr=obj_expr, sense=pyo.maximize)
    return m
