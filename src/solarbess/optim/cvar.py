"""CVaR layer added to the stochastic Pyomo model (Phase 3)."""

from __future__ import annotations

import pyomo.environ as pyo


def add_cvar_layer(
    m: pyo.ConcreteModel,
    alpha: float,
    lambda_cvar: float,
    scenario_set: pyo.Set,
    prob: pyo.Param,
    profit_expr: pyo.Expression,
) -> pyo.Expression | None:
    """Add CVaR variables / constraint and return the cvar_loss expression.

    Returns None when lambda_cvar <= 0 (caller should skip the term).
    """
    if lambda_cvar <= 0:
        return None
    m.z_cvar = pyo.Var()
    m.u_cvar = pyo.Var(scenario_set, within=pyo.NonNegativeReals)

    def cvar_rule(m, o):
        return m.u_cvar[o] >= -profit_expr[o] - m.z_cvar
    m.cvar_constraint = pyo.Constraint(scenario_set, rule=cvar_rule)

    m.cvar_loss = pyo.Expression(
        expr=m.z_cvar + (1.0 / (1.0 - alpha)) * sum(prob[o] * m.u_cvar[o] for o in scenario_set)
    )
    return m.cvar_loss
