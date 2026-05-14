"""HiGHS solver wrapper via Pyomo APPSI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pyomo.environ as pyo
from pyomo.contrib.appsi.base import TerminationCondition
from pyomo.contrib.appsi.solvers import Highs


@dataclass(frozen=True)
class SolveResult:
    termination: str
    objective: float
    wall_time_s: float | None
    info: dict[str, Any]


def solve_highs(model: pyo.ConcreteModel, time_limit_s: float = 60.0, mip_rel_gap: float = 1e-3) -> SolveResult:
    """Solve `model` with HiGHS via Pyomo APPSI.

    Returns a SolveResult with `objective = nan` for any non-optimal outcome
    (infeasible, unbounded, time-limit, or load-solution failure). Callers
    must check `np.isfinite(result.objective)` before reading variable values.
    """
    solver = Highs()
    solver.config.time_limit = time_limit_s
    # Don't raise on infeasible / no-solution - we want a clean nan back.
    solver.config.load_solution = False
    solver.highs_options["mip_rel_gap"] = mip_rel_gap
    try:
        results = solver.solve(model)
    except Exception as exc:
        return SolveResult(termination=f"solver_exception:{exc.__class__.__name__}",
                           objective=float("nan"), wall_time_s=None, info={"error": str(exc)})
    cond = str(results.termination_condition)
    wall: float | None = None
    try:
        wall = float(getattr(results, "timing_info", {}).get("wall_time", float("nan")))
    except Exception:
        wall = None
    optimal_set = {TerminationCondition.optimal}
    if hasattr(TerminationCondition, "locallyOptimal"):
        optimal_set.add(getattr(TerminationCondition, "locallyOptimal"))
    if results.termination_condition in optimal_set:
        try:
            solver.load_vars()
            obj = float(pyo.value(model.obj))
        except Exception:
            obj = float("nan")
    else:
        obj = float("nan")
    return SolveResult(termination=cond, objective=obj, wall_time_s=wall, info={})
