"""OraclePolicy - perfect foresight upper bound.

This is the ONLY policy class that intentionally accepts a TruthPath. The test
suite enforces no other policy receives one.

The oracle solves a single deterministic LP on the sealed truth path and serves
all three callbacks from the cached solution.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pyomo.environ as pyo

from solarbess.asset.battery import AssetState
from solarbess.config import BatteryParams, MarketParams
from solarbess.market.interface import Position
from solarbess.optim.builder import build_deterministic_model
from solarbess.optim.solve import solve_highs
from solarbess.world.oracle import InfoSet

if TYPE_CHECKING:
    from solarbess.world.truth import TruthPath


class OraclePolicy:
    name = "oracle"

    def __init__(
        self,
        battery: BatteryParams,
        market: MarketParams,
        dt_hours: float,
        truth: "TruthPath",
        solver_time_limit_s: float = 30.0,
    ):
        self.battery = battery
        self.market = market
        self.dt = dt_hours
        self._truth = truth
        self.solver_time_limit_s = solver_time_limit_s
        self._q_da: np.ndarray | None = None
        self._q_id: np.ndarray | None = None
        self._charge: np.ndarray | None = None
        self._discharge: np.ndarray | None = None

    def _solve(self, soc_initial_mwh: float) -> None:
        t = self._truth
        m = build_deterministic_model(
            times=t.timestamps,
            solar_mw=t.solar_mw,
            da_price_mtu=t.da_price_mtu,
            id_price_mtu=t.id_price_mtu[:, -1],
            imb_surplus=t.imb_surplus_price,
            imb_shortfall=t.imb_shortfall_price,
            battery=self.battery,
            market=self.market,
            dt_hours=self.dt,
            soc_initial_mwh=soc_initial_mwh,
        )
        res = solve_highs(m, time_limit_s=self.solver_time_limit_s)
        if not np.isfinite(res.objective):
            raise RuntimeError(f"Oracle solve failed: {res.termination}")
        self._q_da = np.array([pyo.value(m.q_da[ts]) for ts in m.T])
        self._q_id = np.array([pyo.value(m.q_id[ts]) for ts in m.T])
        self._charge = np.array([pyo.value(m.charge[ts]) for ts in m.T])
        self._discharge = np.array([pyo.value(m.discharge[ts]) for ts in m.T])

    def decide_day_ahead(self, info: InfoSet, asset_state: AssetState) -> np.ndarray:
        self._solve(asset_state.soc_mwh)
        assert self._q_da is not None
        return self._q_da.copy()

    def decide_intraday(
        self,
        info: InfoSet,
        asset_state: AssetState,
        position: Position,
        auction_round: int,
    ) -> np.ndarray:
        # On the final round, submit the cached q_id (deltas vs prior rounds);
        # otherwise zero. This keeps the total ID quantity equal to the oracle plan.
        n_rounds = position.n_rounds
        n_mtu = info.timestamps.shape[0]
        if self._q_id is None:
            return np.zeros(n_mtu)
        if auction_round == n_rounds - 1:
            already = position.id_trades_mw.sum(axis=1)
            return self._q_id - already
        return np.zeros(n_mtu)

    def decide_dispatch(
        self,
        info: InfoSet,
        asset_state: AssetState,
        position: Position,
    ) -> tuple[float, float]:
        t = info.delivery_period_idx
        if t is None or self._charge is None or self._discharge is None:
            return 0.0, 0.0
        bp = self.battery
        dt = self.dt
        charge = float(self._charge[t])
        discharge = float(self._discharge[t])
        realized_solar = float(self._truth.solar_mw[t])
        charge = min(max(charge, 0.0), realized_solar, bp.p_charge_max_mw)
        room_mwh = bp.soc_max_mwh - asset_state.soc_mwh
        charge = min(charge, room_mwh / max(bp.eta_charge * dt, 1e-9))
        charge = max(charge, 0.0)
        avail_mwh = asset_state.soc_mwh - bp.soc_min_mwh
        discharge = min(max(discharge, 0.0), bp.p_discharge_max_mw, (avail_mwh * bp.eta_discharge) / max(dt, 1e-9))
        discharge = max(discharge, 0.0)
        if charge > 0 and discharge > 0:
            if charge >= discharge:
                charge -= discharge
                discharge = 0.0
            else:
                discharge -= charge
                charge = 0.0
        return charge, discharge
