"""Rule-based policy: a deterministic heuristic over scenario p50 forecasts.

Strategy
--------
1. DA gate: take the per-period p50 forecast from `info.forecast`. Build a plan:
   - Identify cheap+sunny periods (forecast DA price below 30th percentile AND
     forecast solar > 0): mark for charging.
   - Identify peak periods (forecast DA price above 70th percentile):
     mark for discharging.
   - Sweep periods chronologically, accumulating battery state, respecting
     soc bounds and per-step power limits.
   - Submit `q_DA[t] = max(forecast_solar_p50[t] - planned_charge[t] + planned_discharge[t], 0)`.

2. Intraday: when forecast solar mean for a remaining period shifts by more
   than `intraday_threshold_mw` relative to the value used at the previous
   decision, trade the delta at this round's forecast price - sell when the
   forecast moved up, buy back when it moved down. This is myopic; the goal is
   to demonstrate the callback works end-to-end and provide a baseline below
   the Pyomo policy.

3. Dispatch: execute the original plan, but cap `charge <= realized_solar`
   and `discharge` such that the battery does not violate SoC bounds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from solarbess.asset.battery import AssetState
from solarbess.config import BatteryParams, MarketParams
from solarbess.market.interface import Position
from solarbess.world.oracle import DecisionTime, InfoSet


@dataclass
class RuleBasedParams:
    cheap_quantile: float = 0.30
    peak_quantile: float = 0.70
    intraday_threshold_mw: float = 1.0


class RuleBasedPolicy:
    name = "rule_based"

    def __init__(
        self,
        battery: BatteryParams,
        market: MarketParams,
        dt_hours: float,
        params: RuleBasedParams | None = None,
    ):
        self.battery = battery
        self.market = market
        self.dt = dt_hours
        self.params = params or RuleBasedParams()
        # Per-episode planning artefacts
        self._planned_charge_mw: np.ndarray | None = None
        self._planned_discharge_mw: np.ndarray | None = None
        self._planned_q_da: np.ndarray | None = None
        self._last_forecast_solar: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Day ahead
    # ------------------------------------------------------------------
    def decide_day_ahead(self, info: InfoSet, asset_state: AssetState) -> np.ndarray:
        if info.forecast is None:
            raise ValueError("RuleBasedPolicy requires a ScenarioForecast at DA gate.")
        sc = info.forecast
        solar_p50 = np.median(sc.solar_mw, axis=0)
        da_p50 = np.median(sc.da_price, axis=0)
        n = solar_p50.shape[0]

        cheap_thr = np.quantile(da_p50, self.params.cheap_quantile)
        peak_thr = np.quantile(da_p50, self.params.peak_quantile)

        plan_charge = np.zeros(n)
        plan_discharge = np.zeros(n)
        soc = asset_state.soc_mwh
        bp = self.battery
        dt = self.dt

        for t in range(n):
            charge = 0.0
            discharge = 0.0
            if da_p50[t] <= cheap_thr and solar_p50[t] > 0.0:
                room_mwh = bp.soc_max_mwh - soc
                charge = min(bp.p_charge_max_mw, solar_p50[t], room_mwh / max(bp.eta_charge * dt, 1e-9))
                charge = max(charge, 0.0)
            elif da_p50[t] >= peak_thr:
                avail_mwh = soc - bp.soc_min_mwh
                discharge = min(
                    bp.p_discharge_max_mw,
                    (avail_mwh * bp.eta_discharge) / max(dt, 1e-9),
                )
                discharge = max(discharge, 0.0)
            plan_charge[t] = charge
            plan_discharge[t] = discharge
            soc = soc + bp.eta_charge * charge * dt - discharge * dt / bp.eta_discharge
            soc = min(max(soc, bp.soc_min_mwh), bp.soc_max_mwh)

        export_limit = self.market.export_limit_mw
        q_da = np.clip(solar_p50 - plan_charge + plan_discharge, 0.0, export_limit)
        self._planned_charge_mw = plan_charge
        self._planned_discharge_mw = plan_discharge
        self._planned_q_da = q_da
        self._last_forecast_solar = solar_p50.copy()
        return q_da

    # ------------------------------------------------------------------
    # Intraday
    # ------------------------------------------------------------------
    def decide_intraday(
        self,
        info: InfoSet,
        asset_state: AssetState,
        position: Position,
        auction_round: int,
    ) -> np.ndarray:
        n = info.timestamps.shape[0]
        if info.forecast is None or self._last_forecast_solar is None:
            return np.zeros(n)
        sc = info.forecast
        # Align scenario timestamps with the full MTU index
        full_idx = info.timestamps
        sc_idx_in_full = pd.Index(full_idx).get_indexer(sc.timestamps)
        valid = sc_idx_in_full >= 0
        new_solar_p50 = np.median(sc.solar_mw, axis=0)
        delta_solar_full = np.zeros(n)
        for k, j in enumerate(sc_idx_in_full):
            if j < 0 or not valid[k]:
                continue
            delta_solar_full[j] = new_solar_p50[k] - self._last_forecast_solar[j]
        # Trade only when the shift exceeds the threshold
        trade = np.where(np.abs(delta_solar_full) >= self.params.intraday_threshold_mw, delta_solar_full, 0.0)
        # Update last-forecast for the next round
        for k, j in enumerate(sc_idx_in_full):
            if j >= 0:
                self._last_forecast_solar[j] = new_solar_p50[k]
        return trade

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------
    def decide_dispatch(
        self,
        info: InfoSet,
        asset_state: AssetState,
        position: Position,
    ) -> tuple[float, float]:
        t = info.delivery_period_idx
        if t is None or self._planned_charge_mw is None or self._planned_discharge_mw is None:
            return 0.0, 0.0
        bp = self.battery
        dt = self.dt
        realized_solar = float(info.realized_solar_so_far[t])
        if not np.isfinite(realized_solar):
            realized_solar = 0.0
        charge_plan = float(self._planned_charge_mw[t])
        discharge_plan = float(self._planned_discharge_mw[t])
        # Solar-only charging hard cap
        charge = min(charge_plan, realized_solar, bp.p_charge_max_mw)
        # SoC feasibility for both legs
        room_mwh = bp.soc_max_mwh - asset_state.soc_mwh
        charge = min(charge, room_mwh / max(bp.eta_charge * dt, 1e-9))
        charge = max(charge, 0.0)
        avail_mwh = asset_state.soc_mwh - bp.soc_min_mwh
        discharge = min(discharge_plan, bp.p_discharge_max_mw, (avail_mwh * bp.eta_discharge) / max(dt, 1e-9))
        discharge = max(discharge, 0.0)
        # Never charge and discharge in the same step
        if charge > 0 and discharge > 0:
            if charge >= discharge:
                charge -= discharge
                discharge = 0.0
            else:
                discharge -= charge
                charge = 0.0
        return charge, discharge


import pandas as pd  # noqa: E402  - kept at the bottom; only used in decide_intraday
