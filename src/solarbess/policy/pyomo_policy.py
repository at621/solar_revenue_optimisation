"""PyomoPolicy: optimisation-based policy.

`use_stochastic=False` -> Phase 1: deterministic LP on the scenario mean path.
`use_stochastic=True, lambda_cvar=0` -> Phase 2: extensive-form stochastic LP.
`use_stochastic=True, lambda_cvar>0` -> Phase 3: stochastic LP with CVaR + imbalance penalty.

Intraday: re-solves with fixed_q_da (already cleared) and a smaller scenario set
over the remaining tradable periods only.

Dispatch: returns the charge/discharge decision cached from the last solve for
the current period, capped to physical feasibility against realised solar.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pyomo.environ as pyo

from solarbess.asset.battery import AssetState
from solarbess.config import BatteryParams, MarketParams
from solarbess.forecast.scenarios import ScenarioForecast
from solarbess.market.interface import Position
from solarbess.optim.builder import build_deterministic_model, build_stochastic_model
from solarbess.optim.solve import solve_highs
from solarbess.world.oracle import InfoSet


@dataclass
class PyomoPolicyResult:
    q_da: np.ndarray
    charge: np.ndarray
    discharge: np.ndarray
    timestamps: pd.DatetimeIndex


def _scenario_mean(scenarios: ScenarioForecast) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        scenarios.solar_mw.mean(axis=0),
        scenarios.da_price.mean(axis=0),
        scenarios.id_price.mean(axis=0),
        scenarios.imb_surplus.mean(axis=0),
        scenarios.imb_shortfall.mean(axis=0),
    )


class PyomoPolicy:
    name = "pyomo"

    def __init__(
        self,
        battery: BatteryParams,
        market: MarketParams,
        dt_hours: float,
        use_stochastic: bool = True,
        cvar_alpha: float = 0.95,
        lambda_cvar: float = 0.0,
        lambda_imbalance: float = 0.0,
        binary_battery_mode: bool = False,
        mpc_dispatch_horizon: int = 0,
        mpc_dispatch_period: int = 1,
        forecast_gen=None,            # required iff mpc_dispatch_horizon > 0
        forecast_seed: int = 0,
        solver_time_limit_s: float = 30.0,
        name: str | None = None,
    ):
        self.battery = battery
        self.market = market
        self.dt = dt_hours
        self.use_stochastic = use_stochastic
        self.cvar_alpha = cvar_alpha
        self.lambda_cvar = lambda_cvar
        self.lambda_imbalance = lambda_imbalance
        self.binary_battery_mode = binary_battery_mode
        # Phase 5 rolling MPC at dispatch: re-solve every `mpc_dispatch_period` MTUs
        # over the next `mpc_dispatch_horizon` periods with a fresh forecast.
        # 0 disables MPC (uses cached plan from last intraday solve).
        self.mpc_dispatch_horizon = mpc_dispatch_horizon
        self.mpc_dispatch_period = max(1, mpc_dispatch_period)
        self._forecast_gen = forecast_gen
        self._forecast_seed = forecast_seed
        if mpc_dispatch_horizon != 0 and forecast_gen is None:
            raise ValueError("mpc_dispatch_horizon != 0 requires forecast_gen")
        self.solver_time_limit_s = solver_time_limit_s
        if name:
            self.name = name
        else:
            tag = "milp" if binary_battery_mode else "lp"
            mpc_tag = ""
            if mpc_dispatch_horizon != 0:
                h_str = "full" if mpc_dispatch_horizon < 0 else str(mpc_dispatch_horizon)
                mpc_tag = f"_mpc{h_str}p{self.mpc_dispatch_period}"
            if use_stochastic and lambda_cvar > 0:
                self.name = f"pyomo_cvar_{tag}{mpc_tag}"
            elif use_stochastic:
                self.name = f"pyomo_stochastic_{tag}{mpc_tag}"
            else:
                self.name = f"pyomo_deterministic_{tag}{mpc_tag}"
        # Per-episode plan cache: arrays indexed by full MTU
        self._q_da: np.ndarray | None = None
        self._charge_plan: np.ndarray | None = None
        self._discharge_plan: np.ndarray | None = None
        self._full_timestamps: pd.DatetimeIndex | None = None

    # ------------------------------------------------------------------
    def _full_indices(self, info: InfoSet, scen_timestamps: pd.DatetimeIndex) -> np.ndarray:
        full = pd.Index(info.timestamps)
        return full.get_indexer(scen_timestamps)

    # ------------------------------------------------------------------
    def decide_day_ahead(self, info: InfoSet, asset_state: AssetState) -> np.ndarray:
        if info.forecast is None:
            raise ValueError("PyomoPolicy requires a ScenarioForecast at DA gate.")
        sc = info.forecast
        n_mtu = info.timestamps.shape[0]
        # For DA gate, scenarios cover the full delivery day; the indices match.
        idx_in_full = self._full_indices(info, sc.timestamps)

        if self.use_stochastic:
            m = build_stochastic_model(
                times=sc.timestamps,
                scenarios=sc,
                battery=self.battery,
                market=self.market,
                dt_hours=self.dt,
                soc_initial_mwh=asset_state.soc_mwh,
                cvar_alpha=self.cvar_alpha,
                lambda_cvar=self.lambda_cvar,
                lambda_imbalance=self.lambda_imbalance,
                binary_battery_mode=self.binary_battery_mode,
            )
            res = solve_highs(m, time_limit_s=self.solver_time_limit_s)
            if not np.isfinite(res.objective):
                raise RuntimeError(f"Stochastic solve failed: {res.termination}")
            q_da = np.array([pyo.value(m.q_da[t]) for t in m.T])
            # Cache charge/discharge plan for the FIRST scenario as the "expected" dispatch hint.
            scen0 = sc.scenario_ids[0]
            charge = np.array([pyo.value(m.charge[scen0, t]) for t in m.T])
            discharge = np.array([pyo.value(m.discharge[scen0, t]) for t in m.T])
        else:
            mean = _scenario_mean(sc)
            m = build_deterministic_model(
                times=sc.timestamps,
                solar_mw=mean[0],
                da_price_mtu=mean[1],
                id_price_mtu=mean[2],
                imb_surplus=mean[3],
                imb_shortfall=mean[4],
                battery=self.battery,
                market=self.market,
                dt_hours=self.dt,
                soc_initial_mwh=asset_state.soc_mwh,
                binary_battery_mode=self.binary_battery_mode,
            )
            res = solve_highs(m, time_limit_s=self.solver_time_limit_s)
            if not np.isfinite(res.objective):
                raise RuntimeError(f"Deterministic solve failed: {res.termination}")
            q_da = np.array([pyo.value(m.q_da[t]) for t in m.T])
            charge = np.array([pyo.value(m.charge[t]) for t in m.T])
            discharge = np.array([pyo.value(m.discharge[t]) for t in m.T])

        full_q_da = np.zeros(n_mtu)
        full_charge = np.zeros(n_mtu)
        full_discharge = np.zeros(n_mtu)
        full_q_da[idx_in_full] = q_da
        full_charge[idx_in_full] = charge
        full_discharge[idx_in_full] = discharge
        self._q_da = full_q_da
        self._charge_plan = full_charge
        self._discharge_plan = full_discharge
        self._full_timestamps = info.timestamps
        return full_q_da

    # ------------------------------------------------------------------
    def decide_intraday(
        self,
        info: InfoSet,
        asset_state: AssetState,
        position: Position,
        auction_round: int,
    ) -> np.ndarray:
        n_mtu = info.timestamps.shape[0]
        if info.forecast is None or self._q_da is None:
            return np.zeros(n_mtu)
        sc = info.forecast
        idx_in_full = self._full_indices(info, sc.timestamps)
        # Already-committed DA quantity for these periods (passed as parameter to the LP)
        fixed_q = position.da_quantities_mw[idx_in_full]

        # Use deterministic re-solve on the scenario mean for speed; stochastic
        # is also possible but adds little when DA is already locked.
        mean = _scenario_mean(sc)
        m = build_deterministic_model(
            times=sc.timestamps,
            solar_mw=mean[0],
            da_price_mtu=mean[1],
            id_price_mtu=mean[2],
            imb_surplus=mean[3],
            imb_shortfall=mean[4],
            battery=self.battery,
            market=self.market,
            dt_hours=self.dt,
            soc_initial_mwh=asset_state.soc_mwh,
            fixed_q_da=fixed_q,
            enforce_terminal_soc=False,  # intraday re-solve over a tail
            binary_battery_mode=self.binary_battery_mode,
        )
        res = solve_highs(m, time_limit_s=self.solver_time_limit_s)
        if not np.isfinite(res.objective):
            return np.zeros(n_mtu)
        q_id = np.array([pyo.value(m.q_id[t]) for t in m.T])
        # Already-traded intraday quantity for these periods
        already_id = position.id_trades_mw[idx_in_full].sum(axis=1)
        delta = q_id - already_id
        full_trades = np.zeros(n_mtu)
        full_trades[idx_in_full] = delta
        # Refresh cached dispatch plan from this finer re-solve
        if self._charge_plan is not None and self._discharge_plan is not None:
            charge_new = np.array([pyo.value(m.charge[t]) for t in m.T])
            discharge_new = np.array([pyo.value(m.discharge[t]) for t in m.T])
            self._charge_plan[idx_in_full] = charge_new
            self._discharge_plan[idx_in_full] = discharge_new
        return full_trades

    # ------------------------------------------------------------------
    def _rolling_mpc_dispatch(
        self,
        info: InfoSet,
        asset_state: AssetState,
        position: Position,
        t: int,
    ) -> tuple[float, float]:
        """Re-solve a shrinking-tail problem over the next H MTUs with a fresh
        forecast made at `info.now`, fixed past commitments, and current SoC.
        Returns this-period (charge, discharge) and updates the cached plan
        for downstream periods.

        mpc_dispatch_horizon = -1 means "to end of day" (full remaining horizon).
        Short horizons need a terminal SoC value to avoid greedy depletion; we
        re-enforce the terminal SoC constraint only when the tail reaches the
        last delivery period.
        """
        H = self.mpc_dispatch_horizon
        n_mtu = info.timestamps.shape[0]
        end = n_mtu if H < 0 else min(t + H, n_mtu)
        tail_ts = info.timestamps[t:end]
        if len(tail_ts) == 0:
            return 0.0, 0.0
        # Truth is sealed inside the oracle; for MPC we pass through the oracle
        # to get a fresh forecast for the next H periods only.
        sc = self._forecast_gen.generate(
            truth=self._mpc_truth,
            now=info.now,
            decision_time=info.decision_time,
            n_scenarios=max(8, self.mpc_dispatch_horizon * 2),
            seed=self._forecast_seed + 7919 * t,
        )
        # Trim scenarios to the tail window
        idx = pd.Index(sc.timestamps).get_indexer(tail_ts)
        keep = idx >= 0
        idx = idx[keep]
        if len(idx) == 0:
            return 0.0, 0.0
        tail_ts = tail_ts[keep]
        mean = (
            sc.solar_mw[:, idx].mean(axis=0),
            sc.da_price[:, idx].mean(axis=0),
            sc.id_price[:, idx].mean(axis=0),
            sc.imb_surplus[:, idx].mean(axis=0),
            sc.imb_shortfall[:, idx].mean(axis=0),
        )
        fixed_q = position.da_quantities_mw[t:t + len(tail_ts)]
        # Only enforce terminal SoC if our tail actually reaches the end of the day.
        reaches_end = (t + len(tail_ts)) >= n_mtu
        m = build_deterministic_model(
            times=tail_ts,
            solar_mw=mean[0],
            da_price_mtu=mean[1],
            id_price_mtu=mean[2],
            imb_surplus=mean[3],
            imb_shortfall=mean[4],
            battery=self.battery,
            market=self.market,
            dt_hours=self.dt,
            soc_initial_mwh=asset_state.soc_mwh,
            fixed_q_da=fixed_q,
            enforce_terminal_soc=reaches_end,
            binary_battery_mode=self.binary_battery_mode,
        )
        res = solve_highs(m, time_limit_s=self.solver_time_limit_s)
        if not np.isfinite(res.objective):
            return 0.0, 0.0
        # Use the FIRST tail period as our dispatch decision
        first_t = list(m.T)[0]
        charge = float(pyo.value(m.charge[first_t]))
        discharge = float(pyo.value(m.discharge[first_t]))
        return charge, discharge

    def attach_mpc_truth(self, truth) -> None:
        """The MPC dispatch needs to ask the forecast generator for fresh scenarios.
        The simulator passes the sealed TruthPath here at episode start - this is
        kept private and the policy still cannot peek beyond `now` because the
        forecast generator only emits scenarios for periods >= now.
        """
        self._mpc_truth = truth

    def decide_dispatch(
        self,
        info: InfoSet,
        asset_state: AssetState,
        position: Position,
    ) -> tuple[float, float]:
        t = info.delivery_period_idx
        if t is None:
            return 0.0, 0.0
        bp = self.battery
        dt = self.dt
        realized_solar = float(info.realized_solar_so_far[t])
        if not np.isfinite(realized_solar):
            realized_solar = 0.0

        if (
            self.mpc_dispatch_horizon != 0
            and getattr(self, "_mpc_truth", None) is not None
            and (t % self.mpc_dispatch_period == 0)
        ):
            charge, discharge = self._rolling_mpc_dispatch(info, asset_state, position, t)
            # Cache as plan for the next `mpc_dispatch_period - 1` periods
            if self._charge_plan is None:
                self._charge_plan = np.zeros(info.timestamps.shape[0])
                self._discharge_plan = np.zeros(info.timestamps.shape[0])
            self._charge_plan[t] = charge
            self._discharge_plan[t] = discharge
        elif self._charge_plan is None or self._discharge_plan is None:
            return 0.0, 0.0
        else:
            charge = float(self._charge_plan[t])
            discharge = float(self._discharge_plan[t])

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
