"""Phase 6: TreePolicy - multi-stage stochastic via explicit scenario tree.

Difference from PyomoPolicy(use_stochastic=True):
- q_DA is the root decision (same as before).
- q_ID is decomposed by IDA-node (a coarse stage-1 information branch).
  Scenarios sharing the same IDA branch share the same q_ID -> non-anticipativity.
- Dispatch is leaf-specific.

The IDA-node decomposition acts as a richer scenario aggregation than the
two-stage model (which has a separate q_ID per leaf scenario).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyomo.environ as pyo

from solarbess.asset.battery import AssetState
from solarbess.config import BatteryParams, MarketParams
from solarbess.forecast.tree import build_tree_from_scenarios
from solarbess.market.interface import Position
from solarbess.optim.builder import build_deterministic_model
from solarbess.optim.solve import solve_highs
from solarbess.optim.tree_builder import build_tree_stochastic_model
from solarbess.world.oracle import InfoSet


class TreePolicy:
    def __init__(
        self,
        battery: BatteryParams,
        market: MarketParams,
        dt_hours: float,
        n_ida_nodes: int = 3,
        stage1_feature: str = "da_price_first_half",
        cvar_alpha: float = 0.95,
        lambda_cvar: float = 0.0,
        lambda_imbalance: float = 0.0,
        binary_battery_mode: bool = False,
        solver_time_limit_s: float = 60.0,
        name: str | None = None,
    ):
        self.battery = battery
        self.market = market
        self.dt = dt_hours
        self.n_ida_nodes = n_ida_nodes
        self.stage1_feature = stage1_feature
        self.cvar_alpha = cvar_alpha
        self.lambda_cvar = lambda_cvar
        self.lambda_imbalance = lambda_imbalance
        self.binary_battery_mode = binary_battery_mode
        self.solver_time_limit_s = solver_time_limit_s
        self.name = name or f"tree_ida{n_ida_nodes}"
        self._q_da: np.ndarray | None = None
        self._charge_plan: np.ndarray | None = None
        self._discharge_plan: np.ndarray | None = None
        self._tree = None
        self._tree_q_id_by_ida: np.ndarray | None = None

    def _full_indices(self, info: InfoSet, scen_timestamps: pd.DatetimeIndex) -> np.ndarray:
        return pd.Index(info.timestamps).get_indexer(scen_timestamps)

    def decide_day_ahead(self, info: InfoSet, asset_state: AssetState) -> np.ndarray:
        if info.forecast is None:
            raise ValueError("TreePolicy requires a ScenarioForecast at DA gate.")
        sc = info.forecast
        n_mtu = info.timestamps.shape[0]
        idx_in_full = self._full_indices(info, sc.timestamps)

        tree = build_tree_from_scenarios(
            sc, n_ida_nodes=self.n_ida_nodes, stage1_feature=self.stage1_feature
        )
        m = build_tree_stochastic_model(
            times=sc.timestamps,
            tree=tree,
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
            raise RuntimeError(f"Tree solve failed: {res.termination}")

        q_da = np.array([pyo.value(m.q_da[t]) for t in m.T])
        # Pick the IDA branch with highest probability as the dispatch hint
        best_ida = int(tree.ida_node_ids[int(np.argmax(tree.ida_probabilities))])
        # Pick one representative leaf in that IDA branch
        leaf_mask = tree.leaf_to_ida == best_ida
        rep_leaf = int(tree.leaf_ids[np.argmax(leaf_mask)])
        charge = np.array([pyo.value(m.charge[rep_leaf, t]) for t in m.T])
        discharge = np.array([pyo.value(m.discharge[rep_leaf, t]) for t in m.T])
        # Cache q_id_ida (per IDA node) for the intraday callback
        n_ida = len(tree.ida_node_ids)
        q_id_by_ida = np.array(
            [[pyo.value(m.q_id_ida[ida, t]) for t in m.T] for ida in tree.ida_node_ids]
        )

        full_q_da = np.zeros(n_mtu)
        full_charge = np.zeros(n_mtu)
        full_discharge = np.zeros(n_mtu)
        full_q_da[idx_in_full] = q_da
        full_charge[idx_in_full] = charge
        full_discharge[idx_in_full] = discharge

        self._q_da = full_q_da
        self._charge_plan = full_charge
        self._discharge_plan = full_discharge
        self._tree = tree
        # Average q_id across IDA branches weighted by probability - this becomes
        # the policy's intraday recommendation given no further info.
        self._tree_q_id_by_ida = q_id_by_ida
        self._best_ida = best_ida
        self._idx_in_full_da = idx_in_full
        return full_q_da

    def decide_intraday(
        self,
        info: InfoSet,
        asset_state: AssetState,
        position: Position,
        auction_round: int,
    ) -> np.ndarray:
        """For intraday rounds, re-solve a deterministic LP on the scenario MEAN
        with q_da fixed. The tree's q_id is only used as a soft guide; the actual
        intraday is the LP's q_id minus what's already been traded.
        """
        n_mtu = info.timestamps.shape[0]
        if info.forecast is None or self._q_da is None:
            return np.zeros(n_mtu)
        sc = info.forecast
        idx_in_full = self._full_indices(info, sc.timestamps)
        fixed_q = position.da_quantities_mw[idx_in_full]
        mean = (
            sc.solar_mw.mean(axis=0),
            sc.da_price.mean(axis=0),
            sc.id_price.mean(axis=0),
            sc.imb_surplus.mean(axis=0),
            sc.imb_shortfall.mean(axis=0),
        )
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
            enforce_terminal_soc=False,
            binary_battery_mode=self.binary_battery_mode,
        )
        res = solve_highs(m, time_limit_s=self.solver_time_limit_s)
        if not np.isfinite(res.objective):
            return np.zeros(n_mtu)
        q_id = np.array([pyo.value(m.q_id[t]) for t in m.T])
        already = position.id_trades_mw[idx_in_full].sum(axis=1)
        delta = q_id - already
        full_trades = np.zeros(n_mtu)
        full_trades[idx_in_full] = delta
        # Refresh dispatch plan
        if self._charge_plan is not None and self._discharge_plan is not None:
            ch_new = np.array([pyo.value(m.charge[t]) for t in m.T])
            dis_new = np.array([pyo.value(m.discharge[t]) for t in m.T])
            self._charge_plan[idx_in_full] = ch_new
            self._discharge_plan[idx_in_full] = dis_new
        return full_trades

    def decide_dispatch(
        self,
        info: InfoSet,
        asset_state: AssetState,
        position: Position,
    ) -> tuple[float, float]:
        t = info.delivery_period_idx
        if t is None or self._charge_plan is None or self._discharge_plan is None:
            return 0.0, 0.0
        bp = self.battery
        dt = self.dt
        realized_solar = float(info.realized_solar_so_far[t])
        if not np.isfinite(realized_solar):
            realized_solar = 0.0
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
