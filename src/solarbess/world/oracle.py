"""Information Oracle - the only legal channel between the world and the policy.

The InfoSet returned at each decision time contains:
- realized values up to `now` (and ONLY up to `now`)
- a freshly-generated ScenarioForecast over the remaining tradable horizon
- already-cleared DA quantities, if DA gate has passed

The Oracle never returns truth beyond `now` (except at DELIVERY for the period
being dispatched, where realised solar is required for the asset model).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from solarbess.forecast.scenarios import ForecastGenerator, ScenarioForecast
    from solarbess.world.truth import TruthPath


class DecisionTime(str, Enum):
    DA_GATE = "DA_GATE"
    IDA1 = "IDA1"
    IDA2 = "IDA2"
    IDA3 = "IDA3"
    DELIVERY = "DELIVERY"


@dataclass(frozen=True)
class InfoSet:
    decision_time: DecisionTime
    now: pd.Timestamp
    realized_solar_so_far: np.ndarray         # truth values up to `now` only; NaN beyond
    realized_da_price_so_far: np.ndarray
    realized_id_prices_so_far: np.ndarray     # shape (mtu, n_rounds); NaN where not yet cleared
    realized_imb_surplus_so_far: np.ndarray
    realized_imb_shortfall_so_far: np.ndarray
    timestamps: pd.DatetimeIndex
    forecast: "ScenarioForecast | None"
    da_clearing_known: bool
    delivery_period_idx: int | None = None     # only set at DELIVERY


class InformationOracle:
    """Wraps a sealed TruthPath. Exposes views via `view(...)`.

    The policy is granted only the InfoSet returned by `view`; it must never
    receive the TruthPath itself. The OraclePolicy is the only intentional
    exception (it bypasses this class entirely).
    """

    def __init__(self, truth: "TruthPath", forecast_generator: "ForecastGenerator | None"):
        self._truth = truth
        self._fc = forecast_generator
        self._n_mtu = len(truth.timestamps)
        self._mtu_minutes = int(
            (truth.timestamps[1] - truth.timestamps[0]) / pd.Timedelta(minutes=1)
        )

    @property
    def n_mtu(self) -> int:
        return self._n_mtu

    @property
    def timestamps(self) -> pd.DatetimeIndex:
        return self._truth.timestamps

    def _mask_arrays(self, now: pd.Timestamp) -> dict[str, np.ndarray]:
        """Return realised arrays with values BEYOND `now` masked as NaN."""
        ts = self._truth.timestamps
        revealed = np.asarray(ts < now)
        n_rounds = self._truth.id_price_mtu.shape[1]
        out: dict[str, np.ndarray] = {}
        out["solar"] = np.where(revealed, self._truth.solar_mw, np.nan)
        out["da"] = np.where(revealed, self._truth.da_price_mtu, np.nan)
        id_mask = np.broadcast_to(revealed[:, None], (self._n_mtu, n_rounds)).copy()
        out["id"] = np.where(id_mask, self._truth.id_price_mtu, np.nan)
        out["imb_surplus"] = np.where(revealed, self._truth.imb_surplus_price, np.nan)
        out["imb_shortfall"] = np.where(revealed, self._truth.imb_shortfall_price, np.nan)
        return out

    def view(
        self,
        decision_time: DecisionTime,
        now: pd.Timestamp,
        n_scenarios: int | None = None,
        forecast_seed: int | None = None,
        delivery_period_idx: int | None = None,
    ) -> InfoSet:
        masked = self._mask_arrays(now)
        forecast = None
        if self._fc is not None and decision_time != DecisionTime.DELIVERY:
            n = n_scenarios if n_scenarios is not None else 50
            forecast = self._fc.generate(
                truth=self._truth,
                now=now,
                decision_time=decision_time,
                n_scenarios=n,
                seed=forecast_seed if forecast_seed is not None else 0,
            )
        # At DELIVERY for the current period, the asset model needs realized solar
        # for THAT period. Expose it through `realized_solar_so_far[period]`.
        if decision_time == DecisionTime.DELIVERY and delivery_period_idx is not None:
            masked["solar"][delivery_period_idx] = self._truth.solar_mw[delivery_period_idx]
        return InfoSet(
            decision_time=decision_time,
            now=now,
            realized_solar_so_far=masked["solar"],
            realized_da_price_so_far=masked["da"],
            realized_id_prices_so_far=masked["id"],
            realized_imb_surplus_so_far=masked["imb_surplus"],
            realized_imb_shortfall_so_far=masked["imb_shortfall"],
            timestamps=self._truth.timestamps,
            forecast=forecast,
            da_clearing_known=(decision_time != DecisionTime.DA_GATE),
            delivery_period_idx=delivery_period_idx,
        )
