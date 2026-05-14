"""Profit accountant: per-period PnL decomposition and episode aggregation.

Sign convention (MUST match Pyomo objective in optim/builder.py):
- DA revenue:    q_DA * dt * pi_DA          (positive q = export)
- ID revenue:    sum_r q_ID[r] * dt * pi_ID[r]
- Imbalance:     +pi_surplus  * imb_pos * dt   (surplus paid at surplus price)
                 -pi_shortfall * imb_neg * dt  (shortfall charged at shortfall price)
   where physical - schedule == imb_pos - imb_neg, both >= 0.
- Degradation:   degradation_cost * (charge + discharge) * dt
- Fees:          fees * (|q_DA| + sum_r |q_ID[r]|) * dt
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from solarbess.config import BatteryParams, MarketParams
from solarbess.market.interface import Position
from solarbess.world.truth import TruthPath


@dataclass(frozen=True)
class PeriodPnL:
    period_idx: int
    da_revenue: float
    id_revenue: float
    imbalance_revenue: float
    degradation_cost: float
    fees: float
    schedule_mw: float
    physical_mw: float
    imb_pos_mwh: float
    imb_neg_mwh: float
    charge_mw: float
    discharge_mw: float
    realized_solar_mw: float

    @property
    def total(self) -> float:
        return (
            self.da_revenue
            + self.id_revenue
            + self.imbalance_revenue
            - self.degradation_cost
            - self.fees
        )


@dataclass(frozen=True)
class EpisodePnL:
    period_pnls: list[PeriodPnL]
    da_revenue: float
    id_revenue: float
    imbalance_revenue: float
    degradation_cost: float
    fees: float

    @property
    def total(self) -> float:
        return (
            self.da_revenue
            + self.id_revenue
            + self.imbalance_revenue
            - self.degradation_cost
            - self.fees
        )


class ProfitAccountant:
    def __init__(self, market: MarketParams, battery: BatteryParams, dt_hours: float):
        self.market = market
        self.battery = battery
        self.dt = dt_hours

    def settle_period(
        self,
        period_idx: int,
        position: Position,
        physical_mw: float,
        truth: TruthPath,
        charge_mw: float,
        discharge_mw: float,
        realized_solar_mw: float,
    ) -> PeriodPnL:
        dt = self.dt
        q_da = position.da_quantities_mw[period_idx]
        q_id_rounds = position.id_trades_mw[period_idx]
        schedule = q_da + q_id_rounds.sum()

        da_rev = q_da * truth.da_price_mtu[period_idx] * dt
        id_rev = float((q_id_rounds * truth.id_price_mtu[period_idx]).sum() * dt)

        delta = physical_mw - schedule
        imb_pos = max(delta, 0.0)
        imb_neg = max(-delta, 0.0)
        imb_rev = (
            truth.imb_surplus_price[period_idx] * imb_pos * dt
            - truth.imb_shortfall_price[period_idx] * imb_neg * dt
        )

        degradation = self.battery.degradation_cost_eur_per_mwh * (charge_mw + discharge_mw) * dt
        fees = self.market.fees_eur_per_mwh * (abs(q_da) + float(np.abs(q_id_rounds).sum())) * dt

        return PeriodPnL(
            period_idx=period_idx,
            da_revenue=da_rev,
            id_revenue=id_rev,
            imbalance_revenue=imb_rev,
            degradation_cost=degradation,
            fees=fees,
            schedule_mw=schedule,
            physical_mw=physical_mw,
            imb_pos_mwh=imb_pos * dt,
            imb_neg_mwh=imb_neg * dt,
            charge_mw=charge_mw,
            discharge_mw=discharge_mw,
            realized_solar_mw=realized_solar_mw,
        )

    def aggregate(self, period_pnls: list[PeriodPnL]) -> EpisodePnL:
        return EpisodePnL(
            period_pnls=period_pnls,
            da_revenue=sum(p.da_revenue for p in period_pnls),
            id_revenue=sum(p.id_revenue for p in period_pnls),
            imbalance_revenue=sum(p.imbalance_revenue for p in period_pnls),
            degradation_cost=sum(p.degradation_cost for p in period_pnls),
            fees=sum(p.fees for p in period_pnls),
        )
