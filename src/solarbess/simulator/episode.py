"""Single-episode loop.

Daily clock:
1. DA gate (12:00 CET on D-1): policy submits q_DA for every delivery period.
2. For each intraday round r in order: policy submits trade deltas; trades are
   cleared at the round's per-period price; position accumulates.
3. For each delivery period: policy decides battery dispatch given realised
   solar; asset state advances; accountant settles the period.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from solarbess.asset.battery import AssetModel, AssetState
from solarbess.clock import Clock
from solarbess.config import AppConfig
from solarbess.market.interface import Position
from solarbess.market.settlement import EpisodePnL, ProfitAccountant
from solarbess.world.oracle import DecisionTime, InformationOracle
from solarbess.world.truth import TruthPath

if TYPE_CHECKING:
    from solarbess.forecast.scenarios import ForecastGenerator
    from solarbess.policy.base import Policy


@dataclass
class EpisodeResult:
    delivery_date: date
    policy_name: str
    pnl: EpisodePnL
    timeseries: pd.DataFrame
    final_soc_mwh: float
    cumulative_throughput_mwh: float


def run_episode(
    truth: TruthPath,
    cfg: AppConfig,
    policy: "Policy",
    forecast_gen: "ForecastGenerator | None",
    forecast_seed: int,
) -> EpisodeResult:
    clock = Clock(
        delivery_date=truth.delivery_date,
        mtu_minutes=cfg.mtu_minutes,
        intraday_lead_times_h=tuple(cfg.market.intraday_lead_times_h),
    )
    n_mtu = clock.mtus_per_day
    n_rounds = len(cfg.market.intraday_lead_times_h)

    oracle = InformationOracle(truth, forecast_gen)
    asset_model = AssetModel(
        battery=cfg.battery,
        pv_peak_mw=cfg.world.l2_solar.peak_mw,
        inverter_limit_mw=cfg.world.l2_solar.peak_mw * 1.1,
    )
    accountant = ProfitAccountant(cfg.market, cfg.battery, dt_hours=cfg.dt_hours)
    position = Position(n_mtu=n_mtu, n_rounds=n_rounds)
    state = AssetState(soc_mwh=cfg.battery.soc_initial_mwh)

    # Phase 5: MPC-enabled policies need the truth handle to ask the forecast
    # generator for fresh scenarios at each dispatch step. The information
    # firewall is preserved because the generator only emits scenarios for
    # periods >= now.
    if hasattr(policy, "attach_mpc_truth"):
        policy.attach_mpc_truth(truth)

    # 1. DA gate
    da_info = oracle.view(
        DecisionTime.DA_GATE,
        now=clock.da_gate_utc(),
        n_scenarios=cfg.forecast.n_scenarios_da,
        forecast_seed=forecast_seed,
    )
    q_da = np.asarray(policy.decide_day_ahead(da_info, state), dtype=float)
    if q_da.shape != (n_mtu,):
        raise ValueError(f"decide_day_ahead returned shape {q_da.shape}, expected ({n_mtu},)")
    q_da = np.clip(q_da, -cfg.market.import_limit_mw, cfg.market.export_limit_mw)
    position.set_day_ahead(q_da)

    # 2. Intraday rounds
    for r, lead_h in enumerate(cfg.market.intraday_lead_times_h):
        # Use the first delivery period's deadline as the round's "now"
        round_now = clock.mtu_index()[0] - pd.Timedelta(hours=lead_h)
        info = oracle.view(
            DecisionTime[f"IDA{r+1}"],
            now=round_now,
            n_scenarios=cfg.forecast.n_scenarios_id,
            forecast_seed=forecast_seed + 1000 + r,
        )
        trades = np.asarray(policy.decide_intraday(info, state, position, r), dtype=float)
        if trades.shape != (n_mtu,):
            raise ValueError(f"decide_intraday returned shape {trades.shape}, expected ({n_mtu},)")
        # Zero out trades for periods that have already passed (closer than lead_h)
        mtus = clock.mtu_index()
        tradable = mtus >= round_now + pd.Timedelta(hours=lead_h)
        trades = np.where(tradable, trades, 0.0)
        position.add_intraday(trades, r)

    # 3. Delivery + dispatch
    period_pnls = []
    timeseries_rows: list[dict] = []
    mtus = clock.mtu_index()
    for t in range(n_mtu):
        info = oracle.view(
            DecisionTime.DELIVERY,
            now=mtus[t] + pd.Timedelta(minutes=cfg.mtu_minutes),
            delivery_period_idx=t,
        )
        charge, discharge = policy.decide_dispatch(info, state, position)
        new_state, injection = asset_model.step(
            state=state,
            charge_mw=charge,
            discharge_mw=discharge,
            realized_solar_mw=float(truth.solar_mw[t]),
            dt_hours=cfg.dt_hours,
        )
        pnl = accountant.settle_period(
            period_idx=t,
            position=position,
            physical_mw=injection,
            truth=truth,
            charge_mw=charge,
            discharge_mw=discharge,
            realized_solar_mw=float(truth.solar_mw[t]),
        )
        period_pnls.append(pnl)
        timeseries_rows.append(
            {
                "timestamp": mtus[t],
                "q_da_mw": position.da_quantities_mw[t],
                "q_id_mw": position.id_trades_mw[t].sum(),
                "committed_mw": position.committed_mw[t],
                "physical_mw": injection,
                "realized_solar_mw": truth.solar_mw[t],
                "charge_mw": charge,
                "discharge_mw": discharge,
                "soc_mwh": new_state.soc_mwh,
                "da_price": truth.da_price_mtu[t],
                "imb_surplus_price": truth.imb_surplus_price[t],
                "imb_shortfall_price": truth.imb_shortfall_price[t],
                "imb_pos_mwh": pnl.imb_pos_mwh,
                "imb_neg_mwh": pnl.imb_neg_mwh,
                "period_pnl": pnl.total,
            }
        )
        state = new_state

    return EpisodeResult(
        delivery_date=truth.delivery_date,
        policy_name=policy.name,
        pnl=accountant.aggregate(period_pnls),
        timeseries=pd.DataFrame(timeseries_rows),
        final_soc_mwh=state.soc_mwh,
        cumulative_throughput_mwh=state.cumulative_throughput_mwh,
    )
