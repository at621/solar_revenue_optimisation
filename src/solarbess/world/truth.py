"""TruthGenerator: produce one sealed `TruthPath` per delivery day.

Generated up-front and never modified - the policy must access realized values
only through `InformationOracle.view(decision_time)`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from solarbess.clock import Clock
from solarbess.config import AppConfig
from solarbess.world.layers import (
    clear_sky_envelope,
    project_solar_mw,
    sample_clearness,
    sample_da_price_hourly,
    sample_imbalance_prices,
    sample_intraday_prices,
    sample_regime,
    seasonality_template,
)


@dataclass(frozen=True)
class TruthPath:
    delivery_date: date
    timestamps: pd.DatetimeIndex   # UTC, length = mtus_per_day
    solar_mw: np.ndarray           # (mtu,)
    da_price_hourly: np.ndarray    # (24,) EUR/MWh
    da_price_mtu: np.ndarray       # (mtu,) hourly broadcast for settlement
    id_price_mtu: np.ndarray       # (mtu, n_rounds)
    imb_surplus_price: np.ndarray  # (mtu,)
    imb_shortfall_price: np.ndarray  # (mtu,)
    latent: dict[str, Any]         # regime, clearness, envelope, shocks - diagnostics only

    def __post_init__(self) -> None:
        n = len(self.timestamps)
        if self.solar_mw.shape != (n,):
            raise ValueError(f"solar_mw shape {self.solar_mw.shape} != ({n},)")
        if self.da_price_mtu.shape != (n,):
            raise ValueError(f"da_price_mtu shape mismatch")
        if self.imb_surplus_price.shape != (n,) or self.imb_shortfall_price.shape != (n,):
            raise ValueError("imbalance price shape mismatch")
        if self.id_price_mtu.ndim != 2 or self.id_price_mtu.shape[0] != n:
            raise ValueError("id_price_mtu shape mismatch")


class TruthGenerator:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg

    def sample(self, delivery_date: date, seed: int) -> TruthPath:
        cfg = self.cfg
        clock = Clock(
            delivery_date=delivery_date,
            mtu_minutes=cfg.mtu_minutes,
            intraday_lead_times_h=tuple(cfg.market.intraday_lead_times_h),
        )
        clock.assert_non_dst()
        rng = np.random.default_rng(seed)
        times = clock.mtu_index()
        n_mtu = len(times)
        mtus_per_hour = clock.mtus_per_hour

        # L1
        regime = sample_regime(rng, n_mtu, cfg.world.l1_regime)
        seas = seasonality_template(times, cfg.world.l1_seasonality)

        # L2
        envelope = clear_sky_envelope(times, cfg.world.l2_solar)
        clearness, _innov = sample_clearness(
            rng,
            times,
            cfg.world.l2_solar,
            regime,
            cfg.world.l1_regime.stressed_shock_scale,
            cfg.dt_hours,
        )
        solar_mw = project_solar_mw(clearness, envelope, cfg.world.l2_solar.peak_mw)

        # L3
        da_hourly, da_shock_hourly = sample_da_price_hourly(
            rng,
            solar_mw,
            seas,
            regime,
            cfg.world.l3_da,
            cfg.world.l2_solar.peak_mw,
            mtus_per_hour,
        )
        da_mtu = clock.broadcast_hourly_to_mtu(da_hourly)

        # L4
        id_mtu = sample_intraday_prices(
            rng,
            da_mtu,
            da_shock_hourly,
            cfg.world.l4_id,
            mtus_per_hour,
            n_rounds=len(cfg.market.intraday_lead_times_h),
        )

        # L5 - solar anomaly used as proxy for system imbalance state
        expected_solar = envelope * cfg.world.l2_solar.peak_mw * (
            (cfg.world.l2_solar.clearness_mean_summer + cfg.world.l2_solar.clearness_mean_winter)
            / 2.0
        )
        solar_anomaly = solar_mw - expected_solar
        imb_surplus, imb_shortfall = sample_imbalance_prices(
            rng, da_mtu, solar_anomaly, cfg.world.l5_imbalance
        )

        return TruthPath(
            delivery_date=delivery_date,
            timestamps=times,
            solar_mw=solar_mw,
            da_price_hourly=da_hourly,
            da_price_mtu=da_mtu,
            id_price_mtu=id_mtu,
            imb_surplus_price=imb_surplus,
            imb_shortfall_price=imb_shortfall,
            latent={
                "regime": regime,
                "clearness": clearness,
                "envelope": envelope,
                "seasonality": seas,
                "da_shock_hourly": da_shock_hourly,
                "solar_anomaly": solar_anomaly,
            },
        )
