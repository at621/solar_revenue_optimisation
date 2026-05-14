"""Pydantic configuration models with YAML loading.

`AppConfig.from_yaml(path)` validates the YAML and returns a strongly-typed
config object that is the single source of truth for all simulator parameters.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class BatteryParams(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    capacity_mwh: float = Field(gt=0)
    p_charge_max_mw: float = Field(gt=0)
    p_discharge_max_mw: float = Field(gt=0)
    eta_charge: float = Field(gt=0, le=1)
    eta_discharge: float = Field(gt=0, le=1)
    soc_min_mwh: float = Field(ge=0)
    soc_max_mwh: float = Field(gt=0)
    soc_initial_mwh: float = Field(ge=0)
    soc_terminal_min_mwh: float = Field(ge=0)
    degradation_cost_eur_per_mwh: float = Field(ge=0)

    @model_validator(mode="after")
    def _check_bounds(self) -> "BatteryParams":
        if not (self.soc_min_mwh <= self.soc_initial_mwh <= self.soc_max_mwh <= self.capacity_mwh):
            raise ValueError(
                f"SoC bounds invalid: min={self.soc_min_mwh} initial={self.soc_initial_mwh} "
                f"max={self.soc_max_mwh} capacity={self.capacity_mwh}"
            )
        if self.soc_terminal_min_mwh > self.soc_max_mwh:
            raise ValueError("soc_terminal_min_mwh exceeds soc_max_mwh")
        return self


class MarketParams(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    export_limit_mw: float = Field(gt=0)
    import_limit_mw: float = Field(ge=0)
    fees_eur_per_mwh: float = Field(default=0.0, ge=0)
    da_gate_local_time: str = "12:00"
    intraday_lead_times_h: list[int] = Field(default_factory=lambda: [6, 3, 1])

    @model_validator(mode="after")
    def _check_lead_times(self) -> "MarketParams":
        if sorted(self.intraday_lead_times_h, reverse=True) != list(self.intraday_lead_times_h):
            raise ValueError("intraday_lead_times_h must be in descending order (earliest first)")
        if any(h <= 0 for h in self.intraday_lead_times_h):
            raise ValueError("intraday lead times must be positive hours")
        return self


class L1Seasonality(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    annual_amplitude: float = 0.18
    annual_phase_doy: int = 15
    weekly_load_template: list[list[float]] = Field(
        default_factory=lambda: [[1.0] * 96 for _ in range(7)]
    )

    @model_validator(mode="after")
    def _shape_check(self) -> "L1Seasonality":
        if len(self.weekly_load_template) != 7 or any(
            len(row) != 96 for row in self.weekly_load_template
        ):
            raise ValueError("weekly_load_template must be 7x96")
        return self


class L1Regime(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    p_calm_to_stressed: float = Field(default=0.03, ge=0, le=1)
    p_stressed_to_calm: float = Field(default=0.10, ge=0, le=1)
    stressed_shock_scale: float = Field(default=2.5, gt=0)
    initial_state: int = Field(default=0, ge=0, le=1)


class L2Solar(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    peak_mw: float = Field(gt=0, default=50.0)
    latitude_deg: float = 58.7
    clearness_mean_summer: float = Field(default=0.65, gt=0, lt=1)
    clearness_mean_winter: float = Field(default=0.45, gt=0, lt=1)
    clearness_ou_theta: float = Field(default=4.0, gt=0)
    clearness_ou_sigma: float = Field(default=0.35, gt=0)
    cloudy_day_prob: float = Field(default=0.20, ge=0, le=1)
    cloudy_day_extra_sigma: float = Field(default=0.20, ge=0)


class L3DAPrice(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    base_eur_mwh: float = 65.0
    beta_load: float = 18.0
    beta_gas: float = 12.0
    shock_df: int = Field(default=4, gt=2)
    shock_sigma_calm: float = Field(default=8.0, gt=0)
    shock_sigma_stressed: float = Field(default=22.0, gt=0)
    gas_co2_state: float = 0.0
    negative_price_floor: float = -100.0


class L4IDPrice(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    spread_sigma_by_round: list[float] = Field(default_factory=lambda: [6.0, 4.0, 2.5])
    spread_ar1_phi: float = Field(default=0.4, ge=-1, le=1)
    spread_da_shock_corr: float = Field(default=0.3, ge=-1, le=1)


class L5Imbalance(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    markup_long_surplus: float = Field(default=8.0, ge=0)
    markup_short_surplus: float = Field(default=18.0, ge=0)
    markup_long_shortfall: float = Field(default=5.0, ge=0)
    markup_short_shortfall: float = Field(default=25.0, ge=0)
    aggregate_solar_correlation: float = Field(default=0.6, ge=-1, le=1)


class WorldConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    l1_regime: L1Regime = Field(default_factory=L1Regime)
    l1_seasonality: L1Seasonality = Field(default_factory=L1Seasonality)
    l2_solar: L2Solar = Field(default_factory=L2Solar)
    l3_da: L3DAPrice = Field(default_factory=L3DAPrice)
    l4_id: L4IDPrice = Field(default_factory=L4IDPrice)
    l5_imbalance: L5Imbalance = Field(default_factory=L5Imbalance)


class ForecastConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    n_scenarios_da: int = Field(default=50, gt=0)
    n_scenarios_id: int = Field(default=30, gt=0)
    # Lead-time-scaled variance: sigma(dt) = floor + (max - floor) * (1 - exp(-dt / tau))
    solar_sigma_floor: float = Field(default=0.5, ge=0)
    solar_sigma_max: float = Field(default=8.0, gt=0)
    solar_tau_lead_h: float = Field(default=12.0, gt=0)
    solar_corr_tau_h: float = Field(default=2.0, gt=0)
    da_sigma_floor: float = Field(default=1.0, ge=0)
    da_sigma_max: float = Field(default=25.0, gt=0)
    da_tau_lead_h: float = Field(default=18.0, gt=0)
    da_corr_tau_h: float = Field(default=3.0, gt=0)
    id_sigma_floor: float = Field(default=0.5, ge=0)
    id_sigma_max: float = Field(default=15.0, gt=0)
    id_tau_lead_h: float = Field(default=6.0, gt=0)
    id_corr_tau_h: float = Field(default=1.5, gt=0)
    imb_sigma_floor: float = Field(default=2.0, ge=0)
    imb_sigma_max: float = Field(default=35.0, gt=0)
    imb_tau_lead_h: float = Field(default=4.0, gt=0)
    imb_corr_tau_h: float = Field(default=1.0, gt=0)
    # Cross-variable correlations applied at sampling time
    solar_da_corr: float = Field(default=-0.45, ge=-1, le=1)


class CVaRConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    alpha: float = Field(default=0.95, gt=0, lt=1)
    lambda_cvar: float = Field(default=0.0, ge=0)
    lambda_imbalance: float = Field(default=0.0, ge=0)


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    battery: BatteryParams
    market: MarketParams
    world: WorldConfig = Field(default_factory=WorldConfig)
    forecast: ForecastConfig = Field(default_factory=ForecastConfig)
    cvar: CVaRConfig = Field(default_factory=CVaRConfig)
    mtu_minutes: int = Field(default=15, gt=0)

    @model_validator(mode="after")
    def _check_mtu(self) -> "AppConfig":
        if 60 % self.mtu_minutes != 0:
            raise ValueError("mtu_minutes must divide 60")
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AppConfig":
        with open(path, encoding="utf-8") as f:
            raw: dict[str, Any] = yaml.safe_load(f)
        return cls.model_validate(raw)

    @property
    def mtus_per_hour(self) -> int:
        return 60 // self.mtu_minutes

    @property
    def mtus_per_day(self) -> int:
        return 24 * self.mtus_per_hour

    @property
    def dt_hours(self) -> float:
        return self.mtu_minutes / 60.0
