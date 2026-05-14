"""ScenarioForecast + ForecastGenerator.

Forecast = truth + lead-time-scaled, temporally-correlated error.
Solar-DA cross-variable correlation is imposed by rotating two independent
unit-variance noise streams via a 2x2 mixing matrix.

Equal probabilities for MVP (1/N each); the dataclass field is reusable for
non-equal probabilities once scenario reduction is added in a later phase.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from solarbess.config import ForecastConfig
from solarbess.forecast.error_model import (
    lead_time_sigma,
    sample_correlated_paths,
    temporal_correlation,
)
from solarbess.world.oracle import DecisionTime
from solarbess.world.truth import TruthPath


@dataclass(frozen=True)
class ScenarioForecast:
    scenario_ids: np.ndarray             # (N,)
    probabilities: np.ndarray            # (N,) sums to 1
    timestamps: pd.DatetimeIndex         # (T,)
    solar_mw: np.ndarray                 # (N, T)
    da_price: np.ndarray                 # (N, T)
    id_price: np.ndarray                 # (N, T) - uses last auction round as the relevant ID price
    imb_surplus: np.ndarray              # (N, T)
    imb_shortfall: np.ndarray            # (N, T)
    decision_time: DecisionTime

    def __post_init__(self) -> None:
        N = self.scenario_ids.shape[0]
        T = self.timestamps.shape[0]
        for name, arr in [
            ("solar_mw", self.solar_mw),
            ("da_price", self.da_price),
            ("id_price", self.id_price),
            ("imb_surplus", self.imb_surplus),
            ("imb_shortfall", self.imb_shortfall),
        ]:
            if arr.shape != (N, T):
                raise ValueError(f"{name} shape {arr.shape} != ({N}, {T})")
        if not math.isclose(float(self.probabilities.sum()), 1.0, abs_tol=1e-9):
            raise ValueError(f"probabilities must sum to 1 (got {self.probabilities.sum()})")


class ForecastGenerator:
    def __init__(self, cfg: ForecastConfig):
        self.cfg = cfg

    def generate(
        self,
        truth: TruthPath,
        now: pd.Timestamp,
        decision_time: DecisionTime,
        n_scenarios: int,
        seed: int,
    ) -> ScenarioForecast:
        rng = np.random.default_rng(seed)
        ts_all = truth.timestamps
        future_mask = ts_all >= now
        ts = ts_all[future_mask]
        if len(ts) == 0:
            # No tradable periods left - return a trivial single-period stub
            ts = ts_all[-1:]
            future_mask = ts_all == ts[0]
        T = len(ts)
        lead_hours = ((ts - now) / pd.Timedelta(hours=1)).to_numpy(dtype=float)
        # Time index in hours (relative) for temporal correlation
        t_hours = (ts - ts[0]) / pd.Timedelta(hours=1)
        t_hours = t_hours.to_numpy(dtype=float)

        cfg = self.cfg

        # Per-variable temporal correlation matrices and lead-time-scaled sigmas
        def _draws(sigma_floor: float, sigma_max: float, tau_lead: float, tau_corr: float) -> np.ndarray:
            sigma = lead_time_sigma(lead_hours, sigma_floor, sigma_max, tau_lead)
            corr = temporal_correlation(t_hours, tau_corr)
            return sample_correlated_paths(rng, sigma, corr, n_scenarios)

        eps_solar = _draws(cfg.solar_sigma_floor, cfg.solar_sigma_max, cfg.solar_tau_lead_h, cfg.solar_corr_tau_h)
        eps_da = _draws(cfg.da_sigma_floor, cfg.da_sigma_max, cfg.da_tau_lead_h, cfg.da_corr_tau_h)
        # Apply solar-DA cross correlation:
        # eps_da' = rho * eps_solar_normalised + sqrt(1-rho^2) * eps_da_independent (per scenario)
        rho = cfg.solar_da_corr
        if abs(rho) > 1e-6:
            # Rescale solar contribution to match DA sigma magnitude:
            solar_sigma = lead_time_sigma(lead_hours, cfg.solar_sigma_floor, cfg.solar_sigma_max, cfg.solar_tau_lead_h)
            da_sigma = lead_time_sigma(lead_hours, cfg.da_sigma_floor, cfg.da_sigma_max, cfg.da_tau_lead_h)
            scale = np.where(solar_sigma > 1e-6, da_sigma / solar_sigma, 0.0)
            eps_da = rho * eps_solar * scale[None, :] + math.sqrt(max(1 - rho * rho, 0.0)) * eps_da

        eps_id = _draws(cfg.id_sigma_floor, cfg.id_sigma_max, cfg.id_tau_lead_h, cfg.id_corr_tau_h)
        eps_imb_surplus = _draws(cfg.imb_sigma_floor, cfg.imb_sigma_max, cfg.imb_tau_lead_h, cfg.imb_corr_tau_h)
        eps_imb_shortfall = _draws(cfg.imb_sigma_floor, cfg.imb_sigma_max, cfg.imb_tau_lead_h, cfg.imb_corr_tau_h)

        truth_solar = truth.solar_mw[future_mask]
        truth_da = truth.da_price_mtu[future_mask]
        truth_id_last = truth.id_price_mtu[future_mask, -1]
        truth_imb_s = truth.imb_surplus_price[future_mask]
        truth_imb_f = truth.imb_shortfall_price[future_mask]

        solar = np.maximum(truth_solar[None, :] + eps_solar, 0.0)
        da = truth_da[None, :] + eps_da
        id_price = truth_id_last[None, :] + eps_id
        imb_s = np.minimum(truth_imb_s[None, :] + eps_imb_surplus, da - 0.1)
        imb_f = np.maximum(truth_imb_f[None, :] + eps_imb_shortfall, da + 0.1)

        probs = np.full(n_scenarios, 1.0 / n_scenarios)
        return ScenarioForecast(
            scenario_ids=np.arange(n_scenarios),
            probabilities=probs,
            timestamps=ts,
            solar_mw=solar,
            da_price=da,
            id_price=id_price,
            imb_surplus=imb_s,
            imb_shortfall=imb_f,
            decision_time=decision_time,
        )
