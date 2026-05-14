"""Five-layer synthetic world.

L1 latent state (seasonality + Markov regime)
L2 solar generation (clear-sky envelope x stochastic clearness)
L3 day-ahead price (base + residual-load term + heavy-tailed shock)
L4 intraday spread (per round, AR(1), correlated with DA shock)
L5 imbalance markup (sign-dependent)

Critical: L2 produces local solar; L3's price term consumes that same solar
through the residual-load proxy. This is the channel by which sunny days
depress midday prices (PDF slide 5 warning).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from solarbess.config import (
    L1Regime,
    L1Seasonality,
    L2Solar,
    L3DAPrice,
    L4IDPrice,
    L5Imbalance,
)

# ---------------------------------------------------------------------------
# L1 - latent state
# ---------------------------------------------------------------------------


def sample_regime(rng: np.random.Generator, n_steps: int, params: L1Regime) -> np.ndarray:
    """2-state Markov chain. Returns int array of {0=calm, 1=stressed}."""
    state = int(params.initial_state)
    out = np.empty(n_steps, dtype=np.int8)
    p_up = params.p_calm_to_stressed
    p_down = params.p_stressed_to_calm
    u = rng.random(n_steps)
    for i in range(n_steps):
        if state == 0 and u[i] < p_up:
            state = 1
        elif state == 1 and u[i] < p_down:
            state = 0
        out[i] = state
    return out


def seasonality_template(
    timestamps: pd.DatetimeIndex, seasonality: L1Seasonality
) -> np.ndarray:
    """Return a per-MTU multiplicative seasonality template (>0).

    Combines weekly shape (per day-of-week, indexed by MTU-of-day) and annual
    cosine envelope keyed on day-of-year.
    """
    local = timestamps.tz_convert("Europe/Tallinn")
    dow = local.dayofweek.to_numpy()
    doy = local.dayofyear.to_numpy()
    template = np.asarray(seasonality.weekly_load_template, dtype=float)
    mtus_per_day = template.shape[1]
    mtu_of_day = ((local.hour * 60 + local.minute) // (24 * 60 // mtus_per_day)).to_numpy()
    weekly = template[dow, mtu_of_day]
    annual = 1.0 + seasonality.annual_amplitude * np.cos(
        2.0 * np.pi * (doy - seasonality.annual_phase_doy) / 365.0
    )
    return weekly * annual


# ---------------------------------------------------------------------------
# L2 - solar
# ---------------------------------------------------------------------------


def _solar_declination_rad(doy: int) -> float:
    # Spencer 1971 simple formula
    g = 2.0 * math.pi * (doy - 1) / 365.0
    return (
        0.006918
        - 0.399912 * math.cos(g)
        + 0.070257 * math.sin(g)
        - 0.006758 * math.cos(2 * g)
        + 0.000907 * math.sin(2 * g)
        - 0.002697 * math.cos(3 * g)
        + 0.00148 * math.sin(3 * g)
    )


def clear_sky_envelope(
    timestamps: pd.DatetimeIndex, params: L2Solar
) -> np.ndarray:
    """Clear-sky cos(zenith), clipped to [0, 1]; multiply by peak_mw later."""
    local = timestamps.tz_convert("Europe/Tallinn")
    lat = math.radians(params.latitude_deg)
    out = np.zeros(len(timestamps), dtype=float)
    for i, ts in enumerate(local):
        doy = int(ts.dayofyear)
        # solar hour angle: 15 deg per hour from solar noon (approx local noon)
        hour_of_day = ts.hour + ts.minute / 60.0
        ha = math.radians(15.0 * (hour_of_day - 12.0))
        dec = _solar_declination_rad(doy)
        cos_zen = math.sin(lat) * math.sin(dec) + math.cos(lat) * math.cos(dec) * math.cos(ha)
        out[i] = max(0.0, cos_zen)
    return out


def _clearness_seasonal_mean(timestamps: pd.DatetimeIndex, params: L2Solar) -> np.ndarray:
    """Linear interpolation between winter and summer means by absolute DOY phase."""
    doy = timestamps.dayofyear.to_numpy()
    summer_weight = 0.5 * (1.0 - np.cos(2.0 * np.pi * (doy - 172) / 365.0))  # peak at DOY 172
    return params.clearness_mean_winter + summer_weight * (
        params.clearness_mean_summer - params.clearness_mean_winter
    )


def sample_clearness(
    rng: np.random.Generator,
    timestamps: pd.DatetimeIndex,
    params: L2Solar,
    regime: np.ndarray,
    regime_scale_stressed: float,
    dt_hours: float,
) -> tuple[np.ndarray, np.ndarray]:
    """OU process in logit space; returns (clearness in [0.05, 0.95], innovations).

    The innovations are returned because L3 uses them as the shared residual-load
    shock channel.
    """
    n = len(timestamps)
    mean_seasonal = _clearness_seasonal_mean(timestamps, params)
    # Cloudy-day flag per day; one Bernoulli per delivery day (here whole MTU range).
    cloudy_day = rng.random() < params.cloudy_day_prob
    cloudy_offset = (
        -params.cloudy_day_extra_sigma * abs(rng.standard_normal()) if cloudy_day else 0.0
    )

    # OU in logit space around seasonal mean
    def _logit(x: np.ndarray | float) -> np.ndarray:
        x = np.clip(np.asarray(x), 1e-4, 1 - 1e-4)
        return np.log(x / (1 - x))

    def _expit(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    mu_logit = _logit(mean_seasonal + cloudy_offset)
    theta = params.clearness_ou_theta
    sigma = params.clearness_ou_sigma
    out_logit = np.empty(n, dtype=float)
    out_logit[0] = mu_logit[0]
    innovations = np.zeros(n, dtype=float)
    # SDE: dX = theta * (mu - X) * dt + sigma * dW
    for i in range(1, n):
        scale = regime_scale_stressed if regime[i] == 1 else 1.0
        innov = rng.standard_normal() * sigma * scale * math.sqrt(dt_hours)
        innovations[i] = innov
        drift = theta * (mu_logit[i] - out_logit[i - 1]) * dt_hours
        out_logit[i] = out_logit[i - 1] + drift + innov
    clearness = np.clip(_expit(out_logit), 0.05, 0.95)
    return clearness, innovations


def project_solar_mw(
    clearness: np.ndarray, envelope: np.ndarray, peak_mw: float
) -> np.ndarray:
    """Realized solar at the panel: peak * envelope * clearness, in MW."""
    return peak_mw * envelope * clearness


# ---------------------------------------------------------------------------
# L3 - day-ahead price
# ---------------------------------------------------------------------------


def sample_da_price_hourly(
    rng: np.random.Generator,
    solar_mw_mtu: np.ndarray,
    seasonality_mtu: np.ndarray,
    regime_mtu: np.ndarray,
    params: L3DAPrice,
    peak_solar_mw: float,
    mtus_per_hour: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample hourly DA price (24,) and the per-hour DA shock used by L4.

    Price = base + beta_load * (load_shape - 1)  -  beta_solar * (solar/peak)
                   + beta_gas * gas_co2_state    + heavy-tailed shock.

    Solar enters with a NEGATIVE coefficient so high realised solar depresses
    midday prices - this is the L2/L3 coupling channel.
    """
    solar_h = solar_mw_mtu.reshape(24, mtus_per_hour).mean(axis=1)
    season_h = seasonality_mtu.reshape(24, mtus_per_hour).mean(axis=1)
    regime_h = regime_mtu.reshape(24, mtus_per_hour).max(axis=1)
    solar_norm = solar_h / max(peak_solar_mw, 1e-6)

    eps = rng.standard_t(df=params.shock_df, size=24)
    sigma = np.where(regime_h == 1, params.shock_sigma_stressed, params.shock_sigma_calm)
    shock = eps * sigma

    # Solar coefficient scaled large enough that a fully sunny day moves price
    # by roughly 2x the calm shock stdev - so the channel survives the noise.
    price = (
        params.base_eur_mwh
        + params.beta_load * (season_h - 1.0)
        - 3.0 * params.beta_load * solar_norm
        + params.beta_gas * params.gas_co2_state
        + shock
    )
    price = np.maximum(price, params.negative_price_floor)
    return price, shock


# ---------------------------------------------------------------------------
# L4 - intraday spread
# ---------------------------------------------------------------------------


def sample_intraday_prices(
    rng: np.random.Generator,
    da_price_mtu: np.ndarray,
    da_shock_hourly: np.ndarray,
    params: L4IDPrice,
    mtus_per_hour: int,
    n_rounds: int = 3,
) -> np.ndarray:
    """Per-MTU intraday price for each round. Shape (mtus, n_rounds)."""
    n_mtu = len(da_price_mtu)
    n_hour = n_mtu // mtus_per_hour
    sigma_by_round = list(params.spread_sigma_by_round)
    if len(sigma_by_round) < n_rounds:
        sigma_by_round = sigma_by_round + [sigma_by_round[-1]] * (n_rounds - len(sigma_by_round))

    # Correlated component with DA shock direction
    da_dir = np.sign(da_shock_hourly)
    out = np.zeros((n_mtu, n_rounds))
    phi = params.spread_ar1_phi
    rho = params.spread_da_shock_corr
    for r in range(n_rounds):
        sigma = sigma_by_round[r]
        spread = np.empty(n_hour)
        prev = 0.0
        for h in range(n_hour):
            corr_part = rho * sigma * da_dir[h]
            innov = math.sqrt(max(1.0 - rho * rho, 1e-6)) * sigma * rng.standard_normal()
            spread[h] = phi * prev + corr_part + innov
            prev = spread[h]
        spread_mtu = np.repeat(spread, mtus_per_hour)
        out[:, r] = da_price_mtu + spread_mtu
    return out


# ---------------------------------------------------------------------------
# L5 - imbalance
# ---------------------------------------------------------------------------


def sample_imbalance_prices(
    rng: np.random.Generator,
    da_price_mtu: np.ndarray,
    solar_anomaly_mtu: np.ndarray,
    params: L5Imbalance,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (imb_surplus_price, imb_shortfall_price) per MTU.

    system_sign +1 = system long (extra generation, e.g. high solar) -> surplus is cheap,
                                                                       shortfall not too punitive.
    system_sign -1 = system short (deficit) -> surplus very cheap (you give it away),
                                              shortfall expensive.
    """
    n = len(da_price_mtu)
    # System long when solar anomaly positive; with extra noise for wind/import proxy
    proxy_noise = rng.standard_normal(n)
    composite = (
        params.aggregate_solar_correlation * solar_anomaly_mtu
        + math.sqrt(max(1.0 - params.aggregate_solar_correlation ** 2, 1e-6)) * proxy_noise
    )
    system_long = composite > 0

    surplus = np.where(
        system_long,
        da_price_mtu - params.markup_long_surplus,
        da_price_mtu - params.markup_short_surplus,
    )
    shortfall = np.where(
        system_long,
        da_price_mtu + params.markup_long_shortfall,
        da_price_mtu + params.markup_short_shortfall,
    )
    # Sanity: shortfall >= surplus always
    shortfall = np.maximum(shortfall, surplus + 1e-3)
    return surplus, shortfall
