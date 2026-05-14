"""Battery SoC arithmetic + solar-only charging."""

from __future__ import annotations

import pytest

from solarbess.asset.battery import AssetModel, AssetState
from solarbess.config import BatteryParams


@pytest.fixture
def battery() -> BatteryParams:
    return BatteryParams(
        capacity_mwh=10.0,
        p_charge_max_mw=2.0,
        p_discharge_max_mw=2.0,
        eta_charge=0.9,
        eta_discharge=0.9,
        soc_min_mwh=0.5,
        soc_max_mwh=9.5,
        soc_initial_mwh=5.0,
        soc_terminal_min_mwh=1.0,
        degradation_cost_eur_per_mwh=2.0,
    )


def test_charge_increases_soc(battery: BatteryParams) -> None:
    asset = AssetModel(battery, pv_peak_mw=10, inverter_limit_mw=10)
    s = AssetState(soc_mwh=5.0)
    new_s, inj = asset.step(s, charge_mw=2.0, discharge_mw=0.0, realized_solar_mw=3.0, dt_hours=0.25)
    # Δsoc = +0.9 * 2 * 0.25 = +0.45
    assert new_s.soc_mwh == pytest.approx(5.0 + 0.45)
    # injection = 3 + 0 - 2 = 1
    assert inj == pytest.approx(1.0)


def test_discharge_decreases_soc(battery: BatteryParams) -> None:
    asset = AssetModel(battery, pv_peak_mw=10, inverter_limit_mw=10)
    s = AssetState(soc_mwh=5.0)
    new_s, inj = asset.step(s, charge_mw=0.0, discharge_mw=1.0, realized_solar_mw=3.0, dt_hours=0.25)
    # Δsoc = -1 * 0.25 / 0.9 = -0.2778
    assert new_s.soc_mwh == pytest.approx(5.0 - 0.25 / 0.9)
    assert inj == pytest.approx(3.0 + 1.0)


def test_charge_above_solar_raises_when_solar_only(battery: BatteryParams) -> None:
    asset = AssetModel(battery, pv_peak_mw=10, inverter_limit_mw=10)
    s = AssetState(soc_mwh=5.0)
    with pytest.raises(ValueError, match="solar-only"):
        asset.step(s, charge_mw=2.0, discharge_mw=0.0, realized_solar_mw=1.5, dt_hours=0.25)


def test_charge_above_solar_ok_when_grid_charging(battery: BatteryParams) -> None:
    asset = AssetModel(battery, pv_peak_mw=10, inverter_limit_mw=10)
    s = AssetState(soc_mwh=5.0)
    new_s, _ = asset.step(
        s, charge_mw=2.0, discharge_mw=0.0, realized_solar_mw=0.0, dt_hours=0.25, allow_grid_charging=True
    )
    assert new_s.soc_mwh > s.soc_mwh


def test_power_bound_raises(battery: BatteryParams) -> None:
    asset = AssetModel(battery, pv_peak_mw=10, inverter_limit_mw=10)
    s = AssetState(soc_mwh=5.0)
    with pytest.raises(ValueError, match="exceeds p_charge_max"):
        asset.step(s, charge_mw=99.0, discharge_mw=0.0, realized_solar_mw=99.0, dt_hours=0.25)


def test_soc_overflow_raises(battery: BatteryParams) -> None:
    asset = AssetModel(battery, pv_peak_mw=10, inverter_limit_mw=10)
    s = AssetState(soc_mwh=9.0)
    # delta soc = +0.9 * 2 * 1.0 = 1.8 -> 10.8 > soc_max 9.5
    with pytest.raises(ValueError, match="overflow"):
        asset.step(s, charge_mw=2.0, discharge_mw=0.0, realized_solar_mw=2.0, dt_hours=1.0)
