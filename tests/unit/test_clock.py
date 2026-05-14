"""Clock: DST guard, MTU index, hour <-> MTU mapping."""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from solarbess.clock import Clock, DSTTransitionError


def test_normal_day_has_96_mtus() -> None:
    c = Clock(delivery_date=date(2026, 6, 1))
    assert len(c.mtu_index()) == 96
    assert len(c.hour_index()) == 24


def test_dst_spring_raises() -> None:
    # Last Sunday of March 2026 = March 29 (DST start in Europe/Tallinn)
    c = Clock(delivery_date=date(2026, 3, 29))
    assert c.is_dst_transition_day()
    with pytest.raises(DSTTransitionError):
        c.mtu_index()


def test_dst_autumn_raises() -> None:
    # Last Sunday of October 2026 = October 25
    c = Clock(delivery_date=date(2026, 10, 25))
    assert c.is_dst_transition_day()
    with pytest.raises(DSTTransitionError):
        c.mtu_index()


def test_da_gate_12_local_in_summer() -> None:
    c = Clock(delivery_date=date(2026, 6, 1))
    gate = c.da_gate_utc()
    # 12:00 CEST -> 10:00 UTC; but May 31 (D-1) is summer, so 10:00 UTC.
    assert gate.tz_convert("Europe/Tallinn").hour == 12


def test_da_gate_12_local_in_winter() -> None:
    c = Clock(delivery_date=date(2026, 12, 15))
    gate = c.da_gate_utc()
    assert gate.tz_convert("Europe/Tallinn").hour == 12


def test_hour_mtu_broadcast() -> None:
    c = Clock(delivery_date=date(2026, 6, 1))
    hourly = np.arange(24, dtype=float)
    mtu = c.broadcast_hourly_to_mtu(hourly)
    assert mtu.shape == (96,)
    assert np.all(mtu[0:4] == 0.0)
    assert np.all(mtu[4:8] == 1.0)
    assert mtu[-1] == 23.0


def test_mtu_aggregate_inverse() -> None:
    c = Clock(delivery_date=date(2026, 6, 1))
    hourly = np.linspace(40.0, 100.0, 24)
    mtu = c.broadcast_hourly_to_mtu(hourly)
    np.testing.assert_allclose(c.aggregate_mtu_to_hourly_mean(mtu), hourly)
