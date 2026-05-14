"""Market clock: MTU grid, DA gate, intraday auction times, hour <-> MTU mapping.

Conventions:
- Internal timestamps are UTC, tz-aware.
- Display / market-time references use Europe/Tallinn (CET/CEST).
- The DA gate "12:00 CET on D-1" is interpreted in Europe/Tallinn local time;
  the function returns a UTC-aware timestamp.
- MVP assumes 96 MTUs per delivery day (no DST transition). A DST transition
  day raises a clear error; callers can later branch on this.
- DA market product is hourly; we submit q_DA at hourly resolution and broadcast
  the hourly DA price to its four MTUs at settlement. The bidirectional helpers
  `mtu_to_hour_index` and `hour_to_mtu_slice` keep this mapping in one place.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

TALLINN = ZoneInfo("Europe/Tallinn")
UTC = timezone.utc


class DSTTransitionError(ValueError):
    """Raised when a delivery day crosses a DST boundary (MVP unsupported)."""


@dataclass(frozen=True)
class Clock:
    """Resolves all market timing for a single delivery day."""

    delivery_date: date
    mtu_minutes: int = 15
    intraday_lead_times_h: tuple[int, ...] = (6, 3, 1)
    da_gate_local_time: time = time(12, 0)

    @property
    def mtus_per_hour(self) -> int:
        return 60 // self.mtu_minutes

    @property
    def mtus_per_day(self) -> int:
        return 24 * self.mtus_per_hour

    @property
    def dt_hours(self) -> float:
        return self.mtu_minutes / 60.0

    def delivery_start_utc(self) -> pd.Timestamp:
        local_midnight = datetime.combine(self.delivery_date, time(0, 0), tzinfo=TALLINN)
        return pd.Timestamp(local_midnight.astimezone(UTC))

    def delivery_end_utc(self) -> pd.Timestamp:
        local_next = datetime.combine(
            self.delivery_date + timedelta(days=1), time(0, 0), tzinfo=TALLINN
        )
        return pd.Timestamp(local_next.astimezone(UTC))

    def is_dst_transition_day(self) -> bool:
        start = self.delivery_start_utc()
        end = self.delivery_end_utc()
        hours = (end - start) / pd.Timedelta(hours=1)
        return not np.isclose(hours, 24.0)

    def assert_non_dst(self) -> None:
        if self.is_dst_transition_day():
            raise DSTTransitionError(
                f"Delivery date {self.delivery_date} crosses a DST boundary; "
                "MVP requires 24-hour days."
            )

    def mtu_index(self) -> pd.DatetimeIndex:
        self.assert_non_dst()
        start = self.delivery_start_utc()
        return pd.date_range(start, periods=self.mtus_per_day, freq=f"{self.mtu_minutes}min")

    def hour_index(self) -> pd.DatetimeIndex:
        self.assert_non_dst()
        start = self.delivery_start_utc()
        return pd.date_range(start, periods=24, freq="1h")

    def da_gate_utc(self) -> pd.Timestamp:
        """12:00 local time on D-1, returned as UTC."""
        prior_day = self.delivery_date - timedelta(days=1)
        gate_local = datetime.combine(prior_day, self.da_gate_local_time, tzinfo=TALLINN)
        return pd.Timestamp(gate_local.astimezone(UTC))

    def intraday_round_times_utc(self) -> list[pd.DatetimeIndex]:
        """For each auction round, return the per-period deadline timestamps.

        Round r covers delivery periods with `period_start - lead_h hours`. We
        return one timestamp per delivery period for clarity, though in practice
        the auction is gated once per round; the periodwise view simplifies the
        Oracle and the forecast generator.
        """
        self.assert_non_dst()
        out: list[pd.DatetimeIndex] = []
        mtus = self.mtu_index()
        for lead_h in self.intraday_lead_times_h:
            out.append(mtus - pd.Timedelta(hours=lead_h))
        return out

    def mtu_to_hour_index(self) -> np.ndarray:
        """Length-96 array giving the hour-of-day index (0..23) for each MTU."""
        return np.arange(self.mtus_per_day) // self.mtus_per_hour

    def hour_to_mtu_slice(self, hour: int) -> slice:
        """Slice over the four MTUs belonging to a given delivery hour."""
        n = self.mtus_per_hour
        return slice(hour * n, (hour + 1) * n)

    def broadcast_hourly_to_mtu(self, hourly_values: np.ndarray) -> np.ndarray:
        if hourly_values.shape[-1] != 24:
            raise ValueError(f"expected last axis of length 24, got {hourly_values.shape[-1]}")
        return np.repeat(hourly_values, self.mtus_per_hour, axis=-1)

    def aggregate_mtu_to_hourly_mean(self, mtu_values: np.ndarray) -> np.ndarray:
        if mtu_values.shape[-1] != self.mtus_per_day:
            raise ValueError(f"expected last axis {self.mtus_per_day}, got {mtu_values.shape[-1]}")
        new_shape = (*mtu_values.shape[:-1], 24, self.mtus_per_hour)
        return mtu_values.reshape(new_shape).mean(axis=-1)
