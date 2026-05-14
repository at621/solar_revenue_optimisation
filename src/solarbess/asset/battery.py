"""Battery state + step dynamics.

SoC evolves as:
    soc[t+1] = soc[t] + eta_charge * charge * dt - discharge * dt / eta_discharge

Solar-only charging enforced in `step` by raising if `charge > realized_solar`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from solarbess.config import BatteryParams


@dataclass(frozen=True)
class AssetState:
    soc_mwh: float
    cumulative_throughput_mwh: float = 0.0
    realized_solar_mw_last: float = 0.0


class AssetModel:
    def __init__(self, battery: BatteryParams, pv_peak_mw: float, inverter_limit_mw: float):
        self.battery = battery
        self.pv_peak_mw = pv_peak_mw
        self.inverter_limit_mw = inverter_limit_mw

    def step(
        self,
        state: AssetState,
        charge_mw: float,
        discharge_mw: float,
        realized_solar_mw: float,
        dt_hours: float,
        curtail_mw: float = 0.0,
        allow_grid_charging: bool = False,
    ) -> tuple[AssetState, float]:
        """Advance one period. Returns (new_state, physical_injection_mw).

        physical_injection = realized_solar - curtail + discharge - charge

        Enforces:
        - charge in [0, p_charge_max]
        - discharge in [0, p_discharge_max]
        - SoC remains in [soc_min, soc_max]
        - if not allow_grid_charging: charge <= realized_solar - curtail
        """
        bp = self.battery
        if charge_mw < -1e-9 or discharge_mw < -1e-9:
            raise ValueError(f"charge/discharge must be non-negative; got {charge_mw}, {discharge_mw}")
        if charge_mw > bp.p_charge_max_mw + 1e-9:
            raise ValueError(f"charge {charge_mw} exceeds p_charge_max {bp.p_charge_max_mw}")
        if discharge_mw > bp.p_discharge_max_mw + 1e-9:
            raise ValueError(f"discharge {discharge_mw} exceeds p_discharge_max {bp.p_discharge_max_mw}")

        net_solar = realized_solar_mw - curtail_mw
        if not allow_grid_charging and charge_mw > net_solar + 1e-9:
            raise ValueError(
                f"solar-only charging violated: charge {charge_mw} > available solar {net_solar}"
            )

        new_soc = (
            state.soc_mwh
            + bp.eta_charge * charge_mw * dt_hours
            - discharge_mw * dt_hours / bp.eta_discharge
        )
        # Clip to bounds with a small tolerance; raise if grossly out of range.
        if new_soc < bp.soc_min_mwh - 1e-6:
            raise ValueError(
                f"SoC underflow: would reach {new_soc:.6f} below min {bp.soc_min_mwh}"
            )
        if new_soc > bp.soc_max_mwh + 1e-6:
            raise ValueError(
                f"SoC overflow: would reach {new_soc:.6f} above max {bp.soc_max_mwh}"
            )
        new_soc = min(max(new_soc, bp.soc_min_mwh), bp.soc_max_mwh)

        injection = realized_solar_mw - curtail_mw + discharge_mw - charge_mw
        throughput = state.cumulative_throughput_mwh + (charge_mw + discharge_mw) * dt_hours
        return (
            replace(
                state,
                soc_mwh=new_soc,
                cumulative_throughput_mwh=throughput,
                realized_solar_mw_last=realized_solar_mw,
            ),
            injection,
        )
