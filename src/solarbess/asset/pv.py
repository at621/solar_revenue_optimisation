"""Solar PV inverter clipping."""

from __future__ import annotations

import numpy as np


def inverter_clip(raw_solar_mw: np.ndarray, inverter_limit_mw: float) -> np.ndarray:
    """Clip solar at the inverter limit."""
    return np.minimum(raw_solar_mw, inverter_limit_mw)
