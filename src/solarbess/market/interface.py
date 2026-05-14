"""Market interface: tracks the running position and clears trades.

MVP is price-taking: every quantity submitted at the gate is fully cleared at
the per-period DA / ID truth price. The hourly DA market is broadcast to MTUs
by sharing one DA price across the four MTUs of a delivery hour.

Sign convention: positive q = export (sell), negative q = import (buy).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Position:
    """Accumulating market position over the delivery day.

    `da_quantities_mw` is set once at the DA gate (length = mtus_per_day).
    `id_trades_mw` is built up across intraday auction rounds: shape (mtu, n_rounds).
    """

    n_mtu: int
    n_rounds: int
    da_quantities_mw: np.ndarray = field(default=None)  # type: ignore[assignment]
    id_trades_mw: np.ndarray = field(default=None)      # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.da_quantities_mw is None:
            self.da_quantities_mw = np.zeros(self.n_mtu, dtype=float)
        if self.id_trades_mw is None:
            self.id_trades_mw = np.zeros((self.n_mtu, self.n_rounds), dtype=float)

    def set_day_ahead(self, quantities_mw: np.ndarray) -> None:
        if quantities_mw.shape != (self.n_mtu,):
            raise ValueError(f"DA quantities shape {quantities_mw.shape} != ({self.n_mtu},)")
        self.da_quantities_mw = quantities_mw.astype(float).copy()

    def add_intraday(self, trades_mw: np.ndarray, auction_round: int) -> None:
        if trades_mw.shape != (self.n_mtu,):
            raise ValueError(f"ID trades shape {trades_mw.shape} != ({self.n_mtu,})")
        if not 0 <= auction_round < self.n_rounds:
            raise ValueError(f"auction_round {auction_round} out of range")
        self.id_trades_mw[:, auction_round] += trades_mw

    @property
    def committed_mw(self) -> np.ndarray:
        return self.da_quantities_mw + self.id_trades_mw.sum(axis=1)


@dataclass(frozen=True)
class MarketInterface:
    n_mtu: int
    n_rounds: int
    fees_eur_per_mwh: float = 0.0

    def clear_day_ahead(self, quantities_mw: np.ndarray, da_price_mtu: np.ndarray) -> np.ndarray:
        """Return per-MTU DA revenue (positive when selling at positive price)."""
        return quantities_mw * da_price_mtu

    def clear_intraday(
        self,
        trades_mw: np.ndarray,
        id_price_mtu: np.ndarray,
        auction_round: int,
    ) -> np.ndarray:
        """Return per-MTU ID revenue for this round's trades."""
        return trades_mw * id_price_mtu[:, auction_round]
