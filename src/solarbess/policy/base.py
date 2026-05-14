"""Policy contract: three callbacks, one class.

Per the Solar+Battery PDF (slide 9): the harness queries the policy at three
points only. The policy NEVER receives the TruthPath - only InfoSet and
AssetState. The OraclePolicy intentionally violates this by accepting a
TruthPath in __init__; the test suite enforces this is the only exception.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:
    from solarbess.asset.battery import AssetState
    from solarbess.market.interface import Position
    from solarbess.world.oracle import InfoSet


class Policy(Protocol):
    """The Policy Protocol. Three callbacks must be implemented."""

    name: str

    def decide_day_ahead(
        self, info: "InfoSet", asset_state: "AssetState"
    ) -> np.ndarray:
        """Return shape (n_mtu,) of DA quantities for the delivery day."""
        ...

    def decide_intraday(
        self,
        info: "InfoSet",
        asset_state: "AssetState",
        position: "Position",
        auction_round: int,
    ) -> np.ndarray:
        """Return shape (n_mtu,) of intraday trade deltas for this round.

        Periods already past `info.now` should have value 0.
        """
        ...

    def decide_dispatch(
        self,
        info: "InfoSet",
        asset_state: "AssetState",
        position: "Position",
    ) -> tuple[float, float]:
        """Return (charge_mw, discharge_mw) for the current delivery period.

        info.delivery_period_idx tells you which period this is.
        """
        ...
