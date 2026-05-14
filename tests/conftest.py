"""Shared pytest fixtures."""

from __future__ import annotations

from datetime import date

import pytest

from solarbess.config import AppConfig
from solarbess.world.truth import TruthGenerator


@pytest.fixture(scope="session")
def cfg() -> AppConfig:
    return AppConfig.from_yaml("configs/default.yaml")


@pytest.fixture(scope="session")
def truth(cfg: AppConfig):
    tg = TruthGenerator(cfg)
    return tg.sample(date(2026, 6, 1), seed=12345)
