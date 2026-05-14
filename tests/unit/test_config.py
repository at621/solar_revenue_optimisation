"""Config loads, validates, rejects bad YAML."""

from __future__ import annotations

import pytest
import yaml
from pydantic import ValidationError

from solarbess.config import AppConfig


def test_default_loads(cfg: AppConfig) -> None:
    assert cfg.battery.capacity_mwh > 0
    assert cfg.mtu_minutes == 15
    assert cfg.mtus_per_day == 96


def test_soc_bounds_invalid(tmp_path) -> None:
    bad = {
        "mtu_minutes": 15,
        "battery": {
            "capacity_mwh": 10,
            "p_charge_max_mw": 1,
            "p_discharge_max_mw": 1,
            "eta_charge": 0.95,
            "eta_discharge": 0.95,
            "soc_min_mwh": 5,
            "soc_max_mwh": 4,  # invalid: max < min
            "soc_initial_mwh": 5,
            "soc_terminal_min_mwh": 5,
            "degradation_cost_eur_per_mwh": 4,
        },
        "market": {"export_limit_mw": 10, "import_limit_mw": 0},
    }
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(bad))
    with pytest.raises(ValidationError):
        AppConfig.from_yaml(p)


def test_mtu_divides_60() -> None:
    with pytest.raises(ValidationError):
        AppConfig.model_validate(
            {
                "mtu_minutes": 7,
                "battery": {
                    "capacity_mwh": 10, "p_charge_max_mw": 1, "p_discharge_max_mw": 1,
                    "eta_charge": 0.95, "eta_discharge": 0.95,
                    "soc_min_mwh": 1, "soc_max_mwh": 9, "soc_initial_mwh": 5,
                    "soc_terminal_min_mwh": 1, "degradation_cost_eur_per_mwh": 4,
                },
                "market": {"export_limit_mw": 10, "import_limit_mw": 0},
            }
        )
