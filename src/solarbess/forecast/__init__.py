"""Forecast module: scenarios = truth + structured (lead-time-scaled, correlated) error."""

from solarbess.forecast.scenarios import ForecastGenerator, ScenarioForecast

__all__ = ["ForecastGenerator", "ScenarioForecast"]
