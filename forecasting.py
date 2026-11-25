# ============================================================================
# LIVE FORECAST WITH GAP BRIDGING
# ============================================================================
"""
Live forecasts start at today's date, but the latest observed demand usually ends a few
days earlier (the source publishes late). Skipping that gap would make "day 1" a
(gap + 1)-day-ahead forecast that is presented, and given a confidence band, as 1-day-ahead.

forecast_bridged() starts the recursion the day after the last observation and runs it
through the gap days (no teacher forcing), then drops those days, so day k of the
response is a forecast of start_date + k - 1 produced by a gap + k step recursion.
"""
import logging
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from config import STALE_DEMAND_DAYS
from features import predict_recursive
from models import get_horizon_rmse_series
from weather import get_weather_forecast

logger = logging.getLogger(__name__)

DateLike = Union[date, pd.Timestamp]

# Largest observation gap that is bridged by recursion: the last gap at which
# assess_demand_history (age = gap + 1) still calls the history fresh. Beyond this the
# forecast is seeded from the stale history exactly as before, with the stale warning.
MAX_BRIDGE_DAYS = STALE_DEMAND_DAYS - 1


class WeatherFetchError(Exception):
    """The weather provider raised while fetching the forecast window."""


class WeatherEmptyError(Exception):
    """The weather provider returned no rows for the requested window."""


@dataclass(frozen=True)
class BridgedForecast:
    """Result for the requested days only. `gap_days` is 0 when nothing was bridged."""
    predictions: np.ndarray
    df_features: pd.DataFrame
    weather: pd.DataFrame
    gap_days: int
    last_observed: Optional[pd.Timestamp]


def _last_observed_before(historical_demand: Optional[pd.DataFrame],
                          start_date: DateLike) -> Optional[pd.Timestamp]:
    """Latest demand observation date strictly before start_date, or None."""
    if historical_demand is None or len(historical_demand) == 0:
        return None
    dates = pd.to_datetime(historical_demand['date'])
    prior = dates[dates < pd.Timestamp(start_date).normalize()]
    return None if prior.empty else prior.max()


def compute_bridge_gap(historical_demand: Optional[pd.DataFrame],
                       start_date: DateLike) -> Tuple[int, Optional[pd.Timestamp]]:
    """
    Returns (gap_days, last_observed). gap_days is the number of unobserved days between
    the last observation and start_date, or 0 when it is absent, zero, or beyond
    MAX_BRIDGE_DAYS (those cases are not bridged).
    """
    last_obs = _last_observed_before(historical_demand, start_date)
    if last_obs is None:
        return 0, None
    gap = (pd.Timestamp(start_date).normalize() - last_obs).days - 1
    if gap <= 0 or gap > MAX_BRIDGE_DAYS:
        return 0, last_obs
    return gap, last_obs


def _fetch_weather(state: str, run_start: DateLike, n_days: int, use_api: bool) -> pd.DataFrame:
    try:
        weather = get_weather_forecast(state, run_start, n_days, use_api=use_api)
    except Exception as exc:
        logger.exception("Weather forecast retrieval failed for %s", state)
        raise WeatherFetchError(str(exc)) from exc
    if weather is None or len(weather) == 0:
        raise WeatherEmptyError(f"No weather data for {state}")
    return weather


def forecast_bridged(model: Any, state: str, start_date: DateLike, horizon_days: int,
                     metadata: Optional[Dict[str, Any]],
                     historical_demand: Optional[pd.DataFrame],
                     use_api: bool = True) -> BridgedForecast:
    """
    Forecast horizon_days days from start_date, bridging any gap since the last observation.

    Raises WeatherFetchError / WeatherEmptyError for weather problems; any other
    exception comes from the model and is propagated unchanged.
    """
    gap, last_obs = compute_bridge_gap(historical_demand, start_date)
    run_start = start_date if gap == 0 else (last_obs + timedelta(days=1)).date()
    weather = _fetch_weather(state, run_start, gap + horizon_days, use_api)
    if len(weather) <= gap:
        raise WeatherEmptyError(f"Weather for {state} does not cover the forecast days")

    preds, df_features = predict_recursive(
        model, weather, state, metadata=metadata, historical_demand=historical_demand)

    preds = np.asarray(preds, dtype=float)[gap:]
    return BridgedForecast(
        predictions=preds,
        df_features=df_features.iloc[gap:].reset_index(drop=True),
        weather=weather.iloc[gap:].reset_index(drop=True),
        gap_days=gap,
        last_observed=last_obs,
    )


def bridged_horizon_rmse(state: str, gap_days: int, n_days: int,
                         metadata: Optional[Dict[str, Any]]) -> Optional[List[float]]:
    """RMSE for requested days 1..n_days, whose true horizons are gap_days + 1 .. gap_days + n_days."""
    return get_horizon_rmse_series(state, n_days, metadata, first_horizon=gap_days + 1)


def bridge_warning(state: str, last_observed: Optional[pd.Timestamp], gap_days: int) -> Optional[str]:
    """User-facing note for a bridged forecast; None when nothing was bridged."""
    if gap_days <= 0 or last_observed is None:
        return None
    return (f"Latest observed demand for {state} is {pd.Timestamp(last_observed).date()}; "
            f"the {gap_days} day(s) in between are model estimates, so day 1 is a "
            f"{gap_days + 1}-day-ahead forecast and its confidence band is widened accordingly.")


def national_bridge_warning(bridged_by_state: Dict[str, BridgedForecast]) -> Optional[str]:
    """One warning for the All-India view covering every state whose gap was bridged."""
    gapped = {s: b for s, b in bridged_by_state.items() if b.gap_days > 0}
    if not gapped:
        return None
    max_gap = max(b.gap_days for b in gapped.values())
    last_dates = sorted({pd.Timestamp(b.last_observed).date() for b in gapped.values()})
    span = str(last_dates[0]) if len(last_dates) == 1 else f"{last_dates[0]} to {last_dates[-1]}"
    return (f"Latest observed demand for {len(gapped)} state(s) ends {span}; the unobserved days "
            f"in between (up to {max_gap}) are model estimates, so day 1 is up to a "
            f"{max_gap + 1}-day-ahead forecast and the confidence band is widened accordingly.")
