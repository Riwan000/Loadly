# ============================================================================
# TESTS: gap bridging for live forecasts (forecasting.py), Open-Meteo date window,
# ledger horizon_step
# ============================================================================
from datetime import date, timedelta
from unittest import mock

import numpy as np
import pandas as pd
import pytest

import forecasting
import weather
from features import predict_recursive
from forecasting import MAX_BRIDGE_DAYS, bridge_warning, bridged_horizon_rmse, forecast_bridged
from models import get_horizon_rmse
from utils.prediction_ledger import query_ledger, record_forecast_batch

STATE = "Maharashtra"
HORIZON = 4
META = {
    "state_horizon_rmse": {STATE: {str(h): 10.0 * h for h in range(1, 31)}},
    "national_horizon_rmse": {"1": 90.0, "30": 300.0},
}


class StubLinearModel:
    """pred = roll_mean_7 + 0.5 * daily_change  (no intercept, no weather)."""
    feature_names_in_ = np.array(["demand_roll_mean_7", "daily_change"])

    def predict(self, X):
        return X["demand_roll_mean_7"].to_numpy() + 0.5 * X["daily_change"].to_numpy()


def _history(last_obs, n=20):
    dates = pd.date_range(end=last_obs, periods=n, freq="D")
    values = 100.0 + 3.0 * np.arange(n) + 5.0 * np.sin(np.arange(n))
    return pd.DataFrame({"date": dates, "actual_demand_MU": values})


@pytest.fixture
def fake_weather(monkeypatch, sample_weather_df):
    """Replace the weather provider; records every (start, n_days, use_api) request."""
    calls = []

    def _fake(state, start_date, num_days, use_api=True):
        calls.append((pd.Timestamp(start_date).date(), num_days, use_api))
        df = pd.concat([sample_weather_df] * 5, ignore_index=True).head(num_days).copy()
        df["date"] = pd.date_range(start=start_date, periods=num_days, freq="D")
        return df

    monkeypatch.setattr(forecasting, "get_weather_forecast", _fake)
    return calls


class TestForecastBridged:
    """PURPOSE: day 1 must be produced by a recursion that runs through the unobserved days.
    CATCHES: skipping the gap (day 1 silently being a gap+1 day-ahead forecast)."""

    def test_gap_runs_recursion_from_day_after_last_observation(self, fake_weather):
        start = date(2026, 11, 25)
        history = _history("2026-11-21")  # gap = 3 (22, 23, 24)

        result = forecast_bridged(StubLinearModel(), STATE, start, HORIZON, META, history, use_api=False)

        assert result.gap_days == 3
        assert fake_weather == [(date(2026, 11, 22), 3 + HORIZON, False)]
        full_weather = forecasting.get_weather_forecast(STATE, date(2026, 11, 22), 3 + HORIZON)
        expected, expected_feat = predict_recursive(
            StubLinearModel(), full_weather, STATE, historical_demand=history)
        assert result.predictions == pytest.approx(expected[3:])
        assert result.df_features["demand_roll_mean_7"].tolist() == pytest.approx(
            expected_feat["demand_roll_mean_7"].tolist()[3:])
        assert [d.date() for d in result.weather["date"]] == [start + timedelta(days=i) for i in range(HORIZON)]
        assert len(result.df_features) == HORIZON and len(result.predictions) == HORIZON
        assert result.df_features.index.tolist() == list(range(HORIZON))

    def test_bridging_differs_from_skipping_the_gap(self, fake_weather):
        start = date(2026, 11, 25)
        history = _history("2026-11-21")
        bridged = forecast_bridged(StubLinearModel(), STATE, start, HORIZON, META, history, use_api=False)
        direct, _ = predict_recursive(
            StubLinearModel(), forecasting.get_weather_forecast(STATE, start, HORIZON),
            STATE, historical_demand=history)
        assert not np.allclose(bridged.predictions, direct)

    def test_no_gap_is_identical_to_direct_call(self, fake_weather):
        start = date(2026, 11, 25)
        history = _history("2026-11-24")  # observed through yesterday

        result = forecast_bridged(StubLinearModel(), STATE, start, HORIZON, META, history, use_api=False)

        direct, direct_feat = predict_recursive(
            StubLinearModel(), forecasting.get_weather_forecast(STATE, start, HORIZON),
            STATE, historical_demand=history)
        assert result.gap_days == 0
        assert fake_weather[0] == (start, HORIZON, False)
        np.testing.assert_allclose(result.predictions, direct)
        pd.testing.assert_frame_equal(result.df_features, direct_feat)

    def test_gap_beyond_cap_falls_back_to_old_behaviour(self, fake_weather):
        start = date(2026, 11, 25)
        history = _history(start - timedelta(days=MAX_BRIDGE_DAYS + 2))  # gap = cap + 1

        result = forecast_bridged(StubLinearModel(), STATE, start, HORIZON, META, history, use_api=False)

        assert result.gap_days == 0
        assert fake_weather[0] == (start, HORIZON, False)
        assert result.last_observed is not None  # still reported for the caller

    def test_cap_matches_stale_history_threshold(self):
        from config import STALE_DEMAND_DAYS
        from features import assess_demand_history, DEMAND_STATUS_OK, DEMAND_STATUS_STALE
        start = pd.Timestamp("2026-11-25")
        at_cap = _history(start - pd.Timedelta(days=MAX_BRIDGE_DAYS + 1))
        past_cap = _history(start - pd.Timedelta(days=MAX_BRIDGE_DAYS + 2))
        assert MAX_BRIDGE_DAYS == STALE_DEMAND_DAYS - 1
        assert assess_demand_history(at_cap, start).status == DEMAND_STATUS_OK
        assert assess_demand_history(past_cap, start).status == DEMAND_STATUS_STALE

    def test_gap_at_cap_is_bridged(self, fake_weather):
        start = date(2026, 11, 25)
        history = _history(start - timedelta(days=MAX_BRIDGE_DAYS + 1))  # gap = cap
        result = forecast_bridged(StubLinearModel(), STATE, start, HORIZON, META, history, use_api=False)
        assert result.gap_days == MAX_BRIDGE_DAYS

    def test_history_dated_on_or_after_start_is_not_a_gap(self, fake_weather):
        start = date(2026, 11, 25)
        history = _history("2026-11-27")
        result = forecast_bridged(StubLinearModel(), STATE, start, HORIZON, META, history, use_api=False)
        assert result.gap_days == 0

    def test_no_history_behaves_as_before(self, fake_weather):
        result = forecast_bridged(StubLinearModel(), STATE, date(2026, 11, 25), HORIZON, META,
                                  pd.DataFrame({"date": [], "actual_demand_MU": []}), use_api=False)
        assert result.gap_days == 0 and result.last_observed is None

    def test_weather_errors_are_typed(self, monkeypatch):
        monkeypatch.setattr(forecasting, "get_weather_forecast",
                            mock.Mock(side_effect=RuntimeError("boom")))
        with pytest.raises(forecasting.WeatherFetchError):
            forecast_bridged(StubLinearModel(), STATE, date(2026, 11, 25), HORIZON, META,
                             _history("2026-11-24"))
        monkeypatch.setattr(forecasting, "get_weather_forecast", mock.Mock(return_value=pd.DataFrame()))
        with pytest.raises(forecasting.WeatherEmptyError):
            forecast_bridged(StubLinearModel(), STATE, date(2026, 11, 25), HORIZON, META,
                             _history("2026-11-24"))


class TestBridgedHorizonRmse:
    def test_uses_true_horizons(self):
        # table: rmse(h) = 10 * h
        assert bridged_horizon_rmse(STATE, 3, 4, META) == pytest.approx([40.0, 50.0, 60.0, 70.0])
        assert bridged_horizon_rmse(STATE, 0, 3, META) == pytest.approx([10.0, 20.0, 30.0])

    def test_horizon_beyond_metadata_max_is_clamped(self):
        series = bridged_horizon_rmse(STATE, 28, 5, META)  # horizons 29..33; table ends at 30
        assert series == pytest.approx([290.0, 300.0, 300.0, 300.0, 300.0])
        assert get_horizon_rmse(STATE, 100, META) == 300.0
        assert bridged_horizon_rmse("All-India", 40, 2, META) == pytest.approx([300.0, 300.0])

    def test_none_when_unavailable(self):
        assert bridged_horizon_rmse("Nowhere", 2, 3, META) is None


class TestBridgeWarning:
    def test_message_names_last_observation_and_horizon(self):
        msg = bridge_warning(STATE, pd.Timestamp("2026-11-20"), 4)
        assert "2026-11-20" in msg and "4 day(s)" in msg and "5-day-ahead" in msg

    def test_none_without_gap(self):
        assert bridge_warning(STATE, pd.Timestamp("2026-11-20"), 0) is None


class TestOpenMeteoWindow:
    """PURPOSE: a past start_date must produce a start_date/end_date request, not forecast_days.
    CATCHES: silently receiving data that starts today for a past start_date."""

    @staticmethod
    def _payload(start, n):
        days = [(start + timedelta(days=i)).isoformat() for i in range(n)]
        return {"daily": {
            "time": days,
            "temperature_2m_max": [35.0] * n, "temperature_2m_min": [25.0] * n,
            "dewpoint_2m_mean": [20.0] * n, "windspeed_10m_max": [10.0] * n,
            "winddirection_10m_dominant": [180.0] * n,
            "shortwave_radiation_sum": [20.0e6] * n, "cloudcover_mean": [40.0] * n,
        }}

    @staticmethod
    def _api_rows(past_days, anchor_offset=0):
        """
        What Open-Meteo returns for past_days + forecast_days=16: anchor-past .. anchor+15,
        where anchor is the API's own "today" (host today + anchor_offset).
        """
        anchor = date.today() + timedelta(days=anchor_offset)
        return anchor - timedelta(days=past_days), past_days + weather.OPENMETEO_FORECAST_DAYS

    def _call(self, start, n, payload_start=None, payload_n=None, anchor_offset=0):
        past = max(0, (date.today() - start).days) + weather.OPENMETEO_PAST_DAYS_SLACK
        first, count = self._api_rows(past, anchor_offset)
        payload = self._payload(payload_start or first, payload_n or count)
        response = mock.Mock(**{"json.return_value": payload, "raise_for_status.return_value": None})
        with mock.patch.object(weather.requests, "get", return_value=response) as get:
            df = weather.get_weather_forecast_openmeteo(STATE, start, n)
        return df, get.call_args[0][0]

    def test_past_start_sends_past_days_and_forecast_days(self):
        start = date.today() - timedelta(days=4)
        df, url = self._call(start, 9)
        assert "past_days=5" in url and "forecast_days=16" in url
        assert "start_date" not in url and "end_date" not in url
        assert df["date"].iloc[0].date() == start and len(df) == 9
        assert df["date"].diff().dropna().eq(pd.Timedelta(days=1)).all()

    def test_today_start_sends_only_slack_past_day(self):
        df, url = self._call(date.today(), 5)
        assert "past_days=1" in url and "forecast_days=16" in url
        assert df["date"].iloc[0].date() == date.today() and len(df) == 5

    def test_window_longer_than_api_range_returns_only_served_days(self):
        start = date.today() - timedelta(days=4)
        df, _ = self._call(start, 34)  # API serves today-4 .. today+15 = 20 days
        assert len(df) == 4 + weather.OPENMETEO_FORECAST_DAYS
        assert df["date"].iloc[-1].date() == date.today() + timedelta(days=15)

    @pytest.mark.parametrize("anchor_offset", [0, 1])
    def test_api_window_one_day_ahead_of_host_date_still_aligns(self, anchor_offset):
        """UTC host just after midnight UTC but IST already tomorrow: the API's 'today' is host today + 1."""
        start = date.today() - timedelta(days=4)
        df, _ = self._call(start, 9, anchor_offset=anchor_offset)
        assert df["date"].iloc[0].date() == start and len(df) == 9

    def test_null_padded_last_day_is_dropped(self):
        """The API returns an all-null row for the last forecast day; it must not crash the parse."""
        start = date.today() - timedelta(days=2)
        first, count = self._api_rows(3)
        payload = self._payload(first, count)
        for key in payload["daily"]:
            if key != "time":
                payload["daily"][key][-1] = None
        response = mock.Mock(**{"json.return_value": payload, "raise_for_status.return_value": None})
        with mock.patch.object(weather.requests, "get", return_value=response):
            df = weather.get_weather_forecast_openmeteo(STATE, start, 30)
        assert len(df) == 2 + 15 and df["date"].iloc[-1].date() == date.today() + timedelta(days=14)

    def test_start_older_than_past_range_raises(self):
        with pytest.raises(ValueError, match="past range"):
            weather.get_weather_forecast_openmeteo(
                STATE, date.today() - timedelta(days=weather.OPENMETEO_MAX_PAST_DAYS + 1), 5)

    def test_misaligned_response_raises(self):
        start = date.today() - timedelta(days=4)
        with pytest.raises(Exception, match="instead of"):
            self._call(start, 5, payload_start=date.today())  # API ignored past_days

    def test_misaligned_response_falls_back_to_climatology(self, sample_weather_df):
        start = date.today() - timedelta(days=4)
        with mock.patch.object(weather, "get_weather_forecast_openmeteo",
                               side_effect=ValueError("misaligned")), \
             mock.patch.object(weather, "get_weather_forecast_climatology",
                               return_value=sample_weather_df.head(5)) as clim:
            df = weather.get_weather_forecast(STATE, start, 5, use_api=True)
        clim.assert_called_once_with(STATE, start, 5)
        assert len(df) == 5

    def test_api_remainder_filled_with_climatology(self, sample_weather_df):
        start = date.today()
        api_df = pd.DataFrame({"date": pd.date_range(start, periods=16)})
        with mock.patch.object(weather, "get_weather_forecast_openmeteo", return_value=api_df), \
             mock.patch.object(weather, "get_weather_forecast_climatology",
                               return_value=pd.DataFrame({"date": pd.date_range(start + timedelta(days=16), periods=4)})) as clim:
            df = weather.get_weather_forecast(STATE, start, 20, use_api=True)
        clim.assert_called_once_with(STATE, start + timedelta(days=16), 4)
        assert len(df) == 20

    def test_climatology_accepts_past_start_date(self):
        start = date.today() - timedelta(days=5)
        df = weather.get_weather_forecast_climatology(STATE, start, 6)
        assert df["date"].iloc[0].date() == start and len(df) == 6


class TestLedgerHorizonStep:
    def test_horizon_step_is_true_lead_time(self, tmp_path):
        db = tmp_path / "ledger.db"
        items = [{"date": f"2026-11-{25 + i}", "forecasted_demand_mu": 100.0 + i} for i in range(3)]
        record_forecast_batch(items, STATE, db_path=db, auto_reconcile=False, horizon_offset=3)
        record_forecast_batch(items, "Goa", db_path=db, auto_reconcile=False)
        rows = query_ledger(state=STATE, db_path=db)
        steps = sorted(r["horizon_step"] for r in rows["items"]) if isinstance(rows, dict) else \
            sorted(r["horizon_step"] for r in rows)
        assert steps == [4, 5, 6]


class TestForecastEndpointBridging:
    """PURPOSE: the API response keeps its shape but carries the gap warning and wider day-1 band."""

    GAP = 3

    @pytest.fixture
    def client(self, monkeypatch):
        from fastapi.testclient import TestClient
        from api import routes
        from server import app
        last_obs = date.today() - timedelta(days=self.GAP + 1)
        monkeypatch.setattr(routes, "load_historical_demand", lambda state: _history(last_obs, n=60))
        return TestClient(app)

    def test_single_state_gap_warning_dates_and_band(self, client):
        data = client.post("/api/forecast", json={
            "state": "Kerala", "horizon_days": 5, "use_weather_api": False}).json()

        assert data["timeline"][0]["date"] == date.today().isoformat()
        assert len(data["timeline"]) == 5
        assert any(f"{self.GAP + 1}-day-ahead" in w for w in data["warnings"])
        rmse = data["summary"]["horizon_rmse_mu"]
        from models import get_horizon_rmse, load_model_metadata
        assert rmse[0] == pytest.approx(get_horizon_rmse("Kerala", self.GAP + 1, load_model_metadata()), abs=0.01)
