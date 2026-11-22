"""Demand-history data-quality signals: missing / stale history in forecasts, and chat bounds."""

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from config import STALE_DEMAND_DAYS
from features import (
    DEMAND_STATUS_MISSING, DEMAND_STATUS_OK, DEMAND_STATUS_STALE,
    assess_demand_history, demand_history_warning,
)
from server import app

client = TestClient(app)
START = pd.Timestamp("2026-11-21")


def _hist(last_date):
    dates = pd.date_range(end=last_date, periods=40)
    return pd.DataFrame({"date": dates, "actual_demand_MU": 10.0})


# ---------------------------------------------------------------------------
# assess_demand_history
# ---------------------------------------------------------------------------
def test_missing_history_detected_for_none_and_empty():
    assert assess_demand_history(None, START).status == DEMAND_STATUS_MISSING
    assert assess_demand_history(_hist("2026-11-01").iloc[0:0], START).status == DEMAND_STATUS_MISSING


def test_history_only_after_start_counts_as_missing():
    assert assess_demand_history(_hist("2026-12-31"), pd.Timestamp("2020-01-01")).status == DEMAND_STATUS_MISSING


def test_fresh_and_boundary_history_is_ok_and_older_is_stale():
    ok = assess_demand_history(_hist(START - pd.Timedelta(days=STALE_DEMAND_DAYS)), START)
    assert ok.status == DEMAND_STATUS_OK and ok.age_days == STALE_DEMAND_DAYS
    stale = assess_demand_history(_hist(START - pd.Timedelta(days=STALE_DEMAND_DAYS + 1)), START)
    assert stale.status == DEMAND_STATUS_STALE
    assert "demand observation" in demand_history_warning("X", stale)
    assert demand_history_warning("X", ok) is None


# ---------------------------------------------------------------------------
# /api/forecast
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("state", ["Lakshadweep", "Andaman and Nicobar"])
def test_single_state_without_history_returns_422(state):
    res = client.post("/api/forecast", json={"state": state, "horizon_days": 7, "use_weather_api": False})
    assert res.status_code == 422
    assert "no demand history" in res.json()["detail"]


def test_stale_state_forecast_carries_data_quality_warning():
    res = client.post("/api/forecast", json={"state": "Daman and Diu", "horizon_days": 7, "use_weather_api": False})
    assert res.status_code == 200
    warnings = res.json()["summary"]["data_quality_warnings"]
    assert len(warnings) == 1 and "2022-05-31" in warnings[0]


def test_healthy_state_has_no_data_quality_warnings():
    res = client.post("/api/forecast", json={"state": "Maharashtra", "horizon_days": 7, "use_weather_api": False})
    assert res.status_code == 200
    assert res.json()["summary"]["data_quality_warnings"] == []


def test_all_india_excludes_no_history_states_and_lists_them():
    res = client.post("/api/forecast", json={"state": "All-India", "horizon_days": 7, "use_weather_api": False})
    assert res.status_code == 200
    body = res.json()
    dq = body["summary"]["data_quality_warnings"]
    assert any("Excluded" in w and "Lakshadweep" in w and "Andaman and Nicobar" in w for w in dq)
    assert any("Daman and Diu" in w and "demand observation" in w for w in dq)
    names = {c["state"] for c in body["summary"]["top_contributors"]}
    assert not names & {"Lakshadweep", "Andaman and Nicobar"}


# ---------------------------------------------------------------------------
# ChatRequest bounds
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("payload", [
    {"message": "hi", "horizon_days": 10 ** 12},
    {"message": "hi", "horizon_days": 0},
    {"message": "hi", "horizon_days": 31},
    {"message": "hi", "state": "x" * 65},
])
def test_chat_request_rejects_unbounded_fields(payload):
    assert client.post("/api/assistant/chat", json=payload).status_code == 422


def test_chat_request_accepts_valid_bounds():
    from api.schemas import ChatRequest
    assert ChatRequest(message="hi", horizon_days=30, state="Maharashtra").horizon_days == 30
