"""
Unit & Integration tests for FastAPI REST API endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from server import app

client = TestClient(app)


def test_root_endpoint():
    """Verify landing endpoint returns 200 and docs links."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "version" in data
    assert data["documentation"] == "/docs"


def test_health_endpoint():
    """Verify system health check endpoint."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "components" in data
    assert "model" in data["components"]
    assert "rag_vector_store" in data["components"]


def test_get_all_states():
    """Verify state listing returns all 36 Indian states/UTs with coordinates."""
    response = client.get("/api/config/states")
    assert response.status_code == 200
    data = response.json()
    assert "states" in data
    assert len(data["states"]) == 36
    
    first_state = data["states"][0]
    assert "name" in first_state
    assert "lat" in first_state
    assert "lon" in first_state
    assert "region" in first_state


def test_get_model_metadata():
    """Verify model metadata endpoint returns performance stats and thresholds."""
    response = client.get("/api/model/metadata")
    assert response.status_code == 200
    data = response.json()
    assert "model_type" in data
    assert "overall_metrics" in data
    assert "cdd_threshold_kelvin" in data
    assert "extreme_heat_threshold_kelvin" in data


def test_post_forecast_valid():
    """Verify demand forecast generation."""
    payload = {
        "state": "Kerala",
        "horizon_days": 7,
        "use_weather_api": False  # Use climatology fallback for fast, hermetic test
    }
    response = client.post("/api/forecast", json=payload)
    assert response.status_code == 200
    data = response.json()
    
    assert data["state"] == "Kerala"
    assert data["horizon_days"] == 7
    assert "summary" in data
    assert data["summary"]["avg_daily_demand_mu"] > 0
    assert data["summary"]["peak_demand_mu"] >= data["summary"]["min_demand_mu"]
    assert len(data["timeline"]) == 7
    
    first_day = data["timeline"][0]
    assert "date" in first_day
    assert "forecasted_demand_mu" in first_day
    assert "temperature_mean_c" in first_day
    assert "cdd" in first_day


def test_post_forecast_invalid_state():
    """Verify 400 Bad Request on unknown state."""
    payload = {
        "state": "InvalidStateName",
        "horizon_days": 7,
        "use_weather_api": False
    }
    response = client.post("/api/forecast", json=payload)
    assert response.status_code == 400


def test_weather_impact_analytics():
    """Verify weather impact overlay and automated insights."""
    payload = {
        "state": "Kerala",
        "mode": "forecast",
        "horizon_days": 7,
        "use_weather_api": False
    }
    response = client.post("/api/analytics/weather-impact", json=payload)
    assert response.status_code == 200
    data = response.json()
    
    assert data["state"] == "Kerala"
    assert "correlations" in data
    assert "natural_language_insight" in data
    assert len(data["natural_language_insight"]) > 0
    assert len(data["series"]) == 7


def test_feature_importance():
    """Verify feature importance endpoint returns categorized drivers."""
    response = client.get("/api/analytics/feature-importance?top_n=10")
    assert response.status_code == 200
    data = response.json()
    assert "features" in data
    assert len(data["features"]) <= 10
    if len(data["features"]) > 0:
        feat = data["features"][0]
        assert "name" in feat
        assert "importance" in feat
        assert "category" in feat


def test_seasonal_comparison():
    """Verify seasonal historical comparison endpoint."""
    response = client.get("/api/analytics/seasonal-comparison?state=Kerala")
    assert response.status_code == 200
    data = response.json()
    assert data["state"] == "Kerala"
    assert "seasons" in data
    assert "Summer" in data["seasons"] or "Monsoon" in data["seasons"]


def test_assistant_suggestions():
    """Verify contextual prompt chips for a state."""
    response = client.get("/api/assistant/suggestions?state=Maharashtra")
    assert response.status_code == 200
    data = response.json()
    assert data["state"] == "Maharashtra"
    assert len(data["suggestions"]) > 0


def test_assistant_chat_out_of_scope():
    """Verify assistant correctly detects out-of-scope queries."""
    payload = {
        "message": "Give me a recipe for chocolate cake",
        "state": "Kerala"
    }
    response = client.post("/api/assistant/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["is_out_of_scope"] is True
    assert "specifically designed" in data["reply"]
