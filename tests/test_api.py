"""
Unit & Integration tests for FastAPI REST API endpoints.
"""

import os
from urllib.parse import quote

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


# --- SPA catch-all: path traversal regression tests ---------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIST = os.path.join(PROJECT_ROOT, "frontend", "dist")
requires_dist = pytest.mark.skipif(
    not os.path.isfile(os.path.join(FRONTEND_DIST, "index.html")),
    reason="frontend/dist not built",
)


def _index_html() -> bytes:
    with open(os.path.join(FRONTEND_DIST, "index.html"), "rb") as f:
        return f.read()


def _requirements_marker() -> str:
    """First non-comment line of requirements.txt, used to detect leakage."""
    with open(os.path.join(PROJECT_ROOT, "requirements.txt"), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                return line
    pytest.skip("requirements.txt has no content to detect")


def _assert_not_leaked(response, marker: str):
    assert response.status_code in (200, 404)
    assert marker not in response.text
    if response.status_code == 200:
        assert response.content == _index_html()


@requires_dist
@pytest.mark.parametrize("url", [
    "/%2e%2e%2f%2e%2e%2f.env",
    "/%2e%2e/%2e%2e/.env",
    "/..%2f..%2f.env",
    "/..%5c..%5c.env",
    "/assets%2f..%2f..%2f..%2f.env",
])
def test_spa_encoded_traversal_does_not_leak_env(url):
    """Encoded dot segments / backslashes must not escape frontend/dist."""
    _assert_not_leaked(client.get(url), "OPENROUTER_API_KEY")


@requires_dist
@pytest.mark.parametrize("url", [
    "/%2e%2e/%2e%2e/requirements.txt",
    "/%2e%2e%2f%2e%2e%2frequirements.txt",
    "/..%5c..%5crequirements.txt",
])
def test_spa_encoded_traversal_does_not_leak_requirements(url):
    _assert_not_leaked(client.get(url), _requirements_marker())


@requires_dist
def test_spa_absolute_path_is_not_served():
    """Absolute paths (drive letters on Windows, leading slash on POSIX) are refused."""
    abs_path = os.path.join(PROJECT_ROOT, "requirements.txt").replace("\\", "/")
    marker = _requirements_marker()
    _assert_not_leaked(client.get("/" + quote(abs_path, safe="")), marker)
    _assert_not_leaked(client.get("/" + quote(abs_path, safe="/:")), marker)
    backslashed = os.path.join(PROJECT_ROOT, "requirements.txt").replace("/", "\\")
    _assert_not_leaked(client.get("/" + quote(backslashed, safe="")), marker)


@requires_dist
def test_spa_serves_legitimate_dist_file():
    candidates = [
        name for name in os.listdir(FRONTEND_DIST)
        if name != "index.html" and os.path.isfile(os.path.join(FRONTEND_DIST, name))
    ]
    if not candidates:
        pytest.skip("no static files besides index.html in frontend/dist")
    name = candidates[0]
    response = client.get(f"/{name}")
    assert response.status_code == 200
    with open(os.path.join(FRONTEND_DIST, name), "rb") as f:
        assert response.content == f.read()


@requires_dist
def test_spa_unknown_route_returns_index_html():
    response = client.get("/some/unknown/client-route", headers={"accept": "text/html"})
    assert response.status_code == 200
    assert response.content == _index_html()


@requires_dist
def test_spa_does_not_shadow_api_404():
    assert client.get("/api/does-not-exist").status_code == 404
