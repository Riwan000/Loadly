"""
Pydantic schemas for Electricity Demand Forecaster REST API.
"""

from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Config & Metadata
# ---------------------------------------------------------------------------
class StateItem(BaseModel):
    name: str
    lat: float
    lon: float
    region: Optional[str] = None


class StatesResponse(BaseModel):
    states: List[StateItem]


class ModelMetadataResponse(BaseModel):
    model_type: str = "Reduced_LR (Linear Regression)"
    overall_metrics: Dict[str, float] = Field(default_factory=dict)
    confidence_level: float = 0.90
    cdd_threshold_kelvin: float = 297.15
    extreme_heat_threshold_kelvin: float = 305.0
    state_rmse: Optional[Dict[str, float]] = None


# ---------------------------------------------------------------------------
# Forecasting
# ---------------------------------------------------------------------------
class ForecastRequest(BaseModel):
    state: str = Field(..., json_schema_extra={"example": "Maharashtra"})
    horizon_days: int = Field(7, ge=1, le=30, json_schema_extra={"example": 7})
    use_weather_api: bool = Field(True, json_schema_extra={"example": True})


class ForecastTimelineItem(BaseModel):
    date: str
    forecasted_demand_mu: float
    lower_bound_mu: Optional[float] = None
    upper_bound_mu: Optional[float] = None
    temperature_mean_c: float
    temperature_max_c: float
    cdd: float
    extreme_heat: int
    is_holiday: int


class ForecastSummary(BaseModel):
    avg_daily_demand_mu: float
    peak_demand_mu: float
    peak_demand_date: str
    min_demand_mu: float
    high_risk_days_count: int
    state_rmse: Optional[float] = None
    confidence_level: float = 0.90


class ForecastResponse(BaseModel):
    state: str
    horizon_days: int
    summary: ForecastSummary
    timeline: List[ForecastTimelineItem]
    warnings: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Weather Impact & Analytics
# ---------------------------------------------------------------------------
class WeatherImpactRequest(BaseModel):
    state: str = Field(..., json_schema_extra={"example": "Kerala"})
    mode: str = Field("forecast", json_schema_extra={"example": "forecast"})  # "forecast" or "historical"
    horizon_days: int = Field(14, ge=1, le=30, json_schema_extra={"example": 14})
    use_weather_api: bool = Field(True, json_schema_extra={"example": True})


class WeatherSeriesItem(BaseModel):
    date: str
    forecasted_demand_mu: float
    baseline_7d_mu: Optional[float] = None
    temperature_mean_c: float
    temperature_max_c: float
    cdd: float
    extreme_heat: int


class WeatherImpactResponse(BaseModel):
    state: str
    correlations: Dict[str, Optional[float]]
    natural_language_insight: str
    series: List[WeatherSeriesItem]


class FeatureItem(BaseModel):
    name: str
    importance: float
    category: str


class FeatureImportanceResponse(BaseModel):
    features: List[FeatureItem]


class SeasonStats(BaseModel):
    temp_mean: float
    temp_max: float
    cdd_mean: float
    extreme_heat_pct: float
    count: int


class SeasonalComparisonResponse(BaseModel):
    state: str
    seasons: Dict[str, SeasonStats]


# ---------------------------------------------------------------------------
# AI Energy Assistant (RAG & LLM)
# ---------------------------------------------------------------------------
class ChatMessage(BaseModel):
    role: str  # "user" | "assistant" | "system"
    content: str


class ChatRequest(BaseModel):
    message: str = Field(..., json_schema_extra={"example": "Why is peak demand high for Maharashtra?"})
    state: Optional[str] = Field(None, json_schema_extra={"example": "Maharashtra"})
    horizon_days: Optional[int] = Field(7, json_schema_extra={"example": 7})
    chat_history: Optional[List[ChatMessage]] = Field(default_factory=list)


class SourceItem(BaseModel):
    type: str
    snippet: str
    metadata: Optional[Dict[str, Any]] = None


class ChatResponse(BaseModel):
    reply: str
    sources: List[SourceItem] = Field(default_factory=list)
    is_out_of_scope: bool = False


class SuggestionsResponse(BaseModel):
    state: str
    suggestions: List[str]


# ---------------------------------------------------------------------------
# Health & Status
# ---------------------------------------------------------------------------
class HealthResponse(BaseModel):
    status: str
    components: Dict[str, Any]
    last_error: Optional[str] = None
