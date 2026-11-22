"""
Pydantic schemas for Electricity Demand Forecaster REST API.
"""

from typing import List, Dict, Optional, Any, Literal
from pydantic import BaseModel, Field

MAX_FORECAST_HORIZON_DAYS = 30
CHAT_STATE_MAX_CHARS = 64


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
    horizon_days: int = Field(7, ge=1, le=MAX_FORECAST_HORIZON_DAYS, json_schema_extra={"example": 7})
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
    regional_shares: Optional[Dict[str, float]] = None


class ForecastSummary(BaseModel):
    avg_daily_demand_mu: float
    peak_demand_mu: float
    peak_demand_date: str
    min_demand_mu: float
    high_risk_days_count: int
    state_rmse: Optional[float] = None
    horizon_rmse_mu: Optional[List[float]] = None  # RMSE used for each forecast day CI (day 1..N)
    confidence_level: float = 0.90
    is_national: bool = False
    fallback_states: Optional[List[str]] = None  # All-India: states whose model prediction failed
    regional_breakdown: Optional[Dict[str, float]] = None
    top_contributors: Optional[List[Dict[str, Any]]] = None
    extreme_heat_states: Optional[List[str]] = None
    # Stale / missing demand history notices (see config.STALE_DEMAND_DAYS); All-India
    # lists affected states and those excluded from the national sum.
    data_quality_warnings: List[str] = Field(default_factory=list)



class ForecastResponse(BaseModel):
    state: str
    horizon_days: int
    model_name: Optional[str] = "Reduced_LR"
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
CHAT_MESSAGE_MAX_CHARS = 2000
CHAT_HISTORY_CONTENT_MAX_CHARS = 4000
CHAT_HISTORY_MAX_ITEMS = 20


class ChatMessage(BaseModel):
    # Only conversational roles are accepted; a client-supplied "system" turn is rejected.
    role: Literal["user", "assistant"]
    content: str = Field(..., max_length=CHAT_HISTORY_CONTENT_MAX_CHARS)


class ChatRequest(BaseModel):
    message: str = Field(
        ...,
        max_length=CHAT_MESSAGE_MAX_CHARS,
        json_schema_extra={"example": "Why is peak demand high for Maharashtra?"},
    )
    state: Optional[str] = Field(None, max_length=CHAT_STATE_MAX_CHARS, json_schema_extra={"example": "Maharashtra"})
    # Same bounds as ForecastRequest.horizon_days (also feeds query_ledger(limit=...)).
    horizon_days: Optional[int] = Field(7, ge=1, le=MAX_FORECAST_HORIZON_DAYS, json_schema_extra={"example": 7})
    chat_history: Optional[List[ChatMessage]] = Field(
        default_factory=list, max_length=CHAT_HISTORY_MAX_ITEMS
    )


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


# ---------------------------------------------------------------------------
# Prediction Ledger & Verification
# ---------------------------------------------------------------------------
class LedgerRecordItem(BaseModel):
    id: int
    forecast_id: str
    created_at: str
    state: str
    target_date: str
    horizon_step: int
    model_name: str
    predicted_demand_mu: float
    lower_bound_mu: Optional[float] = None
    upper_bound_mu: Optional[float] = None
    temperature_mean_c: Optional[float] = None
    cdd: Optional[float] = None
    extreme_heat: int = 0
    actual_demand_mu: Optional[float] = None
    reconciled_at: Optional[str] = None
    error_mu: Optional[float] = None
    abs_error_mu: Optional[float] = None
    pct_error: Optional[float] = None
    within_ci: Optional[bool] = None


class LedgerRecordsResponse(BaseModel):
    records: List[LedgerRecordItem]
    total_count: int


class VerificationMetricsResponse(BaseModel):
    state: Optional[str] = None
    days_window: int = 30
    total_predictions_logged: int
    reconciled_count: int
    mae_mu: Optional[float] = None
    rmse_mu: Optional[float] = None
    mape_pct: Optional[float] = None
    mean_bias_mu: Optional[float] = None
    ci_coverage_pct: Optional[float] = None


class ActualDemandIngestItem(BaseModel):
    state: str
    target_date: str
    actual_demand_mu: float


class ActualDemandIngestRequest(BaseModel):
    records: List[ActualDemandIngestItem]


class ActualDemandIngestResponse(BaseModel):
    reconciled_count: int
    status: str = "success"


# ---------------------------------------------------------------------------
# Continuous Ingestion Pipeline
# ---------------------------------------------------------------------------
class IngestionRunRequest(BaseModel):
    target_date: Optional[str] = Field(None, json_schema_extra={"example": "2026-11-14"})
    catchup_days: int = Field(1, ge=1, le=30, json_schema_extra={"example": 1})


class IngestionRunResponse(BaseModel):
    status: str = "success"
    dates_processed: List[str]
    dates_missing: List[str] = Field(default_factory=list)
    total_actuals_ingested: int
    reconciled_forecasts: int
    verification_metrics: Optional[Dict[str, Any]] = None
    executed_at: str


# ---------------------------------------------------------------------------
# Automated data update (Grid-India demand + Open-Meteo weather)
# ---------------------------------------------------------------------------
class DataUpdateErrorItem(BaseModel):
    """Sanitised error entry: fixed vocabulary only (never exception text, paths or URLs)."""
    stage: str
    code: str
    message: str
    date: Optional[str] = None


class DataUpdateMissingItem(BaseModel):
    date: str
    reason: str


class DataUpdateWeatherAdded(BaseModel):
    archive: List[str] = Field(default_factory=list)
    provisional: List[str] = Field(default_factory=list)
    upgraded_to_archive: List[str] = Field(default_factory=list)
    finalized: List[str] = Field(default_factory=list)


class DataUpdateRunSummary(BaseModel):
    """Public subset of a run summary (see utils.daily_update.public_summary)."""
    dates_added: List[str] = Field(default_factory=list)
    dates_revised: List[str] = Field(default_factory=list)
    dates_still_missing: List[DataUpdateMissingItem] = Field(default_factory=list)
    weather_dates_added: DataUpdateWeatherAdded = Field(default_factory=DataUpdateWeatherAdded)
    ledger_reconciled: Optional[int] = None
    errors: List[DataUpdateErrorItem] = Field(default_factory=list)


class DataUpdateStatusResponse(BaseModel):
    running: bool = False
    last_run_at: Optional[str] = None
    last_run_status: Optional[str] = None
    last_run_summary: Optional[DataUpdateRunSummary] = None
    last_skipped_at: Optional[str] = None
    last_skip_reason: Optional[str] = None
    scheduler_enabled: bool = False
    next_scheduled_run: Optional[str] = None
    last_demand_date: Optional[str] = None
    last_weather_date: Optional[str] = None
    weather_provisional_dates: List[str] = Field(default_factory=list)


class DataUpdateRequest(BaseModel):
    catchup_days: int = Field(14, ge=1, le=60, json_schema_extra={"example": 14})


class DataUpdateStartResponse(BaseModel):
    status: str = Field(..., json_schema_extra={"example": "started"})


# ---------------------------------------------------------------------------
# MLOps Drift Detection & Adaptive Retraining
# ---------------------------------------------------------------------------
class DriftMetricsItem(BaseModel):
    mape_pct: Optional[float] = None
    mae_mu: Optional[float] = None
    rmse_mu: Optional[float] = None
    mean_bias_mu: Optional[float] = None
    bias_z_score: float = 0.0
    ci_coverage_pct: Optional[float] = None
    baseline_mape_pct: float = 1.68
    baseline_rmse_mu: float = 5.36


class FeatureDriftItem(BaseModel):
    temperature_psi: float = 0.0
    cdd_psi: float = 0.0


class DriftStatusResponse(BaseModel):
    scope: str
    status: str  # HEALTHY | WARNING | CRITICAL | INSUFFICIENT_DATA
    sample_count: int
    window_days: int
    metrics: DriftMetricsItem
    feature_drift: FeatureDriftItem
    diagnostics: List[str]
    recommended_action: str


class RetrainRequest(BaseModel):
    trigger_reason: str = Field("manual", json_schema_extra={"example": "manual_calibration"})
    force: bool = Field(False, json_schema_extra={"example": False})


class ModelEvalStats(BaseModel):
    model_name: str
    mae: Optional[float] = None
    rmse: Optional[float] = None
    r2: Optional[float] = None
    ci_coverage_pct: Optional[float] = None


class RetrainResponse(BaseModel):
    decision: str  # PROMOTED | REJECTED
    trigger_reason: str
    champion: ModelEvalStats
    challenger: ModelEvalStats
    gate_passed: bool
    notes: str
    rollback_available: bool
    warnings: List[str] = Field(default_factory=list)


class RollbackResponse(BaseModel):
    status: str
    message: str
    reloaded: bool
    warning: Optional[str] = None  # e.g. metadata backup missing, so CIs may describe the newer model


class RetrainingHistoryItem(BaseModel):
    id: int
    retrained_at: str
    trigger_reason: str
    champion_model: Optional[str] = None
    challenger_model: Optional[str] = None
    champion_mae: Optional[float] = None
    challenger_mae: Optional[float] = None
    champion_rmse: Optional[float] = None
    challenger_rmse: Optional[float] = None
    champion_r2: Optional[float] = None
    challenger_r2: Optional[float] = None
    champion_coverage: Optional[float] = None
    challenger_coverage: Optional[float] = None
    decision: str
    notes: Optional[str] = None


class RetrainingHistoryResponse(BaseModel):
    history: List[RetrainingHistoryItem]
    total_count: int



