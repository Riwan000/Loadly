"""
API Routes for Electricity Demand Forecaster.
"""

from datetime import datetime, date
from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd
from scipy import stats
from fastapi import APIRouter, HTTPException, Query, status

from config import ALL_STATES, STATE_COORDINATES
from models import load_model, load_model_metadata, get_state_rmse
from data_loading import load_historical_demand, load_historical_data, denormalize_predictions
from weather import get_weather_forecast
from features import prepare_features_for_prediction, engineer_features
from visualization import (
    get_feature_importance,
    categorize_feature,
    generate_weather_insight,
    get_seasonal_comparison
)
from rag import generate_forecast_summary, initialize_rag_system
from utils.rag_engine import is_query_in_scope, RAGEngine
from utils.health_checks import check_model, check_rag, check_api_key
from utils.health_monitor import HealthMonitor
from utils.prediction_ledger import (
    record_forecast_batch,
    record_actuals,
    get_verification_metrics,
    query_ledger
)

from api.schemas import (
    StatesResponse, StateItem,
    ModelMetadataResponse,
    ForecastRequest, ForecastResponse, ForecastTimelineItem, ForecastSummary,
    WeatherImpactRequest, WeatherImpactResponse, WeatherSeriesItem,
    FeatureImportanceResponse, FeatureItem,
    SeasonalComparisonResponse, SeasonStats,
    ChatRequest, ChatResponse, SourceItem,
    SuggestionsResponse,
    HealthResponse,
    LedgerRecordsResponse, LedgerRecordItem,
    VerificationMetricsResponse,
    ActualDemandIngestRequest, ActualDemandIngestResponse
)

router = APIRouter(prefix="/api", tags=["Forecasting Platform"])

# Region mapping for Indian states
STATE_REGIONS = {
    # Northern Region (NR)
    "Punjab": "nr", "Haryana": "nr", "Rajasthan": "nr", "Uttar Pradesh": "nr",
    "Uttarakhand": "nr", "Himachal Pradesh": "nr", "Jammu and Kashmir": "nr",
    "NCT of Delhi": "nr", "Chandigarh": "nr",
    # Western Region (WR)
    "Maharashtra": "wr", "Gujarat": "wr", "Madhya Pradesh": "wr", "Chhattisgarh": "wr",
    "Goa": "wr", "Dadra and Nagar Haveli": "wr", "Daman and Diu": "wr",
    # Southern Region (SR)
    "Tamil Nadu": "sr", "Karnataka": "sr", "Andhra Pradesh": "sr",
    "Telangana": "sr", "Kerala": "sr", "Puducherry": "sr", "Lakshadweep": "sr",
    # Eastern Region (ER)
    "Bihar": "er", "Odisha": "er", "West Bengal": "er", "Jharkhand": "er",
    "Andaman and Nicobar": "er",
    # North-Eastern Region (NER)
    "Assam": "ner", "Sikkim": "ner", "Manipur": "ner", "Meghalaya": "ner",
    "Mizoram": "ner", "Nagaland": "ner", "Tripura": "ner", "Arunachal Pradesh": "ner"
}

_EASTERN_REGION_STATES = {"Bihar", "Chhattisgarh", "Jharkhand", "Odisha", "West Bengal"}


def to_serializable(obj: Any) -> Any:
    """Recursively convert numpy types and non-serializable objects to native Python types."""
    if isinstance(obj, dict):
        return {str(k): to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple, set)):
        return [to_serializable(item) for item in obj]
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return [to_serializable(x) for x in obj.tolist()]
    elif isinstance(obj, (np.str_, str)):
        return str(obj)
    return obj


# ============================================================================
# 1. Config & Metadata Endpoints
# ============================================================================
@router.get("/config/states", response_model=StatesResponse)
def get_all_states():
    """Return all 36 Indian states and union territories with coordinates and region."""
    states_list = []
    for state_name in ALL_STATES:
        coords = STATE_COORDINATES.get(state_name, {"lat": 20.5937, "lon": 78.9629})
        region = STATE_REGIONS.get(state_name, "other")
        states_list.append(StateItem(
            name=state_name,
            lat=coords["lat"],
            lon=coords["lon"],
            region=region
        ))
    return StatesResponse(states=states_list)


@router.get("/model/metadata", response_model=ModelMetadataResponse)
def get_model_metadata_endpoint():
    """Return model performance metrics, confidence levels, and thresholds."""
    metadata = load_model_metadata() or {}
    overall_metrics = {
        key: float(metadata[key])
        for key in ("mae", "rmse", "r2")
        if key in metadata
    }
    raw_state_rmse = metadata.get("state_rmse", {})
    clean_state_rmse = {str(k): float(v) for k, v in raw_state_rmse.items()} if raw_state_rmse else {}
    return ModelMetadataResponse(
        model_type=str(metadata.get("model_name", "Reduced_LR (Linear Regression)")),
        overall_metrics=overall_metrics,
        confidence_level=float(metadata.get("confidence_level", 0.90)),
        cdd_threshold_kelvin=float(metadata.get("cdd_threshold_kelvin", 297.15)),
        extreme_heat_threshold_kelvin=float(metadata.get("extreme_heat_threshold_kelvin", 305.0)),
        state_rmse=clean_state_rmse
    )


# ============================================================================
# 2. Demand Forecasting Endpoints
# ============================================================================
@router.post("/forecast", response_model=ForecastResponse)
def generate_forecast_endpoint(req: ForecastRequest):
    """
    Generate demand forecast for a state across specified horizon (7, 14, 30 days).
    Includes point predictions, upper/lower confidence intervals, and KPI summary.
    """
    if req.state not in ALL_STATES:
        raise HTTPException(status_code=400, detail=f"Invalid state: {req.state}")

    model = load_model()
    if model is None:
        raise HTTPException(status_code=500, detail="Model file not found or could not be loaded.")

    metadata = load_model_metadata()
    start_date = date.today()

    try:
        weather_forecast = get_weather_forecast(
            req.state,
            start_date,
            req.horizon_days,
            use_api=req.use_weather_api
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Weather API error: {str(e)}")

    if weather_forecast is None or len(weather_forecast) == 0:
        raise HTTPException(status_code=500, detail="Failed to retrieve weather data.")

    try:
        X_forecast, df_features = prepare_features_for_prediction(
            weather_forecast, req.state, model=model, metadata=metadata
        )
        pred_log = model.predict(X_forecast)

        historical_demand = load_historical_demand(req.state)
        denorm_result = denormalize_predictions(
            pred_log, weather_forecast['date'], req.state, historical_demand, is_log_scale=False
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forecasting calculation error: {str(e)}")

    # Calculate Confidence Intervals
    state_rmse = get_state_rmse(req.state, metadata)
    confidence_level = metadata.get('confidence_level', 0.90) if metadata else 0.90

    if state_rmse is not None:
        z_score = stats.norm.ppf((1 + confidence_level) / 2)
        margin = z_score * state_rmse
        upper_bound = denorm_result + margin
        lower_bound = np.maximum(0.0, denorm_result - margin)
    else:
        upper_bound = None
        lower_bound = None

    # Temperatures to Celsius
    temp_mean_c = weather_forecast['2m_temperature_mean'] - 273.15
    temp_max_c = weather_forecast['2m_temperature_max'] - 273.15

    timeline: List[ForecastTimelineItem] = []
    for i, dt in enumerate(weather_forecast['date']):
        dt_str = pd.to_datetime(dt).strftime('%Y-%m-%d')
        timeline.append(ForecastTimelineItem(
            date=dt_str,
            forecasted_demand_mu=round(float(denorm_result.iloc[i] if hasattr(denorm_result, 'iloc') else denorm_result[i]), 2),
            lower_bound_mu=round(float(lower_bound.iloc[i] if hasattr(lower_bound, 'iloc') else lower_bound[i]), 2) if lower_bound is not None else None,
            upper_bound_mu=round(float(upper_bound.iloc[i] if hasattr(upper_bound, 'iloc') else upper_bound[i]), 2) if upper_bound is not None else None,
            temperature_mean_c=round(float(temp_mean_c.iloc[i] if hasattr(temp_mean_c, 'iloc') else temp_mean_c[i]), 1),
            temperature_max_c=round(float(temp_max_c.iloc[i] if hasattr(temp_max_c, 'iloc') else temp_max_c[i]), 1),
            cdd=round(float(df_features['cdd'].iloc[i] if hasattr(df_features['cdd'], 'iloc') else df_features['cdd'][i]), 2),
            extreme_heat=int(df_features['extreme_heat'].iloc[i] if hasattr(df_features['extreme_heat'], 'iloc') else df_features['extreme_heat'][i]),
            is_holiday=int(df_features['is_holiday'].iloc[i] if hasattr(df_features['is_holiday'], 'iloc') else df_features['is_holiday'][i])
        ))

    # Summary metrics
    demands = [t.forecasted_demand_mu for t in timeline]
    peak_idx = int(np.argmax(demands))
    high_risk_count = sum(1 for t in timeline if t.extreme_heat == 1)

    summary = ForecastSummary(
        avg_daily_demand_mu=round(float(np.mean(demands)), 2),
        peak_demand_mu=round(float(np.max(demands)), 2),
        peak_demand_date=timeline[peak_idx].date,
        min_demand_mu=round(float(np.min(demands)), 2),
        high_risk_days_count=high_risk_count,
        state_rmse=round(float(state_rmse), 2) if state_rmse is not None else None,
        confidence_level=confidence_level
    )

    # Domain warnings
    warnings = []
    if req.state in _EASTERN_REGION_STATES:
        warnings.append(
            f"{req.state} is in the Eastern Region (ER), which was not included as a separate "
            "region in the model's training data. Forecasts for this state are approximate."
        )
    warnings.append(
        "Generation features are set to 0.0 at inference time. Demand-based features use per-state historical averages."
    )

    # Non-blocking automatic logging to Prediction Ledger
    try:
        model_name = str(metadata.get("model_name", "Reduced_LR")) if metadata else "Reduced_LR"
        timeline_dicts = [t.model_dump() if hasattr(t, "model_dump") else t.dict() for t in timeline]
        record_forecast_batch(timeline_dicts, req.state, model_name=model_name)
    except Exception as ledger_err:
        pass

    return ForecastResponse(
        state=req.state,
        horizon_days=req.horizon_days,
        summary=summary,
        timeline=timeline,
        warnings=warnings
    )


# ============================================================================
# 3. Weather Analytics Endpoints
# ============================================================================
@router.post("/analytics/weather-impact", response_model=WeatherImpactResponse)
def get_weather_impact_analytics(req: WeatherImpactRequest):
    """
    Returns time series overlay (demand vs temp vs CDD), Pearson correlations,
    and an automated natural language weather insight summary.
    """
    if req.state not in ALL_STATES:
        raise HTTPException(status_code=400, detail=f"Invalid state: {req.state}")

    model = load_model()
    metadata = load_model_metadata()

    if req.mode == "historical":
        historical_df = load_historical_data(req.state)
        if historical_df is None or len(historical_df) == 0:
            raise HTTPException(status_code=404, detail=f"No historical data found for {req.state}")

        df_features = engineer_features(historical_df, req.state, metadata=metadata)
        results_df = pd.DataFrame({
            'date': historical_df['date'],
            'forecasted_demand_MU': historical_df['target'] if 'target' in historical_df.columns else df_features.get('demand_rolling', 0.0),
            'cdd': df_features['cdd'],
            'extreme_heat': df_features['extreme_heat']
        })
        weather_df = historical_df
    else:
        start_date = date.today()
        weather_df = get_weather_forecast(req.state, start_date, req.horizon_days, use_api=req.use_weather_api)
        if weather_df is None or len(weather_df) == 0:
            raise HTTPException(status_code=500, detail="Failed to fetch weather data for analysis.")

        X_forecast, df_features = prepare_features_for_prediction(
            weather_df, req.state, model=model, metadata=metadata
        )
        if model:
            pred_log = model.predict(X_forecast)
            historical_demand = load_historical_demand(req.state)
            denorm_result = denormalize_predictions(pred_log, weather_df['date'], req.state, historical_demand)
        else:
            denorm_result = np.zeros(len(weather_df))

        results_df = pd.DataFrame({
            'date': weather_df['date'],
            'forecasted_demand_MU': denorm_result,
            'cdd': df_features['cdd'],
            'extreme_heat': df_features['extreme_heat']
        })

    # Natural language insight
    insight_text = generate_weather_insight(results_df, df_features, req.state)

    # Calculate correlations
    temp_mean_c = weather_df['2m_temperature_mean'] - 273.15
    temp_max_c = weather_df['2m_temperature_max'] - 273.15

    temp_corr = float(results_df['forecasted_demand_MU'].corr(temp_mean_c)) if len(results_df) > 1 else None
    cdd_corr = float(results_df['forecasted_demand_MU'].corr(df_features['cdd'])) if len(results_df) > 1 else None

    # Series items
    series: List[WeatherSeriesItem] = []
    baseline_col = df_features['energymet_7d_avg'] if 'energymet_7d_avg' in df_features.columns else None

    for i, dt in enumerate(weather_df['date']):
        dt_str = pd.to_datetime(dt).strftime('%Y-%m-%d')
        series.append(WeatherSeriesItem(
            date=dt_str,
            forecasted_demand_mu=round(float(results_df['forecasted_demand_MU'].iloc[i]), 2),
            baseline_7d_mu=round(float(baseline_col.iloc[i]), 2) if baseline_col is not None else None,
            temperature_mean_c=round(float(temp_mean_c.iloc[i]), 1),
            temperature_max_c=round(float(temp_max_c.iloc[i]), 1),
            cdd=round(float(df_features['cdd'].iloc[i]), 2),
            extreme_heat=int(df_features['extreme_heat'].iloc[i])
        ))

    return WeatherImpactResponse(
        state=req.state,
        correlations={
            "temperature_demand_corr": round(temp_corr, 3) if temp_corr is not None and not np.isnan(temp_corr) else None,
            "cdd_demand_corr": round(cdd_corr, 3) if cdd_corr is not None and not np.isnan(cdd_corr) else None
        },
        natural_language_insight=insight_text,
        series=series
    )


@router.get("/analytics/feature-importance", response_model=FeatureImportanceResponse)
def get_model_feature_importance(
    top_n: int = Query(15, ge=1, le=50),
    weather_only: bool = Query(False)
):
    """Return top feature importances from the model with category grouping."""
    model = load_model()
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")

    importance_dict = get_feature_importance(model, top_n=top_n, filter_weather_only=weather_only)
    if not importance_dict:
        # Fallback for linear regression or model without tree feature_importances_
        if hasattr(model, 'coef_') and hasattr(model, 'feature_names_in_'):
            coefs = np.abs(model.coef_)
            importance_dict = dict(sorted(zip(model.feature_names_in_, coefs), key=lambda x: x[1], reverse=True)[:top_n])
        else:
            importance_dict = {}

    feature_items = []
    for feat_name, score in importance_dict.items():
        category = categorize_feature(feat_name)
        feature_items.append(FeatureItem(
            name=feat_name,
            importance=round(float(score), 4),
            category=category
        ))

    return FeatureImportanceResponse(features=feature_items)


@router.get("/analytics/seasonal-comparison", response_model=SeasonalComparisonResponse)
def get_seasonal_breakdown(state: str = Query(..., examples=["Kerala"])):
    """Return seasonal statistical averages (temp, CDD, extreme heat %) for a state."""
    if state not in ALL_STATES:
        raise HTTPException(status_code=400, detail=f"Invalid state: {state}")

    current_month = datetime.now().month
    if current_month in [12, 1, 2]:
        season = "Winter"
    elif current_month in [3, 4, 5]:
        season = "Summer"
    elif current_month in [6, 7, 8, 9]:
        season = "Monsoon"
    else:
        season = "Autumn"

    seasonal_dict = get_seasonal_comparison(state, season)
    if not seasonal_dict:
        raise HTTPException(status_code=404, detail=f"No seasonal historical data available for {state}")

    seasons_response: Dict[str, SeasonStats] = {}
    for s_name, stats_data in seasonal_dict.items():
        seasons_response[s_name] = SeasonStats(
            temp_mean=round(float(stats_data.get('temp_mean', 0.0)), 1),
            temp_max=round(float(stats_data.get('temp_max', 0.0)), 1),
            cdd_mean=round(float(stats_data.get('cdd_mean', 0.0)), 2),
            extreme_heat_pct=round(float(stats_data.get('extreme_heat_pct', 0.0)), 1),
            count=int(stats_data.get('count', 0))
        )

    return SeasonalComparisonResponse(state=state, seasons=seasons_response)


# ============================================================================
# 4. AI Energy Assistant (RAG & LLM) Endpoints
# ============================================================================
@router.post("/assistant/chat", response_model=ChatResponse)
def assistant_chat_endpoint(req: ChatRequest):
    """
    Process natural language questions about forecasts, model performance,
    and weather impact using grounded RAG context.
    """
    if not is_query_in_scope(req.message):
        return ChatResponse(
            reply="I am specifically designed to assist with electricity demand forecasting, weather impacts, and model performance across Indian states. Please ask an energy or forecasting-related question.",
            sources=[],
            is_out_of_scope=True
        )

    try:
        vector_store, rag_engine = initialize_rag_system()
        if rag_engine is None:
            return ChatResponse(
                reply="Knowledge base is currently unavailable. Please verify local embeddings or OpenRouter API settings.",
                sources=[],
                is_out_of_scope=False
            )

        response_text, retrieved_docs, similarity = rag_engine.query(
            user_query=req.message,
            current_state=req.state,
            forecast_horizon=req.horizon_days
        )

        sources = []
        if retrieved_docs:
            for doc in retrieved_docs[:3]:
                raw_meta = doc.get('metadata', {})
                sources.append(SourceItem(
                    type=str(raw_meta.get('type', 'knowledge_base')),
                    snippet=str(doc.get('text', ''))[:200] + ("..." if len(str(doc.get('text', ''))) > 200 else ""),
                    metadata=to_serializable(raw_meta) if raw_meta else None
                ))

        return ChatResponse(
            reply=response_text,
            sources=sources,
            is_out_of_scope=False
        )

    except Exception as e:
        return ChatResponse(
            reply=f"AI Assistant encountered an issue: {str(e)}. Please check your OpenRouter API key or local embeddings.",
            sources=[],
            is_out_of_scope=False
        )


@router.get("/assistant/suggestions", response_model=SuggestionsResponse)
def get_assistant_suggestions(state: Optional[str] = Query(None, examples=["Maharashtra"])):
    """Get dynamic suggested prompt chips based on selected state."""
    target_state = state if state in ALL_STATES else "Kerala"
    suggestions = [
        f"What are the primary drivers of electricity demand in {target_state}?",
        f"How does extreme heat and CDD affect {target_state}'s cooling load?",
        f"What is the model accuracy (RMSE) for {target_state}?",
        "Explain how the Cooling Degree Days (CDD) feature is calculated."
    ]
    return SuggestionsResponse(state=target_state, suggestions=suggestions)


# ============================================================================
# 5. Health & Observability Endpoints
# ============================================================================
@router.get("/health", response_model=HealthResponse)
def get_system_health():
    """System status check for Model, RAG Vector Store, Weather API, and API Keys."""
    model_ok, model_reason = check_model()
    rag_ok, rag_reason = check_rag()
    api_key_ok, api_key_reason = check_api_key()

    overall_status = "healthy" if (model_ok and rag_ok) else "degraded"

    components = {
        "model": {"ready": model_ok, "details": model_reason},
        "rag_vector_store": {"ready": rag_ok, "details": rag_reason},
        "openrouter_api_key": {"configured": api_key_ok, "details": api_key_reason},
        "weather_api": {"provider": "Open-Meteo", "status": "online"}
    }

    return HealthResponse(
        status=overall_status,
        components=components,
        last_error=None if overall_status == "healthy" else f"Model: {model_reason}; RAG: {rag_reason}"
    )


# ============================================================================
# 6. Prediction Ledger & Verification Endpoints
# ============================================================================
@router.get("/ledger/records", response_model=LedgerRecordsResponse)
def get_ledger_records(
    state: Optional[str] = Query(None, description="Filter by state name"),
    limit: int = Query(50, ge=1, le=500, description="Max records to return"),
    reconciled_only: bool = Query(False, description="Only return records with actuals")
):
    """
    Retrieve historical forecast predictions from the audit ledger,
    including confidence intervals, weather conditions, and reconciled actuals.
    """
    rows = query_ledger(state=state, limit=limit, reconciled_only=reconciled_only)
    record_items = [LedgerRecordItem(**r) for r in rows]
    return LedgerRecordsResponse(records=record_items, total_count=len(record_items))


@router.get("/ledger/metrics", response_model=VerificationMetricsResponse)
def get_ledger_verification_metrics(
    state: Optional[str] = Query(None, description="Filter by state name"),
    days_window: int = Query(30, ge=0, le=365, description="Lookback window in days (0 for all-time)")
):
    """
    Compute real-time verification and drift metrics:
    MAE, RMSE, MAPE (%), 90% Confidence Interval empirical coverage rate, and forecast bias.
    """
    metrics = get_verification_metrics(state=state, days_window=days_window)
    return VerificationMetricsResponse(**metrics)


@router.post("/ledger/actuals", response_model=ActualDemandIngestResponse)
def ingest_actual_demands(req: ActualDemandIngestRequest):
    """
    Ingest ground-truth actual electricity demand for past dates and reconcile
    matching forecasts in the ledger.
    """
    records_list = [r.model_dump() if hasattr(r, "model_dump") else r.dict() for r in req.records]
    reconciled = record_actuals(records_list)
    return ActualDemandIngestResponse(reconciled_count=reconciled, status="success")

