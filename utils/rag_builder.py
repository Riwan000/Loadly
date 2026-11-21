"""
Knowledge base builder for RAG system.
Creates structured knowledge documents from model metadata, feature importance, and diagnostics.
"""

import json
import os
from typing import List, Dict, Optional
from pathlib import Path


def load_model_metadata(metadata_path: str = "artifacts/model_metadata.json") -> Dict:
    """
    Load model metadata from JSON file.
    
    Parameters:
        metadata_path: Path to model_metadata.json
        
    Returns:
        Dictionary containing model metadata
    """
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Model metadata file not found: {metadata_path}")
    
    with open(metadata_path, 'r') as f:
        return json.load(f)


def build_model_metrics_documents(metadata: Dict) -> List[Dict]:
    """
    Build knowledge documents from model performance metrics.
    
    Parameters:
        metadata: Model metadata dictionary
        
    Returns:
        List of document dictionaries with 'text' and 'metadata' keys
    """
    documents = []
    
    # Overall model metrics
    model_name = metadata.get('model_name', 'Reduced_LR')
    rmse_val = metadata.get('rmse')
    mae_val = metadata.get('mae')
    r2_val = metadata.get('r2')
    rmse_str = f"{float(rmse_val):.2f}" if isinstance(rmse_val, (int, float)) else "N/A"
    mae_str = f"{float(mae_val):.2f}" if isinstance(mae_val, (int, float)) else "N/A"
    r2_str = f"{float(r2_val):.4f}" if isinstance(r2_val, (int, float)) else "N/A"
    n_features = metadata.get('n_features', 15)

    overall_text = (
        f"Model Performance: The active production electricity demand forecasting model ({model_name}, "
        f"{n_features} engineered features) has overall test RMSE of {rmse_str} MU, "
        f"MAE of {mae_str} MU, and R² of {r2_str}. "
        f"The model was trained on historical Indian grid data ({metadata.get('trained_on', '2013-2023')}). "
        f"Confidence interval level is set to {metadata.get('confidence_level', 0.9) * 100:.0f}%."
    )
    documents.append({
        'text': overall_text,
        'metadata': {
            'type': 'model_metrics',
            'scope': 'overall',
            'model_name': model_name,
            'rmse': rmse_val,
            'mae': mae_val,
            'r2': r2_val
        }
    })
    
    # State-specific metrics
    state_rmse = metadata.get('state_rmse', {})
    for state, rmse in state_rmse.items():
        state_text = (
            f"State Performance - {state}: The model has RMSE of {rmse:.2f} MU for {state}. "
            f"This indicates the typical prediction error for electricity demand forecasts in {state}. "
            f"Lower RMSE values indicate better model performance for this state."
        )
        documents.append({
            'text': state_text,
            'metadata': {
                'type': 'model_metrics',
                'scope': 'state',
                'state': state,
                'rmse': rmse
            }
        })
    
    # Summary statistics
    summary = metadata.get('state_rmse_summary', {})
    if summary:
        summary_text = (
            f"Model Performance Summary: The model was successfully trained for {summary.get('total_states', 0)} states. "
            f"Average RMSE across all states is {summary.get('mean_rmse', 0):.2f} MU. "
            f"Best performing state has RMSE of {summary.get('min_rmse', 0):.2f} MU, "
            f"while the most challenging state has RMSE of {summary.get('max_rmse', 0):.2f} MU."
        )
        documents.append({
            'text': summary_text,
            'metadata': {
                'type': 'model_metrics',
                'scope': 'summary',
                'mean_rmse': summary.get('mean_rmse'),
                'min_rmse': summary.get('min_rmse'),
                'max_rmse': summary.get('max_rmse')
            }
        })
    
    return documents


def build_feature_importance_documents(feature_importance: Dict, top_n: int = 10) -> List[Dict]:
    """
    Build knowledge documents from feature importance data.
    
    Parameters:
        feature_importance: Dictionary mapping feature names to importance scores
        top_n: Number of top features to include in documents
        
    Returns:
        List of document dictionaries
    """
    documents = []
    
    if not feature_importance:
        return documents
    
    # Sort features by importance
    sorted_features = sorted(
        feature_importance.items(),
        key=lambda x: x[1],
        reverse=True
    )[:top_n]
    
    # Create document for top features
    top_features_text = "Top Features for Electricity Demand Forecasting: "
    feature_descriptions = []
    
    for i, (feature, importance) in enumerate(sorted_features, 1):
        # Categorize feature
        category = categorize_feature_simple(feature)
        feature_descriptions.append(
            f"{i}. {feature} ({category}) with relative importance of {importance:.4f}"
        )
    
    top_features_text += "; ".join(feature_descriptions) + ". "
    top_features_text += (
        "These features are the most influential in predicting electricity demand. "
        "Higher importance values indicate stronger influence on forecast predictions."
    )
    
    documents.append({
        'text': top_features_text,
        'metadata': {
            'type': 'feature_importance',
            'top_n': int(top_n),
            'features': [str(f[0]) for f in sorted_features]
        }
    })
    
    # Create individual documents for top 5 features
    for i, (feature, importance) in enumerate(sorted_features[:5], 1):
        category = categorize_feature_simple(str(feature))
        feature_text = (
            f"Feature: {feature} is a {category} feature with importance score of {float(importance):.4f}, "
            f"making it one of the top {i} most important features for electricity demand forecasting. "
            f"This feature significantly influences the model's predictions."
        )
        documents.append({
            'text': feature_text,
            'metadata': {
                'type': 'feature_importance',
                'feature': str(feature),
                'importance': float(importance),
                'category': category,
                'rank': int(i)
            }
        })
    
    return documents


def categorize_feature_simple(feature_name: str) -> str:
    """
    Categorize a feature name into a simple category.
    
    Parameters:
        feature_name: Name of the feature
        
    Returns:
        Category string
    """
    feature_lower = feature_name.lower()
    
    if 'temperature' in feature_lower or 'temp' in feature_lower:
        return 'Temperature'
    elif 'humidity' in feature_lower or 'dewpoint' in feature_lower or 'utci' in feature_lower:
        return 'Humidity'
    elif 'wind' in feature_lower:
        return 'Wind'
    elif 'solar' in feature_lower or 'radiation' in feature_lower:
        return 'Solar'
    elif 'cloud' in feature_lower:
        return 'Cloud Cover'
    elif 'holiday' in feature_lower or 'weekend' in feature_lower or 'month' in feature_lower or 'day' in feature_lower:
        return 'Calendar'
    elif 'demand' in feature_lower or 'energy' in feature_lower or 'generation' in feature_lower:
        return 'Historical Demand'
    elif 'heat' in feature_lower or 'cdd' in feature_lower:
        return 'Heat/Cooling'
    else:
        return 'Other'


def build_diagnostic_documents(diagnostic_summary: str, state: Optional[str] = None) -> List[Dict]:
    """
    Build knowledge documents from diagnostic summaries.
    
    Parameters:
        diagnostic_summary: Text summary of model diagnostics
        state: Optional state name for context
        
    Returns:
        List of document dictionaries
    """
    documents = []
    
    if not diagnostic_summary:
        return documents
    
    # Split summary into sentences and create structured documents
    # This is a simple approach - could be enhanced with NLP parsing
    diagnostic_text = (
        f"Model Diagnostics{f' for {state}' if state else ''}: {diagnostic_summary} "
        f"These diagnostics help understand model performance, identify risky periods, "
        f"and explain forecast confidence levels."
    )
    
    documents.append({
        'text': diagnostic_text,
        'metadata': {
            'type': 'diagnostics',
            'state': state,
            'summary': diagnostic_summary
        }
    })
    
    return documents


def build_forecast_summary_documents(forecast_summary: Dict) -> List[Dict]:
    """
    Build knowledge documents from forecast summary.
    Creates a single locked document per forecast with exact schema - no recomputation allowed.
    
    Parameters:
        forecast_summary: Dictionary containing forecast summary data with locked schema
        
    Returns:
        List containing a single document dictionary with the complete forecast object
    """
    documents = []
    
    if not forecast_summary:
        return documents
    
    # Create a single comprehensive document with the complete forecast object
    # Format as JSON-like structure for the LLM to parse exactly
    state = forecast_summary.get('state', 'Unknown')
    horizon = forecast_summary.get('horizon_days', 0)
    start_date = forecast_summary.get('start_date', '')
    end_date = forecast_summary.get('end_date', '')
    avg_demand = forecast_summary.get('avg_demand', 0)
    peak_demand = forecast_summary.get('peak_demand', 0)
    peak_date = forecast_summary.get('peak_date', '')
    min_demand = forecast_summary.get('min_demand', 0)
    high_risk_days_count = forecast_summary.get('high_risk_days_count', 0)
    high_risk_dates = forecast_summary.get('high_risk_dates', [])
    extreme_heat_dates = forecast_summary.get('extreme_heat_dates', [])
    rmse = forecast_summary.get('rmse')
    confidence_level = forecast_summary.get('confidence_level', 0.9)
    
    # Build comprehensive forecast text with exact values
    forecast_text = (
        f"FORECAST SUMMARY FOR {state.upper()}:\n"
        f"Forecast period: {start_date} to {end_date} ({horizon} days)\n"
        f"Average daily demand: {avg_demand} MU\n"
        f"Peak demand: {peak_demand} MU on {peak_date}\n"
        f"Minimum demand: {min_demand} MU\n"
    )
    
    if rmse is not None:
        forecast_text += f"Model RMSE for {state}: {rmse} MU\n"
    
    forecast_text += f"Confidence level: {confidence_level}\n"
    
    if high_risk_days_count > 0:
        forecast_text += f"High-risk days count: {high_risk_days_count}\n"
        if high_risk_dates:
            forecast_text += f"High-risk dates: {', '.join(high_risk_dates)}\n"
    else:
        forecast_text += "High-risk days count: 0\n"
        forecast_text += "High-risk dates: []\n"
    
    if extreme_heat_dates:
        forecast_text += f"Extreme heat dates: {', '.join(extreme_heat_dates)}\n"
    else:
        forecast_text += "Extreme heat dates: []\n"
    
    # Add structured JSON representation for precise parsing
    import json
    forecast_json = json.dumps(forecast_summary, indent=2)
    forecast_text += f"\nSTRUCTURED FORECAST DATA:\n{forecast_json}"
    
    documents.append({
        'text': forecast_text,
        'metadata': {
            'type': 'forecast',
            'state': state,
            'horizon_days': horizon,
            'start_date': start_date,
            'end_date': end_date,
            'forecast_summary': forecast_summary  # Include full object in metadata
        }
    })
    
    return documents


def build_platform_knowledge_documents() -> List[Dict]:
    """
    Build static domain & platform architecture knowledge documents covering
    All-India aggregation, Cooling Degree Days (CDD), Prediction Ledger,
    and the MLOps Drift Detection & Adaptive Retraining Engine.
    """
    return [
        {
            'text': (
                "All-India National Aggregate Demand View: Loadly computes All-India electricity demand "
                "using bottom-up hierarchical summation across all 36 Indian states and union territories, "
                "grouped into India's 5 regional grids — Northern Region (NR), Western Region (WR), "
                "Southern Region (SR), Eastern Region (ER), and North-Eastern Region (NER). "
                "National portfolio uncertainty is calculated via quadrature (root-sum-of-squares of state RMSEs, "
                "typically ~31.5 MU for 1-sigma, giving ±51.8 MU for a 90% confidence band) rather than naive addition, "
                "reflecting cross-state error diversification."
            ),
            'metadata': {'type': 'platform_architecture', 'topic': 'all_india_national'}
        },
        {
            'text': (
                "Cooling Degree Days (CDD) & Extreme Heat Methodology: Cooling Degree Days (CDD) measure "
                "weather-driven cooling load using a base threshold of 24°C (297.15 K): "
                "CDD = max(0, T_mean_celsius - 24). When daily mean temperature exceeds 24°C, every additional "
                "degree increases air-conditioning and refrigeration demand. Extreme heat flags are triggered "
                "above the 95th historical percentile (~32°C / 305.2 K), identifying heatwave surge risk days."
            ),
            'metadata': {'type': 'methodology', 'topic': 'cdd_extreme_heat'}
        },
        {
            'text': (
                "Prediction Ledger & Forecast Verification Engine: Every forecast generated by Loadly "
                "(both state-level and All-India National) is logged into an immutable SQLite Prediction Ledger "
                "(data/prediction_ledger.db). A continuous ingestion worker pulls ground-truth grid demand actuals "
                "and reconciles them against prior predictions to compute real-time MAE, RMSE, MAPE (%), "
                "Mean Bias Error (MBE), and empirical 90% Confidence Interval hit rate."
            ),
            'metadata': {'type': 'platform_architecture', 'topic': 'prediction_ledger'}
        },
        {
            'text': (
                "MLOps Model Drift Detection & Adaptive Retraining Engine: Loadly continuously monitors model health "
                "across 4 drift pillars: (1) Trailing 14-day MAPE decay (Healthy <= 3.5%, Warning 3.5-5.0%, Critical > 5.0%), "
                "(2) Directional Bias Z-score (|z| >= 1.96 indicates systematic under- or over-prediction), "
                "(3) Empirical 90% CI Coverage degradation (Healthy >= 80%, Critical < 70%), and "
                "(4) Population Stability Index (PSI) for temperature/CDD covariate shift (PSI >= 0.25 indicates severe shift). "
                "Adaptive retraining uses a Champion vs. Challenger validation gate (requiring Challenger RMSE within 2% of Champion, "
                "R² >= 0.985, and CI coverage >= 85%), atomic file swap, zero-downtime in-memory hot reload, "
                "and 1-click rollback to best_model_prev.pkl."
            ),
            'metadata': {'type': 'platform_architecture', 'topic': 'mlops_drift_retraining'}
        }
    ]
