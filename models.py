# ============================================================================
# MODEL LOADING FUNCTIONS
# ============================================================================
import os
import pickle
import json
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)


def _is_streamlit_running():
    try:
        from streamlit.runtime import exists as st_exists
        return st_exists()
    except Exception:
        return False


def cache_model_data(func):
    """Decorator that uses Streamlit cache if running in Streamlit, else lru_cache."""
    if _is_streamlit_running():
        import streamlit as st
        return st.cache_data(func)
    return lru_cache(maxsize=4)(func)


@cache_model_data
def load_model():
    """
    Load the trained forecasting model (Reduced_LR best_model.pkl or fallback).

    Returns:
        model: Trained model object or None if file not found
    """
    possible_paths = [
        'artifacts/best_model.pkl',
        'best_model.pkl',
        '../artifacts/best_model.pkl',
        'artifacts/xgb2_model.pkl',
        'xgb2_model.pkl'
    ]

    for model_path in possible_paths:
        if os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                return model
            except Exception as e:
                logger.error(f"Error loading model from {model_path}: {str(e)}")
                if _is_streamlit_running():
                    import streamlit as st
                    st.error(f"Error loading model from {model_path}: {str(e)}")
                continue

    msg = "Model file not found. Please check artifacts/best_model.pkl path."
    logger.error(msg)
    if _is_streamlit_running():
        import streamlit as st
        st.error(msg)
    return None


@cache_model_data
def load_model_metadata():
    """
    Load model metadata including state-wise RMSE values.

    Returns:
        dict: Model metadata dictionary or None if file not found
    """
    possible_paths = [
        'artifacts/model_metadata.json',
        'model_metadata.json',
        '../artifacts/model_metadata.json',
        '../model_metadata.json',
        './model_metadata.json'
    ]

    for metadata_path in possible_paths:
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                return metadata
            except Exception as e:
                logger.warning(f"Error loading metadata from {metadata_path}: {str(e)}")
                continue

    return None


def get_state_rmse(state_name, metadata):
    """
    Get state-specific RMSE from metadata.

    Parameters:
        state_name (str): Name of the state
        metadata (dict): Model metadata dictionary

    Returns:
        float: RMSE value for the state, or None if not found
    """
    if metadata is None or 'state_rmse' not in metadata:
        return None

    state_rmse_dict = metadata['state_rmse']

    if state_name in state_rmse_dict:
        return state_rmse_dict[state_name]

    # Handle state name variations
    if state_name == "Jammu and Kashmir":
        if "J&K" in state_rmse_dict:
            return state_rmse_dict["J&K"]
    elif state_name == "J&K":
        if "Jammu and Kashmir" in state_rmse_dict:
            return state_rmse_dict["Jammu and Kashmir"]

    return None


NATIONAL_NAMES = ("All-India", "National")


def _lookup_horizon(table, horizon):
    """
    Look up horizon `horizon` (1-based) in a {"1": rmse, ...} table.

    Returns the running maximum over horizons <= `horizon`: forecast error grows with
    lead time, so backtest noise must not let an interval narrow as the horizon grows.
    Horizons beyond the table's last entry reuse the last (largest) value.
    """
    if not table:
        return None
    try:
        points = sorted((int(k), float(v)) for k, v in table.items())
    except (TypeError, ValueError):
        return None
    horizon = max(1, int(horizon))
    usable = [v for h, v in points if h <= horizon] or [points[0][1]]
    return max(usable)


def _state_aliases(state_name):
    if state_name == "Jammu and Kashmir":
        return ("Jammu and Kashmir", "J&K")
    if state_name == "J&K":
        return ("J&K", "Jammu and Kashmir")
    return (state_name,)


def get_horizon_rmse(state_name, horizon, metadata):
    """
    Forecast RMSE (MU) for `state_name` at lead time `horizon` days (1 = next day).

    Lookup order:
      1. metadata["state_horizon_rmse"][state][horizon]  (state-specific backtest)
      2. metadata["horizon_rmse"][horizon] scaled by state_rmse / horizon-1 overall RMSE
      3. get_state_rmse(): the flat one-step RMSE (legacy behaviour)
      4. None
    "All-India"/"National" uses metadata["national_horizon_rmse"] when present;
    callers otherwise combine per-state values themselves.
    """
    if not metadata:
        return None

    if state_name in NATIONAL_NAMES:
        return _lookup_horizon(metadata.get("national_horizon_rmse"), horizon)

    per_state = metadata.get("state_horizon_rmse") or {}
    for alias in _state_aliases(state_name):
        if alias in per_state:
            value = _lookup_horizon(per_state[alias], horizon)
            if value is not None:
                return value

    state_rmse = get_state_rmse(state_name, metadata)
    overall_h = _lookup_horizon(metadata.get("horizon_rmse"), horizon)
    overall_1 = _lookup_horizon(metadata.get("horizon_rmse"), 1)
    if state_rmse is not None and overall_h is not None and overall_1:
        return overall_h * float(state_rmse) / overall_1

    return state_rmse


def get_horizon_rmse_series(state_name, n_days, metadata):
    """List of get_horizon_rmse for horizons 1..n_days, or None if any is unavailable."""
    values = [get_horizon_rmse(state_name, h, metadata) for h in range(1, n_days + 1)]
    if any(v is None for v in values):
        return None
    return values


def reload_model_cache():
    """Clear memory caches for model and model metadata for zero-downtime hot reload."""
    if hasattr(load_model, "cache_clear"):
        load_model.cache_clear()
    if hasattr(load_model_metadata, "cache_clear"):
        load_model_metadata.cache_clear()
    logger.info("Cleared in-memory model and metadata caches for hot reload.")

