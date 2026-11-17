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


def reload_model_cache():
    """Clear memory caches for model and model metadata for zero-downtime hot reload."""
    if hasattr(load_model, "cache_clear"):
        load_model.cache_clear()
    if hasattr(load_model_metadata, "cache_clear"):
        load_model_metadata.cache_clear()
    logger.info("Cleared in-memory model and metadata caches for hot reload.")

