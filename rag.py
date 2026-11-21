# ============================================================================
# RAG FUNCTIONS
# ============================================================================
import os
import threading
import logging

from models import load_model
from visualization import get_feature_importance
from utils.embeddings import get_embeddings
from utils.vector_store import VectorStore
from utils.rag_builder import (
    load_model_metadata as load_model_metadata_rag,
    build_model_metrics_documents,
    build_feature_importance_documents,
    build_forecast_summary_documents,
    build_platform_knowledge_documents,
)
from utils.rag_engine import RAGEngine

logger = logging.getLogger(__name__)

_rag_lock = threading.Lock()
_rag_cache = {
    "vector_store": None,
    "rag_engine": None,
    "metadata_mtime": None,
}
_METADATA_PATH = "artifacts/model_metadata.json"
_RAG_INDEX_DIR = "data/rag_index"
_RAG_INDEX_PATH = os.path.join(_RAG_INDEX_DIR, "knowledge_base.index")


def invalidate_rag_cache() -> None:
    """Clear the cached RAG vector store and engine (e.g., after model retraining)."""
    with _rag_lock:
        _rag_cache["vector_store"] = None
        _rag_cache["rag_engine"] = None
        _rag_cache["metadata_mtime"] = None


def generate_forecast_summary(results_df, state, horizon_days, df_features=None, metadata=None):
    """
    Generate structured forecast summary for RAG system.
    Creates a single locked object per forecast with exact values.

    Parameters:
        results_df: DataFrame with forecast results
        state (str): State name
        horizon_days (int): Forecast horizon in days
        df_features: Optional DataFrame with features
        metadata: Optional model metadata for RMSE and confidence_level

    Returns:
        dict: Forecast summary with locked schema, or None
    """
    if results_df is None or len(results_df) == 0:
        return None

    state_rmse = None
    confidence_level = 0.9
    if metadata:
        state_rmse_dict = metadata.get('state_rmse', {})
        state_rmse = state_rmse_dict.get(state)
        if state_rmse is None and state == "Jammu and Kashmir":
            state_rmse = state_rmse_dict.get("J&K")
        confidence_level = metadata.get('confidence_level', 0.9)

    high_risk_dates = []
    extreme_heat_dates = []

    if 'extreme_heat' in results_df.columns:
        extreme_heat_rows = results_df[results_df['extreme_heat'] == 1]
        extreme_heat_dates = [row['date'].strftime('%Y-%m-%d') for _, row in extreme_heat_rows.iterrows()]

        high_demand_threshold = results_df['forecasted_demand_MU'].quantile(0.9)
        high_risk_rows = results_df[
            (results_df['extreme_heat'] == 1) |
            (results_df['forecasted_demand_MU'] >= high_demand_threshold)
        ]
        high_risk_dates = [row['date'].strftime('%Y-%m-%d') for _, row in high_risk_rows.iterrows()]

    summary = {
        'state': state,
        'horizon_days': int(horizon_days),
        'start_date': results_df['date'].min().strftime('%Y-%m-%d'),
        'end_date': results_df['date'].max().strftime('%Y-%m-%d'),
        'avg_demand': round(float(results_df['forecasted_demand_MU'].mean()), 1),
        'peak_demand': round(float(results_df['forecasted_demand_MU'].max()), 1),
        'peak_date': results_df.loc[results_df['forecasted_demand_MU'].idxmax(), 'date'].strftime('%Y-%m-%d'),
        'min_demand': round(float(results_df['forecasted_demand_MU'].min()), 1),
        'high_risk_days_count': len(high_risk_dates),
        'high_risk_dates': high_risk_dates,
        'extreme_heat_dates': extreme_heat_dates,
        'rmse': round(float(state_rmse), 2) if state_rmse is not None else None,
        'confidence_level': confidence_level
    }

    return summary


def _get_metadata_mtime():
    return os.path.getmtime(_METADATA_PATH) if os.path.exists(_METADATA_PATH) else None


def initialize_rag_system(force_rebuild: bool = False):
    """
    Initialize and cache the RAG system with a thread-safe singleton.
    Automatically rebuilds the vector index if artifacts/model_metadata.json
    has been updated (e.g., after MLOps adaptive retraining).

    Returns:
        Tuple of (VectorStore, RAGEngine) or (None, None) if initialization fails
    """
    current_mtime = _get_metadata_mtime()

    with _rag_lock:
        if (
            not force_rebuild
            and _rag_cache["rag_engine"] is not None
            and _rag_cache["metadata_mtime"] == current_mtime
        ):
            return _rag_cache["vector_store"], _rag_cache["rag_engine"]

        try:
            os.makedirs(_RAG_INDEX_DIR, exist_ok=True)
            index_exists = os.path.exists(_RAG_INDEX_PATH)
            index_mtime = os.path.getmtime(_RAG_INDEX_PATH) if index_exists else None

            # Rebuild if forced, missing, or stale relative to model_metadata.json
            needs_rebuild = (
                force_rebuild
                or not index_exists
                or (current_mtime is not None and index_mtime is not None and current_mtime > index_mtime)
            )

            if not needs_rebuild:
                vector_store = VectorStore(dimension=384, index_path=_RAG_INDEX_PATH)
                if vector_store.get_size() > 0:
                    rag_engine = RAGEngine(vector_store)
                    _rag_cache["vector_store"] = vector_store
                    _rag_cache["rag_engine"] = rag_engine
                    _rag_cache["metadata_mtime"] = current_mtime
                    return vector_store, rag_engine

            # Fresh VectorStore build
            vector_store = VectorStore(dimension=384, index_path=None)
            vector_store.index_path = _RAG_INDEX_PATH

            docs = build_platform_knowledge_documents()

            try:
                metadata = load_model_metadata_rag(_METADATA_PATH)
                docs.extend(build_model_metrics_documents(metadata))

                model = load_model()
                if model is not None:
                    feature_importance = get_feature_importance(model, top_n=15)
                    if feature_importance:
                        docs.extend(build_feature_importance_documents(feature_importance))
            except FileNotFoundError:
                logger.warning("[RAG] Model metadata file not found; indexing platform docs only.")
            except Exception as e:
                logger.warning(f"[RAG] Error building model docs: {e}")

            if docs:
                texts = [doc['text'] for doc in docs]
                metadata_list = [doc['metadata'] for doc in docs]
                try:
                    embeddings = get_embeddings(texts)
                    vector_store.add_documents(embeddings, texts, metadata_list)
                    vector_store.save_index()
                except Exception as e:
                    logger.error(f"[RAG] Could not initialize embeddings: {e}")
                    return None, None

            rag_engine = RAGEngine(vector_store)
            _rag_cache["vector_store"] = vector_store
            _rag_cache["rag_engine"] = rag_engine
            _rag_cache["metadata_mtime"] = current_mtime
            return vector_store, rag_engine

        except Exception as e:
            logger.error(f"[RAG] Error initializing RAG system: {e}")
            return None, None
