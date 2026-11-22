# ============================================================================
# PYTEST FIXTURES & CONFIGURATION
# ============================================================================
import pytest
import pandas as pd
import numpy as np
from models import load_model, load_model_metadata


@pytest.fixture
def sample_weather_df():
    """Weather data for 10 days (realistic range: 290-320 K)."""
    dates = pd.date_range('2023-01-01', periods=10, freq='D')
    temps = np.linspace(290, 310, 10)  # 17°C to 37°C

    return pd.DataFrame({
        'date': dates,
        '2m_temperature_max': temps + 5,
        '2m_temperature_min': temps - 5,
        '2m_temperature_mean': temps,
        '2m_dewpoint_temperature_min': temps - 10,
        '2m_dewpoint_temperature_mean': temps - 5,
        '2m_dewpoint_temperature_max': temps,
        '10m_u_component_of_wind_min': np.zeros(10),
        '10m_u_component_of_wind_mean': np.ones(10),
        '10m_u_component_of_wind_max': 2 * np.ones(10),
        '10m_v_component_of_wind_min': np.zeros(10),
        '10m_v_component_of_wind_mean': np.ones(10),
        '10m_v_component_of_wind_max': 2 * np.ones(10),
        'surface_solar_radiation_downwards_min': np.zeros(10),
        'surface_solar_radiation_downwards_mean': 100 * np.ones(10),
        'surface_solar_radiation_downwards_max': 200 * np.ones(10),
        'total_cloud_cover_min': np.zeros(10),
        'total_cloud_cover_mean': 50 * np.ones(10),
        'total_cloud_cover_max': 100 * np.ones(10),
        'utci_min': temps - 5,
        'utci_mean': temps,
        'utci_max': temps + 5,
    })


@pytest.fixture
def single_day_weather_df():
    """Single-day weather data (for lag edge cases)."""
    return pd.DataFrame({
        'date': pd.to_datetime(['2023-01-01']),
        '2m_temperature_max': [300.0],
        '2m_temperature_min': [290.0],
        '2m_temperature_mean': [295.0],
        '2m_dewpoint_temperature_min': [285.0],
        '2m_dewpoint_temperature_mean': [290.0],
        '2m_dewpoint_temperature_max': [295.0],
        '10m_u_component_of_wind_min': [0.0],
        '10m_u_component_of_wind_mean': [1.0],
        '10m_u_component_of_wind_max': [2.0],
        '10m_v_component_of_wind_min': [0.0],
        '10m_v_component_of_wind_mean': [1.0],
        '10m_v_component_of_wind_max': [2.0],
        'surface_solar_radiation_downwards_min': [0.0],
        'surface_solar_radiation_downwards_mean': [100.0],
        'surface_solar_radiation_downwards_max': [200.0],
        'total_cloud_cover_min': [0.0],
        'total_cloud_cover_mean': [50.0],
        'total_cloud_cover_max': [100.0],
        'utci_min': [290.0],
        'utci_mean': [295.0],
        'utci_max': [300.0],
    })


@pytest.fixture
def metadata():
    """Model metadata with CDD and extreme heat thresholds."""
    return {
        'cdd_threshold_kelvin': 297.15,
        'extreme_heat_threshold_kelvin': 305.0,
        'state_rmse': {'Maharashtra': 4.2, 'Tamil Nadu': 3.8},
    }


@pytest.fixture
def all_states():
    """All 36 Indian states/UTs for mapping tests."""
    return [
        'Arunachal Pradesh', 'Assam', 'Bihar', 'Chandigarh', 'Chhattisgarh',
        'Goa', 'Gujarat', 'Haryana', 'Himachal Pradesh', 'Jammu and Kashmir',
        'Jharkhand', 'Karnataka', 'Kerala', 'Madhya Pradesh', 'Maharashtra',
        'Manipur', 'Meghalaya', 'Mizoram', 'Nagaland', 'Odisha',
        'Punjab', 'Rajasthan', 'Sikkim', 'Tamil Nadu', 'Telangana',
        'Tripura', 'Uttar Pradesh', 'Uttarakhand', 'West Bengal',
        'NCT of Delhi', 'Puducherry', 'Daman and Diu', 'Dadra and Nagar Haveli',
        'Lakshadweep', 'Andaman and Nicobar'
    ]


@pytest.fixture
def loaded_model():
    """Load the actual trained XGBoost model."""
    try:
        return load_model()
    except:
        pytest.skip("Model file not found - skipping model-dependent tests")


@pytest.fixture
def loaded_metadata():
    """Load the actual model metadata."""
    try:
        return load_model_metadata()
    except:
        return {
            'cdd_threshold_kelvin': 297.15,
            'extreme_heat_threshold_kelvin': 305.0,
        }


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_sample_data_with_temps(state, temps):
    """
    Create weather data with specific temperatures.

    Args:
        state: State name
        temps: List of temperatures (mean values in Kelvin)

    Returns:
        pd.DataFrame with weather data
    """
    dates = pd.date_range('2023-01-01', periods=len(temps))
    return pd.DataFrame({
        'date': dates,
        '2m_temperature_max': [t + 5 for t in temps],
        '2m_temperature_min': [t - 5 for t in temps],
        '2m_temperature_mean': temps,
        '2m_dewpoint_temperature_min': [t - 10 for t in temps],
        '2m_dewpoint_temperature_mean': [t - 5 for t in temps],
        '2m_dewpoint_temperature_max': temps,
        '10m_u_component_of_wind_min': np.zeros(len(temps)),
        '10m_u_component_of_wind_mean': np.ones(len(temps)),
        '10m_u_component_of_wind_max': 2 * np.ones(len(temps)),
        '10m_v_component_of_wind_min': np.zeros(len(temps)),
        '10m_v_component_of_wind_mean': np.ones(len(temps)),
        '10m_v_component_of_wind_max': 2 * np.ones(len(temps)),
        'surface_solar_radiation_downwards_min': np.zeros(len(temps)),
        'surface_solar_radiation_downwards_mean': 100 * np.ones(len(temps)),
        'surface_solar_radiation_downwards_max': 200 * np.ones(len(temps)),
        'total_cloud_cover_min': np.zeros(len(temps)),
        'total_cloud_cover_mean': 50 * np.ones(len(temps)),
        'total_cloud_cover_max': 100 * np.ones(len(temps)),
        'utci_min': [t - 5 for t in temps],
        'utci_mean': temps,
        'utci_max': [t + 5 for t in temps],
    })


# ============================================================================
# TEST ISOLATION: never touch production artifacts/ or the real prediction ledger
# ============================================================================
import shutil
from utils import daily_update, model_retrainer, prediction_ledger

_ARTIFACT_FILES_TO_SEED = ("best_model.pkl", "model_metadata.json")


@pytest.fixture(autouse=True)
def isolated_ledger_db(tmp_path, monkeypatch):
    """Point the default prediction-ledger DB at a per-test temp file."""
    monkeypatch.setattr(prediction_ledger, "DEFAULT_DB_PATH", tmp_path / "prediction_ledger.db")


@pytest.fixture(autouse=True)
def isolated_model_artifacts(tmp_path, monkeypatch):
    """
    Redirect retrain/rollback file paths to a temp dir seeded with a copy of the current
    champion + metadata (so champion evaluation still works). The training dataset is
    read-only and stays at its real location unless a test patches DATASET_PATH.
    """
    art_dir = tmp_path / "artifacts"
    art_dir.mkdir()
    for name in _ARTIFACT_FILES_TO_SEED:
        src = model_retrainer.ARTIFACTS_DIR / name
        if src.exists():
            shutil.copy2(src, art_dir / name)

    monkeypatch.setattr(model_retrainer, "MODEL_PATH", art_dir / "best_model.pkl")
    monkeypatch.setattr(model_retrainer, "MODEL_PREV_PATH", art_dir / "best_model_prev.pkl")
    monkeypatch.setattr(model_retrainer, "MODEL_CANDIDATE_PATH", art_dir / "best_model_candidate.pkl")
    monkeypatch.setattr(model_retrainer, "METADATA_PATH", art_dir / "model_metadata.json")
    monkeypatch.setattr(model_retrainer, "METADATA_PREV_PATH", art_dir / "model_metadata_prev.json")
    monkeypatch.setattr(model_retrainer, "RETRAIN_LOCK_PATH", art_dir / "retrain.lock")
    return art_dir


@pytest.fixture(autouse=True)
def isolated_data_update(tmp_path, monkeypatch):
    """Daily update job: default paths point at a temp tree (never data/ or logs/); scheduler stays off."""
    root = tmp_path / "loadly_root"
    monkeypatch.setattr(daily_update, "default_paths", lambda: daily_update.DataPaths.under(root))
    monkeypatch.setattr(daily_update, "_last_signature", None)
    monkeypatch.delenv("LOADLY_AUTO_UPDATE", raising=False)
    monkeypatch.delenv("LOADLY_UPDATE_TIME", raising=False)
    return root


# ============================================================================
# ADMIN AUTH: operator endpoints require X-Admin-Key (see api/auth.py)
# ============================================================================
TEST_ADMIN_KEY = "test-admin-key-0123456789abcdefghij"


@pytest.fixture
def admin_headers(monkeypatch):
    """Configure a valid admin key for this test and return the matching request headers."""
    monkeypatch.setenv("LOADLY_ADMIN_KEY", TEST_ADMIN_KEY)
    return {"X-Admin-Key": TEST_ADMIN_KEY}
