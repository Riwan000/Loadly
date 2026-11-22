# ============================================================================
# FEATURE ENGINEERING FUNCTIONS
# ============================================================================
import logging
import pandas as pd
import numpy as np
import holidays
from dataclasses import dataclass
from typing import Optional
from config import STATE_REGIONS, STALE_DEMAND_DAYS
from data_loading import (
    get_state_historical_averages,
    load_historical_demand,
    load_historical_data,
    get_fallback_baseline
)

# ----------------------------------------------------------------------------
# SHARED FEATURE LISTS (single source of truth for training, retraining, serving)
# ----------------------------------------------------------------------------
# Demand features are derived from y(t-1) and earlier only (strictly no y(t)).
# Rolling/change features: see utils/train_pipeline.add_demand_lag_features.
DEMAND_LAGS = (1, 7, 14)
DEMAND_LAG_FEATURE_NAMES = [f'demand_lag_{k}' for k in DEMAND_LAGS]
# Weather/calendar features (known for the target day, or lagged weather).
STATIC_FEATURE_NAMES = [
    '2m_temperature_mean', '2m_temperature_max', 'temperature_lag_1',
    'temp_range', 'CDD', 'month', 'day_of_week', 'is_weekend',
]
# The 15-feature set of the original production model (no direct demand lag: it cannot
# reconstruct yesterday's level). Kept so legacy artifacts still load and as the
# backtest reference.
LEGACY_FEATURE_NAMES = [
    'demand_roll_mean_7', 'demand_roll_mean_14', 'demand_roll_mean_30',
    'demand_roll_std_7', 'demand_roll_std_30', 'daily_change', 'daily_pct_change',
    *STATIC_FEATURE_NAMES,
]
# Features of the current production model family: legacy set + direct demand lags.
# utils/train_pipeline.py and utils/model_retrainer.py both import THIS list.
PRODUCTION_LAG_FEATURE_NAMES = ['demand_lag_1', 'demand_lag_7']
PRODUCTION_FEATURE_NAMES = LEGACY_FEATURE_NAMES + PRODUCTION_LAG_FEATURE_NAMES


def engineer_features(df, state_name, metadata=None, historical_demand=None):
    """
    Engineer features for demand forecasting models.

    This function calculates calendar features, holiday flags, cooling degree days,
    temperature lags/ranges/interactions, state/region assignments, and 
    historical demand rolling statistics (mean, std, daily changes) needed by the
    15-feature Best Model (Reduced Linear Regression) and XGBoost models.

    Parameters:
        df (pd.DataFrame): Weather data with date column
        state_name (str): Name of the state
        metadata (dict, optional): model_metadata.json contents. Used to read
            train-derived thresholds (cdd_threshold_kelvin,
            extreme_heat_threshold_kelvin) so inference matches training exactly.
        historical_demand (pd.DataFrame, optional): Pre-loaded historical demand data.

    Returns:
        pd.DataFrame: Data with all engineered features added
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])

    # -----------------------------
    # 1. Calendar features
    # -----------------------------
    df['day_of_week'] = df['date'].dt.dayofweek
    df['dayofweek'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['day_of_year'] = df['date'].dt.dayofyear

    # Weekend flags
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

    # Season encoding
    def get_season(month):
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Summer'
        elif month in [6, 7, 8, 9]:
            return 'Monsoon'
        else:
            return 'Autumn'

    df['season'] = df['month'].apply(get_season)

    # -----------------------------
    # 2. Holiday detection (India)
    # -----------------------------
    years = df['date'].dt.year.unique()
    valid_years = [int(y) for y in years if 2010 <= y <= 2035]
    if not valid_years:
        valid_years = range(2010, 2030)
    india_holidays = holidays.India(years=valid_years)
    df['holiday_name'] = df['date'].apply(lambda x: india_holidays.get(x.date() if hasattr(x, 'date') else x))
    df['holiday_name'] = df['holiday_name'].fillna('No Holiday')
    df['is_holiday'] = (df['holiday_name'] != 'No Holiday').astype(int)

    # -----------------------------
    # 3. State & Region assignment
    # -----------------------------
    normalized_state = state_name
    if state_name == 'Jammu and Kashmir':
        normalized_state = 'J&K'

    df['states'] = normalized_state
    df['region'] = STATE_REGIONS.get(state_name, 'nr')

    # -----------------------------
    # 4. Temperature & CDD features
    # -----------------------------
    threshold_temp = (metadata.get('cdd_threshold_kelvin', 297.15)
                      if metadata else 297.15)
    df['cdd'] = np.maximum(0, df['2m_temperature_mean'] - threshold_temp)
    df['CDD'] = df['cdd']

    # Extreme heat flag
    if metadata and 'extreme_heat_threshold_kelvin' in metadata:
        heat_threshold = metadata['extreme_heat_threshold_kelvin']
    else:
        heat_threshold = 305.0  # fixed historical threshold
    df['extreme_heat'] = (df['2m_temperature_mean'] > heat_threshold).astype(int)
    df['extreme_heat_flag'] = df['extreme_heat'].astype(int)

    # Temperature interaction features
    df['temp_min_heat_interaction'] = df['2m_temperature_min'] * df['extreme_heat_flag']
    df['temp_mean_heat_interaction'] = df['2m_temperature_mean'] * df['extreme_heat_flag']
    df['temp_max_heat_interaction'] = df['2m_temperature_max'] * df['extreme_heat_flag']

    # Temperature range
    df['temp_range'] = df['2m_temperature_max'] - df['2m_temperature_min']
    df['temp_range_heat'] = df['temp_range'] * df['extreme_heat_flag']

    # -----------------------------
    # 5. Weather Lag & Rolling features
    # -----------------------------
    # For lag 1: shift(1) with historical weather fallback for the first row
    hist_weather = load_historical_data(state_name)
    temp_lag1_vals = []
    for i, row in df.iterrows():
        if i == 0:
            prior_date = row['date'] - pd.Timedelta(days=1)
            found_prior = False
            if hist_weather is not None and not hist_weather.empty:
                prior_rows = hist_weather[hist_weather['date'] == prior_date]
                if len(prior_rows) > 0:
                    temp_lag1_vals.append(float(prior_rows['2m_temperature_mean'].values[0]))
                    found_prior = True
            if not found_prior:
                temp_lag1_vals.append(float(row['2m_temperature_mean']))
        else:
            temp_lag1_vals.append(float(df['2m_temperature_mean'].iloc[i-1]))

    df['temperature_lag_1'] = temp_lag1_vals

    for lag in [1, 3, 7]:
        df[f'temp_mean_lag{lag}'] = df['2m_temperature_mean'].shift(lag).bfill()

    df['temp_mean_roll3'] = df['2m_temperature_mean'].rolling(window=3, min_periods=1).mean()
    df['temp_mean_roll7'] = df['2m_temperature_mean'].rolling(window=7, min_periods=1).mean()
    df['temp_max_roll3'] = df['2m_temperature_max'].rolling(window=3, min_periods=1).mean()
    df['temp_max_roll7'] = df['2m_temperature_max'].rolling(window=7, min_periods=1).mean()
    df['temp_min_roll3'] = df['2m_temperature_min'].rolling(window=3, min_periods=1).mean()
    df['temp_min_roll7'] = df['2m_temperature_min'].rolling(window=7, min_periods=1).mean()

    # -----------------------------
    # 6. Demand Rolling Statistics (for Best Model 15 features)
    # -----------------------------
    if historical_demand is None:
        historical_demand = load_historical_demand(state_name)

    fallback_baseline = get_fallback_baseline(state_name)
    
    if historical_demand is not None and len(historical_demand) > 0:
        hist_sorted = historical_demand.sort_values('date')
        first_forecast_date = df['date'].iloc[0]
        prior_demand = hist_sorted[hist_sorted['date'] < first_forecast_date]
        if len(prior_demand) >= 30:
            demands = list(prior_demand['actual_demand_MU'].values)
        else:
            demands = list(hist_sorted['actual_demand_MU'].values)
    else:
        demands = []

    roll_7, roll_14, roll_30 = [], [], []
    std_7, std_30 = [], []
    d_change, d_pct_change = [], []

    curr_demands = demands.copy()
    for _ in range(len(df)):
        if len(curr_demands) >= 30:
            r7 = float(np.mean(curr_demands[-7:]))
            r14 = float(np.mean(curr_demands[-14:]))
            r30 = float(np.mean(curr_demands[-30:]))
            s7 = float(np.std(curr_demands[-7:], ddof=1)) if len(curr_demands[-7:]) > 1 else 0.0
            s30 = float(np.std(curr_demands[-30:], ddof=1)) if len(curr_demands[-30:]) > 1 else 0.0
            dc = float(curr_demands[-1] - curr_demands[-2]) if len(curr_demands) >= 2 else 0.0
            dp = float((curr_demands[-1] - curr_demands[-2]) / curr_demands[-2]) if len(curr_demands) >= 2 and curr_demands[-2] != 0 else 0.0
        elif len(curr_demands) > 0:
            m_val = float(np.mean(curr_demands))
            r7 = r14 = r30 = m_val
            s7 = s30 = float(np.std(curr_demands, ddof=1)) if len(curr_demands) > 1 else 0.0
            dc = 0.0
            dp = 0.0
        else:
            r7 = r14 = r30 = fallback_baseline
            s7 = fallback_baseline * 0.05
            s30 = fallback_baseline * 0.08
            dc = 0.0
            dp = 0.0

        roll_7.append(r7)
        roll_14.append(r14)
        roll_30.append(r30)
        std_7.append(s7)
        std_30.append(s30)
        d_change.append(dc)
        d_pct_change.append(dp)

    df['demand_roll_mean_7'] = roll_7
    df['demand_roll_mean_14'] = roll_14
    df['demand_roll_mean_30'] = roll_30
    df['demand_roll_std_7'] = std_7
    df['demand_roll_std_30'] = std_30
    df['daily_change'] = d_change
    df['daily_pct_change'] = d_pct_change

    # Direct demand lags (same values on every row: this path has no recursion; the
    # recursive path in predict_recursive overwrites them per day).
    lag_buffer = np.full((1, DEMAND_BUFFER_LEN), np.nan)
    seed = np.asarray(demands, dtype=float)[-DEMAND_BUFFER_LEN:]
    if len(seed):
        lag_buffer[0, -len(seed):] = seed
        lag_values = demand_features_from_buffer(lag_buffer)
        for name in DEMAND_LAG_FEATURE_NAMES:
            df[name] = float(lag_values[name][0])
    else:
        for name in DEMAND_LAG_FEATURE_NAMES:
            df[name] = fallback_baseline

    # -----------------------------
    # 7. Legacy state historical averages
    # -----------------------------
    hist = get_state_historical_averages(state_name)
    df['energymet_7d_avg'] = hist.get('energymet_7d_avg', 0.0)
    df['monthly_energy_avg'] = hist.get('monthly_energy_avg', 0.0)
    df['monthly_rolling_mean'] = hist.get('monthly_rolling_mean', 0.0)
    df['demand_rolling'] = hist.get('demand_rolling', 0.0)
    df['state_share'] = hist.get('state_share', 0.0)
    df['max.demand met during the day(mw)'] = hist.get('max_demand_mw', 0.0)
    df['generation_mu'] = 0.0
    df['generation_7d_avg'] = 0.0
    df['monthly_generation_avg'] = 0.0
    df['gen_rolling'] = 0.0

    return df


def prepare_features_for_prediction(weather_df, state_name, model=None, metadata=None, historical_demand=None):
    """
    Prepare the exact feature set the model expects.

    Ensures that the features match exactly what the model was trained on.
    Feature names and order match the model requirements (15 features for
    Reduced Linear Regression Best Model, or 72+ features for XGBoost).

    Parameters:
        weather_df (pd.DataFrame): Raw weather data
        state_name (str): Name of the state
        model: Optional loaded model (used to extract exact feature order)
        metadata (dict): Optional model metadata
        historical_demand (pd.DataFrame, optional): Pre-loaded historical demand data

    Returns:
        tuple: (X_features, df_features)
            - X_features: DataFrame with only model features in correct order
            - df_features: Full dataframe with all engineered features
    """
    df_features = engineer_features(weather_df, state_name, metadata=metadata, historical_demand=historical_demand)

    categorical_cols = ['states', 'region', 'season', 'holiday_name']

    season_values = df_features['season'].copy() if 'season' in df_features.columns else None
    holiday_values = df_features['holiday_name'].copy() if 'holiday_name' in df_features.columns else None

    # Determine expected features
    if model is not None and hasattr(model, 'feature_names_in_'):
        expected_features = list(model.feature_names_in_)
    else:
        # Default to full feature set with one-hot encodings for backward compatibility
        expected_features_base = [
            '10m_u_component_of_wind_min', '10m_u_component_of_wind_mean', '10m_u_component_of_wind_max',
            '10m_v_component_of_wind_min', '10m_v_component_of_wind_mean', '10m_v_component_of_wind_max',
            '2m_temperature_min', '2m_temperature_mean', '2m_temperature_max',
            '2m_dewpoint_temperature_min', '2m_dewpoint_temperature_mean', '2m_dewpoint_temperature_max',
            'surface_solar_radiation_downwards_min', 'surface_solar_radiation_downwards_mean', 'surface_solar_radiation_downwards_max',
            'total_cloud_cover_min', 'total_cloud_cover_mean', 'total_cloud_cover_max',
            'utci_min', 'utci_mean', 'utci_max',
            'max.demand met during the day(mw)', 'generation_mu', 'state_share',
            'month', 'day_of_week', 'is_weekend', 'energymet_7d_avg', 'generation_7d_avg',
            'monthly_generation_avg', 'monthly_energy_avg', 'dayofweek', 'weekend',
            'monthly_rolling_mean', 'demand_rolling', 'gen_rolling', 'year',
            'extreme_heat', 'extreme_heat_flag', 'temp_min_heat_interaction',
            'temp_mean_heat_interaction', 'temp_max_heat_interaction'
        ]
        all_states_ordered = [
            'Arunachal Pradesh', 'Assam', 'Bihar', 'Chandigarh', 'Chhattisgarh',
            'Goa', 'Gujarat', 'Haryana', 'J&K', 'Jharkhand', 'Karnataka', 'Kerala',
            'Maharashtra', 'Manipur', 'Meghalaya', 'Mizoram', 'Nagaland', 'Odisha',
            'Puducherry', 'Punjab', 'Rajasthan', 'Sikkim', 'Tamil Nadu', 'Telangana',
            'Tripura', 'Uttarakhand', 'West Bengal'
        ]
        regions_ordered = ['ner', 'nr', 'sr', 'wr']
        seasons_ordered = ['Monsoon', 'Summer', 'Winter']
        holidays_ordered = ['Dussehra', 'Guru Nanak Jayanti', 'No Holiday']

        expected_features = expected_features_base.copy()
        expected_features.extend([f'states_{s}' for s in all_states_ordered])
        expected_features.extend([f'region_{r}' for r in regions_ordered])
        expected_features.extend([f'season_{s}' for s in seasons_ordered])
        expected_features.extend([f'holiday_name_{h}' for h in holidays_ordered])

    # Check if one-hot encoding is needed (i.e. if any expected feature starts with a categorical prefix)
    needs_one_hot = any(
        feat.startswith(('states_', 'region_', 'season_', 'holiday_name_'))
        for feat in expected_features
    )

    if needs_one_hot:
        for col in categorical_cols:
            if col in df_features.columns:
                dummies = pd.get_dummies(df_features[col], prefix=col)
                df_features = pd.concat([df_features, dummies], axis=1)

        # Restore season and holiday_name columns
        if season_values is not None:
            df_features['season'] = season_values.values
        if holiday_values is not None:
            df_features['holiday_name'] = holiday_values.values

        # Ensure all state one-hot columns are present
        all_states_ordered = [
            'Arunachal Pradesh', 'Assam', 'Bihar', 'Chandigarh', 'Chhattisgarh',
            'Goa', 'Gujarat', 'Haryana', 'J&K', 'Jharkhand', 'Karnataka', 'Kerala',
            'Maharashtra', 'Manipur', 'Meghalaya', 'Mizoram', 'Nagaland', 'Odisha',
            'Puducherry', 'Punjab', 'Rajasthan', 'Sikkim', 'Tamil Nadu', 'Telangana',
            'Tripura', 'Uttarakhand', 'West Bengal'
        ]
        for state in all_states_ordered:
            col_name = f'states_{state}'
            if col_name not in df_features.columns:
                df_features[col_name] = 0
            if state_name == state or (state_name == 'Jammu and Kashmir' and state == 'J&K'):
                df_features[col_name] = 1

        current_region = STATE_REGIONS.get(state_name, 'nr')
        # Legacy models (xgb1/xgb2) were trained with one-hot regions ner/nr/sr/wr and no
        # 'er' column, so Eastern states are collapsed onto 'nr'. Known limitation; the
        # current production model has no region features and never reaches this branch.
        if current_region == 'er':
            current_region = 'nr'
        regions_ordered = ['ner', 'nr', 'sr', 'wr']
        for region in regions_ordered:
            col_name = f'region_{region}'
            if col_name not in df_features.columns:
                df_features[col_name] = 0
            if region == current_region:
                df_features[col_name] = 1

        seasons_ordered = ['Monsoon', 'Summer', 'Winter']
        for season in seasons_ordered:
            col_name = f'season_{season}'
            if col_name not in df_features.columns:
                df_features[col_name] = 0
            if season_values is not None:
                df_features[col_name] = (season_values == season).astype(int)

        holidays_ordered = ['Dussehra', 'Guru Nanak Jayanti', 'No Holiday']
        for holiday in holidays_ordered:
            col_name = f'holiday_name_{holiday}'
            if col_name not in df_features.columns:
                df_features[col_name] = 0
            if holiday_values is not None:
                df_features[col_name] = holiday_values.apply(
                    lambda x: 1 if holiday in str(x) else 0
                )

    # Build feature matrix in exact expected order, filling missing with 0
    X = pd.DataFrame(index=df_features.index)
    for feat in expected_features:
        if feat in df_features.columns:
            X[feat] = df_features[feat]
        else:
            X[feat] = 0.0

    return X, df_features


# ============================================================================
# RECURSIVE (MULTI-STEP) PREDICTION
# ============================================================================
# The production model is a ONE-STEP-AHEAD model: its 7 demand features are built
# from y(t-1) and earlier (see utils/train_pipeline.add_demand_lag_features). To
# forecast a horizon of H days, prediction k must therefore be appended to the
# demand buffer before features for day k+1 are built.
DEMAND_FEATURE_NAMES = [
    'demand_roll_mean_7', 'demand_roll_mean_14', 'demand_roll_mean_30',
    'demand_roll_std_7', 'demand_roll_std_30',
    'daily_change', 'daily_pct_change',
    *DEMAND_LAG_FEATURE_NAMES,
]
DEMAND_BUFFER_LEN = 30  # longest rolling window (demand_roll_*_30)

logger = logging.getLogger(__name__)


def demand_features_from_buffer(buffer):
    """
    Vectorised demand features for R independent series, mirroring training exactly.

    Parameters:
        buffer (np.ndarray): shape (R, DEMAND_BUFFER_LEN). Last column is y(t-1), the
            most recent known value; shorter histories are left-padded with NaN.
            Every row needs at least one non-NaN value.

    Training definitions (add_demand_lag_features): demand_lag_k = y(t-k); rolling mean/std over the series
    shifted by one with min_periods=1 (so a window shorter than w uses what exists),
    std with ddof=1 (0.0 for a single observation), daily_change = y(t-1) - y(t-2),
    daily_pct_change = daily_change / y(t-2) with 0.0 where undefined (y(t-2) == 0).
    """
    buf = np.asarray(buffer, dtype=float)

    def _window_mean_std(w):
        win = buf[:, -w:]
        valid = ~np.isnan(win)
        count = valid.sum(axis=1)
        mean = np.where(valid, win, 0.0).sum(axis=1) / np.maximum(count, 1)
        sq_dev = np.where(valid, (win - mean[:, None]) ** 2, 0.0).sum(axis=1)
        std = np.where(count > 1, np.sqrt(sq_dev / np.maximum(count - 1, 1)), 0.0)
        return mean, std

    out = {}
    for w in (7, 14, 30):
        mean, std = _window_mean_std(w)
        out[f'demand_roll_mean_{w}'] = mean
        if w in (7, 30):
            out[f'demand_roll_std_{w}'] = std

    # Direct lags: y(t-k) = buf[:, -k]. Training leaves lag_k NaN (and drops the row) when
    # fewer than k prior days exist; a too-short serving buffer instead falls back to the
    # mean of the values that do exist (finite, no leak, never used in training).
    available_mean = _window_mean_std(buf.shape[1])[0]
    for k in DEMAND_LAGS:
        lagged = buf[:, -k]
        out[f'demand_lag_{k}'] = np.where(np.isnan(lagged), available_mean, lagged)

    last, prev = buf[:, -1], buf[:, -2]
    both = ~np.isnan(last) & ~np.isnan(prev)
    change = np.where(both, last - prev, 0.0)
    safe_prev = np.where(both & (prev != 0), prev, 1.0)
    out['daily_change'] = change
    out['daily_pct_change'] = np.where(both & (prev != 0), change / safe_prev, 0.0)
    return out


def recursive_forecast_batch(model, feature_names, static, static_names, buffer,
                             actuals=None, is_log_scale=False):
    """
    Day-by-day prediction for R independent series at once (core shared by
    predict_recursive and the training-time backtest in utils/backtest.py).

    Parameters:
        model: fitted regressor exposing predict(DataFrame)
        feature_names (list[str]): model inputs in order; demand features are computed
            here, every other name must appear in static_names
        static (np.ndarray): shape (R, H, len(static_names)) non-demand features
        static_names (list[str]): column names of the last axis of `static`
        buffer (np.ndarray): shape (R, DEMAND_BUFFER_LEN) initial demand history,
            most recent value last, NaN-left-padded
        actuals (np.ndarray, optional): shape (R, H). Where not NaN, the observed value
            (instead of the prediction) is appended to the buffer = teacher forcing.
        is_log_scale (bool): model output is log1p(MU); buffer receives expm1(output)

    Returns:
        tuple: (raw model outputs, buffer values used per day), both shape (R, H).
        The second element is the demand series the recursion actually fed back.
    """
    static = np.asarray(static, dtype=float)
    n_runs, horizon, _ = static.shape
    static_idx = {name: i for i, name in enumerate(static_names)}
    missing = [f for f in feature_names if f not in DEMAND_FEATURE_NAMES and f not in static_idx]
    if missing:
        raise ValueError(f"Features neither demand-derived nor supplied as static: {missing}")

    buf = np.array(buffer, dtype=float, copy=True)
    preds = np.zeros((n_runs, horizon))
    fed_back = np.zeros((n_runs, horizon))
    for k in range(horizon):
        demand = demand_features_from_buffer(buf)
        columns = {
            f: demand[f] if f in demand else static[:, k, static_idx[f]]
            for f in feature_names
        }
        raw = np.asarray(model.predict(pd.DataFrame(columns, columns=list(feature_names))), dtype=float)
        preds[:, k] = raw

        value_mu = np.expm1(raw) if is_log_scale else raw
        value_mu = np.maximum(0.0, value_mu)  # same clipping as denormalize_predictions
        if actuals is not None:
            observed = np.asarray(actuals, dtype=float)[:, k]
            value_mu = np.where(np.isnan(observed), value_mu, observed)
        fed_back[:, k] = value_mu
        buf = np.concatenate([buf[:, 1:], value_mu[:, None]], axis=1)
    return preds, fed_back


DEMAND_STATUS_OK = "ok"
DEMAND_STATUS_STALE = "stale"
DEMAND_STATUS_MISSING = "missing"


@dataclass(frozen=True)
class DemandHistoryQuality:
    """Freshness of the demand history that seeds a recursive forecast."""
    status: str  # DEMAND_STATUS_OK | DEMAND_STATUS_STALE | DEMAND_STATUS_MISSING
    last_date: Optional[pd.Timestamp] = None
    age_days: Optional[int] = None  # forecast start minus last observation


def assess_demand_history(historical_demand, first_date, stale_days=STALE_DEMAND_DAYS):
    """Classify history before `first_date` as missing (none), stale (> stale_days old) or ok."""
    first_date = pd.to_datetime(first_date)
    if historical_demand is None or len(historical_demand) == 0:
        return DemandHistoryQuality(DEMAND_STATUS_MISSING)
    prior = historical_demand[historical_demand['date'] < first_date]
    if len(prior) == 0:
        return DemandHistoryQuality(DEMAND_STATUS_MISSING)
    last_date = pd.to_datetime(prior['date'].max())
    age_days = int((first_date - last_date).days)
    status = DEMAND_STATUS_STALE if age_days > stale_days else DEMAND_STATUS_OK
    return DemandHistoryQuality(status, last_date, age_days)


def demand_history_warning(state_name, quality, stale_days=STALE_DEMAND_DAYS):
    """Human-readable data-quality notice for a quality result, or None when ok."""
    if quality.status == DEMAND_STATUS_MISSING:
        return f"{state_name} has no demand history, so no reliable forecast can be produced."
    if quality.status == DEMAND_STATUS_STALE:
        return (f"{state_name}: the latest demand observation is {quality.last_date.date()} "
                f"({quality.age_days} days before the forecast start; limit {stale_days}). "
                "The forecast is seeded from stale data and is less reliable.")
    return None


def _seed_demand_buffer(historical_demand, first_date, state_name):
    """Last DEMAND_BUFFER_LEN observed values strictly before first_date (NaN-left-padded)."""
    values = np.array([], dtype=float)
    if historical_demand is not None and len(historical_demand) > 0:
        hist = historical_demand.sort_values('date')
        prior = hist[hist['date'] < first_date]
        values = prior['actual_demand_MU'].to_numpy(dtype=float)[-DEMAND_BUFFER_LEN:]
        if len(prior) > 0:
            gap_days = (first_date - prior['date'].iloc[-1]).days - 1
            if gap_days > 0:
                logger.warning(
                    "Demand history for %s ends %d day(s) before the forecast start; "
                    "the gap is not back-filled so early horizons are staler than nominal.",
                    state_name, gap_days)
    if len(values) == 0:
        logger.warning("No demand history for %s; seeding recursion with the fallback baseline.", state_name)
        values = np.full(DEMAND_BUFFER_LEN, get_fallback_baseline(state_name))
    padded = np.full(DEMAND_BUFFER_LEN, np.nan)
    padded[-len(values):] = values
    return padded[None, :]


def predict_recursive(model, weather_df, state_name, metadata=None, historical_demand=None,
                      teacher_forcing=False, is_log_scale=False):
    """
    Multi-day forecast that feeds each prediction back into the demand features.

    Day k's demand_roll_* / daily_change / daily_pct_change are computed from the
    observed history plus predictions for days 1..k-1 (training definitions).

    Parameters:
        model: loaded model
        weather_df (pd.DataFrame): weather for the forecast days (ascending dates)
        state_name (str)
        metadata (dict, optional): model_metadata.json contents
        historical_demand (pd.DataFrame, optional): date + actual_demand_MU; only rows
            before the first weather date seed the buffer
        teacher_forcing (bool): when True, observed demand from historical_demand for
            day k (if known) replaces the prediction in the buffer. Use for
            backtests over historical weather, where one-step-ahead behaviour is the
            honest question; never for a live forecast.
        is_log_scale (bool): legacy log1p-target models

    Returns:
        tuple: (predictions, df_features)
            - predictions: np.ndarray of raw model outputs (same scale as model.predict)
            - df_features: engineered features, with the demand columns the model
              actually saw on each day

    Models that do not use the demand features (legacy XGBoost) or whose inputs are
    not all engineered columns fall back to a single batch predict (nothing to feed back).
    """
    if historical_demand is None:
        historical_demand = load_historical_demand(state_name)

    df_features = engineer_features(weather_df, state_name, metadata=metadata,
                                    historical_demand=historical_demand)

    expected = list(model.feature_names_in_) if hasattr(model, 'feature_names_in_') else None
    uses_demand = bool(expected) and any(f in DEMAND_FEATURE_NAMES for f in expected)
    recursive_ok = uses_demand and all(f in df_features.columns for f in expected)

    if not recursive_ok:
        if uses_demand:
            logger.warning("Model inputs are not all engineered columns; falling back to non-recursive prediction.")
        X, df_features = prepare_features_for_prediction(
            weather_df, state_name, model=model, metadata=metadata, historical_demand=historical_demand)
        return np.asarray(model.predict(X), dtype=float), df_features

    first_date = pd.to_datetime(df_features['date'].iloc[0])
    buffer = _seed_demand_buffer(historical_demand, first_date, state_name)

    static_names = [f for f in expected if f not in DEMAND_FEATURE_NAMES]
    static = df_features[static_names].to_numpy(dtype=float)[None, :, :]

    actuals = None
    if teacher_forcing and historical_demand is not None and len(historical_demand) > 0:
        observed = historical_demand.drop_duplicates('date').set_index('date')['actual_demand_MU']
        actuals = observed.reindex(pd.to_datetime(df_features['date'])).to_numpy(dtype=float)[None, :]

    preds, fed_back = recursive_forecast_batch(model, expected, static, static_names, buffer,
                                               actuals=actuals, is_log_scale=is_log_scale)

    # Record the demand features the model actually saw: day k used buffer + fed_back[:k].
    seen = {name: [] for name in DEMAND_FEATURE_NAMES}
    replay = buffer.copy()
    for k in range(len(df_features)):
        demand = demand_features_from_buffer(replay)
        for name in DEMAND_FEATURE_NAMES:
            seen[name].append(float(demand[name][0]))
        replay = np.concatenate([replay[:, 1:], fed_back[:, k:k + 1]], axis=1)
    for name in DEMAND_FEATURE_NAMES:
        df_features[name] = seen[name]
    return preds[0], df_features
