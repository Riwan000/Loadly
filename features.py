# ============================================================================
# FEATURE ENGINEERING FUNCTIONS
# ============================================================================
import pandas as pd
import numpy as np
import holidays
from data_loading import (
    get_state_historical_averages,
    load_historical_demand,
    load_historical_data,
    get_fallback_baseline
)


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
    state_to_region = {
        'Arunachal Pradesh': 'ner', 'Assam': 'ner', 'Bihar': 'er', 'Chhattisgarh': 'er',
        'Goa': 'wr', 'Gujarat': 'wr', 'Haryana': 'nr', 'Himachal Pradesh': 'nr',
        'Jammu and Kashmir': 'nr', 'Jharkhand': 'er', 'Karnataka': 'sr', 'Kerala': 'sr',
        'Madhya Pradesh': 'wr', 'Maharashtra': 'wr', 'Manipur': 'ner', 'Meghalaya': 'ner',
        'Mizoram': 'ner', 'Nagaland': 'ner', 'Odisha': 'er', 'Punjab': 'nr',
        'Rajasthan': 'nr', 'Sikkim': 'ner', 'Tamil Nadu': 'sr', 'Telangana': 'sr',
        'Tripura': 'ner', 'Uttar Pradesh': 'nr', 'Uttarakhand': 'nr', 'West Bengal': 'er',
        'NCT of Delhi': 'nr', 'Puducherry': 'sr', 'Chandigarh': 'nr',
        'Dadra and Nagar Haveli': 'wr', 'Daman and Diu': 'wr', 'Lakshadweep': 'sr',
        'Andaman and Nicobar': 'sr'
    }

    normalized_state = state_name
    if state_name == 'Jammu and Kashmir':
        normalized_state = 'J&K'

    df['states'] = normalized_state
    df['region'] = state_to_region.get(state_name, 'nr')

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

        state_to_region = {
            'Arunachal Pradesh': 'ner', 'Assam': 'ner', 'Bihar': 'er', 'Chhattisgarh': 'er',
            'Goa': 'wr', 'Gujarat': 'wr', 'Haryana': 'nr', 'Himachal Pradesh': 'nr',
            'Jammu and Kashmir': 'nr', 'Jharkhand': 'er', 'Karnataka': 'sr', 'Kerala': 'sr',
            'Madhya Pradesh': 'wr', 'Maharashtra': 'wr', 'Manipur': 'ner', 'Meghalaya': 'ner',
            'Mizoram': 'ner', 'Nagaland': 'ner', 'Odisha': 'er', 'Punjab': 'nr',
            'Rajasthan': 'nr', 'Sikkim': 'ner', 'Tamil Nadu': 'sr', 'Telangana': 'sr',
            'Tripura': 'ner', 'Uttar Pradesh': 'nr', 'Uttarakhand': 'nr', 'West Bengal': 'er',
            'NCT of Delhi': 'nr', 'Puducherry': 'sr', 'Chandigarh': 'nr',
            'Dadra and Nagar Haveli': 'wr', 'Daman and Diu': 'wr', 'Lakshadweep': 'sr',
            'Andaman and Nicobar': 'sr'
        }
        current_region = state_to_region.get(state_name, 'nr')
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
