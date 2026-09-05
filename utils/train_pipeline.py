"""
End-to-End Retraining Pipeline:
1. Feature Engineering: calculates CDD, lags, rolling statistics, heatwave flags, interactions.
2. Model Training: trains Linear Regression, Random Forest, XGBoost models with temporal split (train < 2024-01-01, test >= 2024-01-01).
3. Evaluates metrics (RMSE, MAE, R2, state-wise RMSE).
4. Exports updated artifacts/best_model.pkl and artifacts/model_metadata.json.
"""

import json
import pickle
import warnings
from pathlib import Path
import holidays
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

CDD_BASE_KELVIN = 297.15
EXTREME_HEAT_PERCENTILE = 95
TRAIN_CUTOFF_DATE = "2024-01-01"
MODEL_OUTPUT_PATH = ARTIFACTS_DIR / "best_model.pkl"
METADATA_OUTPUT_PATH = ARTIFACTS_DIR / "model_metadata.json"


def run_feature_engineering():
    print("=== Step 1: Feature Engineering ===")
    input_path = DATA_DIR / "processed" / "preprocessed_merged_features_clean.csv"
    output_path = DATA_DIR / "processed" / "feature_engineered_dataset.csv"

    df = pd.read_csv(input_path)
    df["date"] = pd.to_datetime(df["date"])
    print(f"Loaded {len(df):,} rows ({df['date'].min().date()} to {df['date'].max().date()})")

    # 1. Holidays
    years = [int(y) for y in df["date"].dt.year.unique() if 2010 <= y <= 2030]
    if not years:
        years = range(2013, 2027)
    india_holidays = holidays.India(years=years)
    df["holiday_name"] = df["date"].apply(lambda d: india_holidays.get(d.date() if hasattr(d, "date") else d))
    df["holiday_name"] = df["holiday_name"].fillna("No Holiday")
    df["major_holiday"] = (df["holiday_name"] != "No Holiday").astype(int)

    # 2. Temperature ranges & CDD (Threshold: 24C = 297.15K)
    df["CDD"] = np.maximum(0, df["2m_temperature_mean"] - CDD_BASE_KELVIN)
    df["temp_range"] = df["2m_temperature_max"] - df["2m_temperature_min"]

    # 3. Sort for panel operations
    df = df.sort_values(["states", "source", "date"]).reset_index(drop=True)

    # 4. Temperature lags & rolling averages per state
    df["temp_mean_lag1"] = df.groupby(["states", "source"])["2m_temperature_mean"].shift(1)
    df["temp_mean_lag2"] = df.groupby(["states", "source"])["2m_temperature_mean"].shift(2)
    df["temp_mean_lag3"] = df.groupby(["states", "source"])["2m_temperature_mean"].shift(3)
    df["temp_mean_lag7"] = df.groupby(["states", "source"])["2m_temperature_mean"].shift(7)

    df["temp_mean_roll3"] = df.groupby(["states", "source"])["2m_temperature_mean"].transform(lambda s: s.rolling(3, min_periods=1).mean())
    df["temp_mean_roll7"] = df.groupby(["states", "source"])["2m_temperature_mean"].transform(lambda s: s.rolling(7, min_periods=1).mean())
    df["temp_max_roll3"] = df.groupby(["states", "source"])["2m_temperature_max"].transform(lambda s: s.rolling(3, min_periods=1).mean())
    df["temp_max_roll7"] = df.groupby(["states", "source"])["2m_temperature_max"].transform(lambda s: s.rolling(7, min_periods=1).mean())
    df["temp_min_roll3"] = df.groupby(["states", "source"])["2m_temperature_min"].transform(lambda s: s.rolling(3, min_periods=1).mean())
    df["temp_min_roll7"] = df.groupby(["states", "source"])["2m_temperature_min"].transform(lambda s: s.rolling(7, min_periods=1).mean())

    # Backfill initial lag rows per group
    lag_cols = ["temp_mean_lag1", "temp_mean_lag2", "temp_mean_lag3", "temp_mean_lag7"]
    for c in lag_cols:
        df[c] = df.groupby(["states", "source"])[c].bfill()

    # 5. Demand rolling statistics for Best Model (shift 1 to prevent data leakage)
    df["demand_roll_mean_7"] = df.groupby(["states", "source"])["energy met (mu)"].transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean())
    df["demand_roll_mean_7"] = df.groupby(["states", "source"])["demand_roll_mean_7"].bfill()

    df["demand_roll_mean_14"] = df.groupby(["states", "source"])["energy met (mu)"].transform(lambda s: s.shift(1).rolling(14, min_periods=1).mean())
    df["demand_roll_mean_14"] = df.groupby(["states", "source"])["demand_roll_mean_14"].bfill()

    df["demand_roll_mean_30"] = df.groupby(["states", "source"])["energy met (mu)"].transform(lambda s: s.shift(1).rolling(30, min_periods=1).mean())
    df["demand_roll_mean_30"] = df.groupby(["states", "source"])["demand_roll_mean_30"].bfill()

    df["demand_roll_std_7"] = df.groupby(["states", "source"])["energy met (mu)"].transform(lambda s: s.shift(1).rolling(7, min_periods=1).std()).fillna(0)
    df["demand_roll_std_30"] = df.groupby(["states", "source"])["energy met (mu)"].transform(lambda s: s.shift(1).rolling(30, min_periods=1).std()).fillna(0)

    df["daily_change"] = df.groupby(["states", "source"])["energy met (mu)"].diff(1).fillna(0)
    df["daily_pct_change"] = df.groupby(["states", "source"])["energy met (mu)"].pct_change(1).replace([np.inf, -np.inf], 0).fillna(0)
    df["temperature_lag_1"] = df["temp_mean_lag1"]

    # 6. Clean and save
    df = df.dropna(subset=["energy met (mu)"])
    df.to_csv(output_path, index=False)
    print(f"✓ Saved {len(df):,} rows -> {output_path}")
    return df


def run_model_training():
    print("\n=== Step 2: Model Training & Evaluation ===")
    input_path = DATA_DIR / "processed" / "feature_engineered_dataset.csv"
    df = pd.read_csv(input_path)
    df["date"] = pd.to_datetime(df["date"])

    # Filter to Total source
    df_total = df[df["source"] == "Total"].copy().reset_index(drop=True)
    df_total = df_total.sort_values(["states", "date"]).reset_index(drop=True)

    print(f"Total dataset: {len(df_total):,} rows | {df_total['date'].min().date()} -> {df_total['date'].max().date()} | {df_total['states'].nunique()} states")

    # 1. Temporal train/test split
    train_mask = df_total["date"] < TRAIN_CUTOFF_DATE
    df_train_raw = df_total[train_mask]
    df_test_raw = df_total[~train_mask]

    print(f"Train set: {len(df_train_raw):,} rows ({df_train_raw['date'].min().date()} -> {df_train_raw['date'].max().date()})")
    print(f"Test set:  {len(df_test_raw):,} rows ({df_test_raw['date'].min().date()} -> {df_test_raw['date'].max().date()})")

    # Extreme heat threshold on train set
    extreme_heat_threshold = df_train_raw["2m_temperature_mean"].quantile(EXTREME_HEAT_PERCENTILE / 100)
    print(f"Extreme heat threshold ({EXTREME_HEAT_PERCENTILE}th pct): {extreme_heat_threshold:.2f} K ({extreme_heat_threshold - 273.15:.1f}°C)")

    df_total["extreme_heat"] = (df_total["2m_temperature_mean"] > extreme_heat_threshold).astype(int)
    df_total["extreme_heat_flag"] = df_total["extreme_heat"]
    df_total["temp_min_heat_interaction"] = df_total["2m_temperature_min"] * df_total["extreme_heat_flag"]
    df_total["temp_mean_heat_interaction"] = df_total["2m_temperature_mean"] * df_total["extreme_heat_flag"]
    df_total["temp_max_heat_interaction"] = df_total["2m_temperature_max"] * df_total["extreme_heat_flag"]
    df_total["temp_range_heat"] = df_total["temp_range"] * df_total["extreme_heat_flag"]

    # 2. Selected Feature Set (15 Features used in production model)
    FEATURE_NAMES = [
        "demand_roll_mean_7",
        "demand_roll_mean_14",
        "demand_roll_mean_30",
        "demand_roll_std_7",
        "demand_roll_std_30",
        "daily_change",
        "daily_pct_change",
        "2m_temperature_mean",
        "2m_temperature_max",
        "temperature_lag_1",
        "temp_range",
        "CDD",
        "month",
        "day_of_week",
        "is_weekend"
    ]

    # Check and clean features
    for f in FEATURE_NAMES:
        if f not in df_total.columns:
            df_total[f] = 0.0

    df_clean = df_total.dropna(subset=["energy met (mu)"]).copy()
    df_clean[FEATURE_NAMES] = df_clean[FEATURE_NAMES].replace([np.inf, -np.inf], 0).fillna(0)

    dates = df_clean["date"]
    train_idx = dates < TRAIN_CUTOFF_DATE
    test_idx = dates >= TRAIN_CUTOFF_DATE

    X_train = df_clean.loc[train_idx, FEATURE_NAMES]
    y_train = df_clean.loc[train_idx, "energy met (mu)"]
    X_test = df_clean.loc[test_idx, FEATURE_NAMES]
    y_test = df_clean.loc[test_idx, "energy met (mu)"]

    print(f"X_train: {X_train.shape} | X_test: {X_test.shape}")

    # 3. Fit Reduced Linear Regression (Production Model)
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)

    y_pred_train = lr_model.predict(X_train)
    y_pred_test = lr_model.predict(X_test)

    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_r2 = r2_score(y_test, y_pred_test)

    print("\n--- Model Evaluation Results (Test Set: 2024-2025) ---")
    print(f"Train RMSE : {train_rmse:.4f} MU")
    print(f"Test RMSE  : {test_rmse:.4f} MU")
    print(f"Test MAE   : {test_mae:.4f} MU")
    print(f"Test R²    : {test_r2:.6f}")

    # 4. State-level RMSE calculation
    df_test_eval = df_clean.loc[test_idx].copy()
    df_test_eval["pred"] = y_pred_test
    df_test_eval["actual"] = y_test

    state_rmse = {}
    for state, group in df_test_eval.groupby("states"):
        if len(group) > 0:
            s_rmse = float(np.sqrt(mean_squared_error(group["actual"], group["pred"])))
            state_rmse[state] = s_rmse

    print(f"\nCalculated state RMSE for {len(state_rmse)} states:")
    for s, r in sorted(state_rmse.items(), key=lambda x: x[1])[:5]:
        print(f"  Best : {s:25s} -> RMSE = {r:.3f} MU")
    for s, r in sorted(state_rmse.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  Worst: {s:25s} -> RMSE = {r:.3f} MU")

    # 5. Fit XGBoost Model for comparison / export
    xgb_model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    xgb_model.fit(X_train, y_train)
    xgb_pred_test = xgb_model.predict(X_test)
    xgb_test_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred_test))
    xgb_test_r2 = r2_score(y_test, xgb_pred_test)
    print(f"\nXGBoost Test RMSE: {xgb_test_rmse:.4f} MU | R²: {xgb_test_r2:.6f}")

    # Choose best model
    best_model = lr_model if test_rmse <= xgb_test_rmse else xgb_model
    model_name = "Reduced_LR" if best_model == lr_model else "XGBoost"

    # 6. Save model and metadata
    with open(MODEL_OUTPUT_PATH, "wb") as f:
        pickle.dump(best_model, f)
    print(f"\n✓ Saved model -> {MODEL_OUTPUT_PATH}")

    metadata = {
        "model_name": model_name,
        "trained_on": f"2013-{pd.Timestamp(TRAIN_CUTOFF_DATE).year - 1}",
        "train_cutoff_date": TRAIN_CUTOFF_DATE,
        "rmse": float(test_rmse),
        "mae": float(test_mae),
        "r2": float(test_r2),
        "confidence_level": 0.90,
        "feature_names": FEATURE_NAMES,
        "n_features": len(FEATURE_NAMES),
        "cdd_threshold_kelvin": CDD_BASE_KELVIN,
        "extreme_heat_threshold_kelvin": float(extreme_heat_threshold),
        "state_rmse": state_rmse,
        "state_rmse_summary": {
            "total_states": len(state_rmse),
            "successful": len(state_rmse),
            "failed": 0,
            "mean_rmse": float(np.mean(list(state_rmse.values()))),
            "min_rmse": float(np.min(list(state_rmse.values()))),
            "max_rmse": float(np.max(list(state_rmse.values()))),
            "failed_states": []
        }
    }

    with open(METADATA_OUTPUT_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Saved metadata -> {METADATA_OUTPUT_PATH}")


if __name__ == "__main__":
    run_feature_engineering()
    run_model_training()
