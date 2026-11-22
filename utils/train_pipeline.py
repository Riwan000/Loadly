"""
End-to-End Retraining Pipeline:
1. Feature Engineering: calculates CDD, lags, rolling statistics, heatwave flags, interactions.
2. Model Training: trains Linear Regression and XGBoost with a rolling temporal split (test = last 12 months of data, train = everything before).
3. Evaluates one-step metrics (RMSE, MAE, R2, state-wise RMSE) and a recursive 1..30-day backtest (utils/backtest.py); the best model is chosen on the multi-horizon backtest.
4. Promotion gate (utils/backtest.evaluate_promotion_gate): the selected model must beat persistence at
   h=1 and be within PROMOTION_TOLERANCE of the better naive baseline at h=7/14/30. If no candidate passes,
   artifacts are NOT replaced and the numbers are reported.
5. The chosen model is refit on ALL rows (train + test window) and exported to artifacts/best_model.pkl and
   artifacts/model_metadata.json. Error estimates in the metadata come from the held-out backtest of the
   train-only fit.
"""

import json
import pickle
import warnings
from pathlib import Path
import holidays
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from features import (
    DEMAND_LAGS,
    DEMAND_LAG_FEATURE_NAMES,
    LEGACY_FEATURE_NAMES,
    PRODUCTION_FEATURE_NAMES,
)
from utils.backtest import (
    PROMOTION_TOLERANCE,
    TEST_WINDOW_MONTHS,
    evaluate_promotion_gate,
    mean_horizon_rmse,
    rolling_split_cutoff,
    run_horizon_backtest,
)

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

CDD_BASE_KELVIN = 297.15
EXTREME_HEAT_PERCENTILE = 95
# XGBoost replaces the best linear model only if its mean multi-horizon backtest RMSE is lower by this fraction.
LINEAR_PREFERENCE_MARGIN = 0.02
# Ridge alpha grid and forward-chaining folds, both searched inside the TRAIN period only.
RIDGE_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)
RIDGE_CV_FOLDS = 3
REFERENCE_MODEL = "Reference_LR_15"  # current 15-feature LR; reported, never deployable
LINEAR_CANDIDATES = ("LR_lags", "Ridge_lags")
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

    # 5. Demand-derived features (lags, rolling stats, day-over-day change).
    # All computed from the *shifted* (t-1 and earlier) demand series so that
    # no feature at row t can see y_t - see add_demand_lag_features() docstring.
    df = add_demand_lag_features(df, group_cols=["states", "source"], target_col="energy met (mu)")
    df["temperature_lag_1"] = df["temp_mean_lag1"]

    # 6. Drop per-group warm-up rows where demand-derived features are
    # unavailable (no leaked fill). daily_change/daily_pct_change need two
    # prior known days (row indices 0 and 1 within each group); the rolling
    # means/stds use min_periods=1 on the shifted series, so only row 0 is NaN.
    warmup_cols = ["daily_change", "daily_pct_change"]
    df = df.dropna(subset=warmup_cols)

    # 7. Clean and save
    df = df.dropna(subset=["energy met (mu)"])
    df.to_csv(output_path, index=False)
    print(f"✓ Saved {len(df):,} rows -> {output_path}")
    return df


def add_demand_lag_features(df, group_cols, target_col, date_col="date"):
    """Add demand-derived, leakage-safe lag/rolling/change features.

    Every feature here is computed strictly from information available
    *before* row t (i.e. y_{t-1} and earlier) within each (group_cols) panel:

    - demand_roll_mean_{7,14,30} / demand_roll_std_{7,30}: rolling stats of
      the demand series shifted by 1, so the window ending at t only uses
      y_{t-1}, y_{t-2}, ....
    - daily_change: y_{t-1} - y_{t-2} (the day-over-day change between the
      two most recently *known* days as of forecast time for row t) — this
      matches how features.py computes it at inference
      (curr_demands[-1] - curr_demands[-2]) using only observed history.
    - daily_pct_change: (y_{t-1} - y_{t-2}) / y_{t-2}, same alignment.
    - demand_lag_{1,7,14}: y_{t-k}, k >= 1. When `date_col` is present the lag is a
      date lookup (the value observed exactly k days before, NaN if that day is
      missing), so a hole in the series never silently pairs t with an older day. Without
      a date column it is a positional shift. This is what
      features.demand_features_from_buffer feeds back during recursion.

    The first row of each group has no shifted history and is left as NaN
    (no bfill/ffill), so the caller can drop true warm-up rows instead of
    silently filling them with future-derived values.
    """
    df = df.copy()
    df["_shifted_target"] = df.groupby(group_cols)[target_col].shift(1)
    shifted_grouped = df.groupby(group_cols)["_shifted_target"]

    df["demand_roll_mean_7"] = shifted_grouped.transform(lambda s: s.rolling(7, min_periods=1).mean())
    df["demand_roll_mean_14"] = shifted_grouped.transform(lambda s: s.rolling(14, min_periods=1).mean())
    df["demand_roll_mean_30"] = shifted_grouped.transform(lambda s: s.rolling(30, min_periods=1).mean())
    df["demand_roll_std_7"] = shifted_grouped.transform(lambda s: s.rolling(7, min_periods=1).std())
    df["demand_roll_std_30"] = shifted_grouped.transform(lambda s: s.rolling(30, min_periods=1).std())

    # Fill the single-observation std NaN (std of one value is undefined) with
    # 0, but keep row-0-of-group NaNs (no shifted history at all) as NaN so
    # the caller can drop them as true warm-up rows.
    has_history = df["_shifted_target"].notna()
    df.loc[has_history, "demand_roll_std_7"] = df.loc[has_history, "demand_roll_std_7"].fillna(0)
    df.loc[has_history, "demand_roll_std_30"] = df.loc[has_history, "demand_roll_std_30"].fillna(0)

    lag1 = df.groupby(group_cols)[target_col].shift(1)
    lag2 = df.groupby(group_cols)[target_col].shift(2)
    df["daily_change"] = lag1 - lag2

    # NaN caused by warm-up (lag1/lag2 unavailable) stays NaN and is dropped
    # by the caller. NaN/inf caused by a genuine lag2 == 0 (0/0 or x/0) is a
    # defined "no baseline to compare against" case and is mapped to 0,
    # matching the inference-time convention in features.py.
    has_history = lag1.notna() & lag2.notna()
    pct_change = (lag1 - lag2) / lag2
    undefined_division = has_history & (pct_change.isna() | np.isinf(pct_change))
    pct_change = pct_change.where(~undefined_division, 0.0)
    df["daily_pct_change"] = pct_change

    if date_col in df.columns:
        by_date = df.set_index(group_cols + [date_col])[target_col]
        if not by_date.index.is_unique:
            raise ValueError(f"Duplicate ({', '.join(group_cols)}, {date_col}) rows; demand lags are ambiguous.")
    for k in DEMAND_LAGS:
        if date_col in df.columns:
            keys = df[group_cols].assign(**{date_col: df[date_col] - pd.Timedelta(days=k)})
            lagged = by_date.reindex(pd.MultiIndex.from_frame(keys)).to_numpy()
        else:
            lagged = df.groupby(group_cols)[target_col].shift(k)
        df[f"demand_lag_{k}"] = lagged

    df = df.drop(columns=["_shifted_target"])
    return df




TARGET = "energy met (mu)"


def make_estimator(kind, alpha=None):
    """Fresh unfitted estimator for a candidate kind: 'lr' | 'ridge' | 'xgb'."""
    if kind == "lr":
        return LinearRegression()
    if kind == "ridge":
        return Ridge(alpha=alpha)
    if kind == "xgb":
        return XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=5,
                            subsample=0.8, colsample_bytree=0.8, random_state=42)
    raise ValueError(f"Unknown estimator kind: {kind}")


def _rmse(actual, pred):
    return float(np.sqrt(mean_squared_error(actual, pred)))


def select_ridge_alpha(X, y, dates, alphas=RIDGE_ALPHAS, folds=RIDGE_CV_FOLDS):
    """
    Pick Ridge alpha with forward-chaining validation on the given (TRAIN-only) rows.

    The sorted unique dates are cut into folds+1 chunks; fold i trains on chunks[:i] and
    validates on chunk i (one-step RMSE, pooled over states). Returns (best_alpha, {alpha: mean RMSE}).
    """
    dates = np.asarray(dates)
    unique_dates = np.sort(pd.unique(dates))
    chunks = np.array_split(unique_dates, folds + 1)
    scores = {}
    for alpha in alphas:
        errs = []
        for i in range(1, folds + 1):
            fit_rows = np.isin(dates, np.concatenate(chunks[:i]))
            val_rows = np.isin(dates, chunks[i])
            model = Ridge(alpha=alpha).fit(X[fit_rows], y[fit_rows])
            errs.append(_rmse(y[val_rows], model.predict(X[val_rows])))
        scores[float(alpha)] = float(np.mean(errs))
    return min(scores, key=scores.get), scores


def select_candidate(results, margin=LINEAR_PREFERENCE_MARGIN):
    """
    Choose the deployable candidate, or None.

    results: {name: {"score": mean 1..30 backtest RMSE, "gate_passed": bool, "kind": str}}
    Only gate-passing candidates are eligible. Among them the best (lowest score) linear
    model wins unless a tree model beats it by more than `margin` (trees cannot extrapolate
    beyond the training demand range, and a ~1% gap is inside backtest noise).
    """
    eligible = {n: r for n, r in results.items() if r["gate_passed"]}
    if not eligible:
        return None
    linear = {n: r for n, r in eligible.items() if r["kind"] != "xgb"}
    trees = {n: r for n, r in eligible.items() if r["kind"] == "xgb"}
    best_linear = min(linear, key=lambda n: linear[n]["score"]) if linear else None
    best_tree = min(trees, key=lambda n: trees[n]["score"]) if trees else None
    if best_linear is None:
        return best_tree
    if best_tree is not None and trees[best_tree]["score"] < linear[best_linear]["score"] * (1 - margin):
        return best_tree
    return best_linear


def fit_candidate(name, kind, feature_names, df_clean, fit_rows, train_cutoff, alpha_search=True):
    """
    Fit one candidate on the train rows and evaluate it on the held-out window.

    df_clean: all Total rows with cleaned features; fit_rows: boolean mask of rows with valid
    demand lags. Returns a dict with the fitted (train-only) model, one-step metrics,
    state_rmse, the recursive 1..30-day backtest and the promotion-gate verdict.
    """
    is_train = (df_clean["date"] < train_cutoff) & fit_rows
    is_test = (df_clean["date"] >= train_cutoff) & fit_rows
    X_train = df_clean.loc[is_train, feature_names]
    y_train = df_clean.loc[is_train, TARGET]
    X_test = df_clean.loc[is_test, feature_names]
    y_test = df_clean.loc[is_test, TARGET]

    alpha, alpha_scores = None, None
    if kind == "ridge" and alpha_search:
        alpha, alpha_scores = select_ridge_alpha(
            X_train.to_numpy(float), y_train.to_numpy(float), df_clean.loc[is_train, "date"].to_numpy())
    model = make_estimator(kind, alpha).fit(X_train, y_train)

    pred = model.predict(X_test)
    eval_df = pd.DataFrame({"states": df_clean.loc[is_test, "states"].to_numpy(),
                            "actual": y_test.to_numpy(), "pred": pred})
    state_rmse = {s: _rmse(g["actual"], g["pred"]) for s, g in eval_df.groupby("states") if len(g)}
    backtest = run_horizon_backtest(model, df_clean, feature_names, train_cutoff)
    gate_passed, gate_reasons = evaluate_promotion_gate(backtest)
    return {
        "name": name, "kind": kind, "feature_names": list(feature_names), "model": model,
        "alpha": alpha, "alpha_cv_rmse": alpha_scores,
        "one_step": {"rmse": _rmse(y_test, pred), "mae": float(mean_absolute_error(y_test, pred)),
                     "r2": float(r2_score(y_test, pred)), "n": int(len(y_test))},
        "state_rmse": state_rmse, "backtest": backtest, "score": mean_horizon_rmse(backtest),
        "gate_passed": gate_passed, "gate_reasons": gate_reasons,
    }


def refit_on_all_rows(result, df_clean, fit_rows):
    """Clone the evaluated candidate and fit it on ALL valid rows (train + test window)."""
    final = clone(result["model"])
    final.fit(df_clean.loc[fit_rows, result["feature_names"]], df_clean.loc[fit_rows, TARGET])
    return final


def _candidate_specs():
    """name -> (kind, feature list). The reference is the pre-lag production model."""
    return {
        REFERENCE_MODEL: ("lr", LEGACY_FEATURE_NAMES),
        "LR_lags": ("lr", PRODUCTION_FEATURE_NAMES),
        "Ridge_lags": ("ridge", PRODUCTION_FEATURE_NAMES),
        "XGBoost_lags": ("xgb", PRODUCTION_FEATURE_NAMES),
    }


def _horizon_row(backtest, horizons=(1, 7, 14, 30)):
    return {h: backtest["horizon_rmse"].get(str(h)) for h in horizons}


def _naive_baseline_one_step(df_total, eval_rows):
    """One-step persistence (y_{t-1}) and seasonal-naive (y_{t-7}) RMSE/MAE/R2 on the evaluation rows."""
    out = {}
    for name, col in [("persistence", "persistence_pred"), ("seasonal_naive", "seasonal_naive_pred")]:
        eval_df = df_total.loc[eval_rows, [TARGET, col]].dropna()
        out[name] = {"rmse": _rmse(eval_df[TARGET], eval_df[col]),
                     "mae": float(mean_absolute_error(eval_df[TARGET], eval_df[col])),
                     "r2": float(r2_score(eval_df[TARGET], eval_df[col])), "n": int(len(eval_df))}
    return out


def prepare_model_frame(dataset_path):
    """
    Load the Total-source rows and build the cleaned model frame.

    Returns (df_total, df_clean, fit_rows, train_cutoff). fit_rows marks rows whose demand lags
    are defined; those are the only rows trained on or scored one-step (series warm-up and holes
    in the series are excluded, never zero-filled). The backtest uses every row.
    """
    df = pd.read_csv(dataset_path)
    df["date"] = pd.to_datetime(df["date"])
    df_total = df[df["source"] == "Total"].copy()
    df_total = df_total.sort_values(["states", "date"]).reset_index(drop=True)
    print(f"Total dataset: {len(df_total):,} rows | {df_total['date'].min().date()} -> {df_total['date'].max().date()} | {df_total['states'].nunique()} states")

    if not set(DEMAND_LAG_FEATURE_NAMES).issubset(df_total.columns):
        # Dataset predates the lag features: derive them with the same (date-aware) definition.
        df_total = add_demand_lag_features(df_total, group_cols=["states"], target_col=TARGET)

    train_cutoff = rolling_split_cutoff(df_total["date"])
    # Naive baselines on actual lagged demand (one-step), same rows as the models.
    df_total["persistence_pred"] = df_total.groupby("states")[TARGET].shift(1)
    df_total["seasonal_naive_pred"] = df_total.groupby("states")[TARGET].shift(7)

    all_features = sorted({f for _, feats in _candidate_specs().values() for f in feats})
    for f in all_features:
        if f not in df_total.columns:
            df_total[f] = 0.0
    df_clean = df_total.dropna(subset=[TARGET]).copy()
    lag_cols = [f for f in all_features if f in DEMAND_LAG_FEATURE_NAMES]
    fit_rows = df_clean[lag_cols].notna().all(axis=1)
    other_features = [f for f in all_features if f not in lag_cols]
    df_clean[other_features] = df_clean[other_features].replace([np.inf, -np.inf], 0).fillna(0)
    return df_total, df_clean, fit_rows, train_cutoff


def run_model_training(dataset_path=None, model_path=None, metadata_path=None, deploy=True):
    """
    Evaluate every candidate on the held-out rolling window, apply the promotion gate and, only if a
    candidate passes, refit it on ALL rows and write the model + metadata.

    Returns {"deployed", "selected", "results", "table", "train_cutoff"}. Nothing is written when
    `deploy` is False or no candidate passes the gate.
    """
    print("\n=== Step 2: Model Training & Evaluation ===")
    dataset_path = Path(dataset_path) if dataset_path else DATA_DIR / "processed" / "feature_engineered_dataset.csv"
    model_path = Path(model_path) if model_path else MODEL_OUTPUT_PATH
    metadata_path = Path(metadata_path) if metadata_path else METADATA_OUTPUT_PATH

    df_total, df_clean, fit_rows, train_cutoff = prepare_model_frame(dataset_path)
    is_train_raw = df_total["date"] < train_cutoff
    print(f"Train cutoff {train_cutoff.date()} | rows usable for fitting (valid demand lags): "
          f"{int(fit_rows.sum()):,} of {len(df_clean):,}")
    extreme_heat_threshold = df_total.loc[is_train_raw, "2m_temperature_mean"].quantile(EXTREME_HEAT_PERCENTILE / 100)

    results = {}
    for name, (kind, feats) in _candidate_specs().items():
        r = results[name] = fit_candidate(name, kind, feats, df_clean, fit_rows, train_cutoff)
        extra = f" alpha={r['alpha']}" if r["alpha"] is not None else ""
        print(f"{name:16s} mean-horizon RMSE {r['score']:.3f} | "
              + " ".join(f"h{h}={v:.2f}" for h, v in _horizon_row(r["backtest"]).items() if v is not None)
              + f" | gate {'PASS' if r['gate_passed'] else 'FAIL'}{extra}")

    deployable = {n: r for n, r in results.items() if n != REFERENCE_MODEL}
    selected = select_candidate(deployable)
    ref_bt = results[REFERENCE_MODEL]["backtest"]
    table = {n: _horizon_row(r["backtest"]) for n, r in results.items()}
    for base in ("persistence", "seasonal_naive"):
        table[base] = {h: ref_bt["baseline_horizon_rmse"][base].get(str(h)) for h in (1, 7, 14, 30)}
    summary = {"deployed": False, "selected": selected, "results": results, "table": table,
               "train_cutoff": str(train_cutoff.date())}

    if selected is None:
        print("\nNO candidate passed the promotion gate; artifacts NOT replaced.")
        for n, r in deployable.items():
            print(f"  {n}: " + " | ".join(r["gate_reasons"]))
        return summary
    print(f"\nSelected: {selected} (mean 1..30-day backtest RMSE: "
          + ", ".join(f"{n} {r['score']:.3f}" for n, r in deployable.items())
          + f"; XGBoost must win by >{LINEAR_PREFERENCE_MARGIN:.0%})")
    if not deploy:
        return summary

    chosen = results[selected]
    final_model = refit_on_all_rows(chosen, df_clean, fit_rows)
    metadata = _build_metadata(chosen, results, df_total, df_clean, fit_rows, train_cutoff, extreme_heat_threshold)
    with open(model_path, "wb") as f:
        pickle.dump(final_model, f)
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved model -> {model_path}\nSaved metadata -> {metadata_path}")
    summary["deployed"] = True
    return summary


def _build_metadata(chosen, results, df_total, df_clean, fit_rows, train_cutoff, extreme_heat_threshold):
    state_rmse = chosen["state_rmse"]
    best_bt = chosen["backtest"]
    first_date = df_clean.loc[fit_rows, "date"].min().date()
    last_date = df_clean.loc[fit_rows, "date"].max().date()
    test_rows = (df_clean["date"] >= train_cutoff) & fit_rows
    baselines = _naive_baseline_one_step(df_total.loc[df_clean.index], test_rows)
    for key, name in [("linear_regression", "LR_lags"), ("ridge", "Ridge_lags"),
                      ("xgboost", "XGBoost_lags"), ("reference_lr_15", REFERENCE_MODEL)]:
        baselines[key] = results[name]["one_step"]
    return {
        "model_name": chosen["name"],
        "trained_on": f"{first_date} to {last_date}",
        "train_cutoff_date": str(train_cutoff.date()),
        "test_window_months": TEST_WINDOW_MONTHS,
        "error_estimates_from": (f"held-out backtest, train-only fit (trained before {train_cutoff.date()}); "
                                 f"deployed model refit on all data through {last_date}"),
        "rmse": chosen["one_step"]["rmse"],
        "mae": chosen["one_step"]["mae"],
        "r2": chosen["one_step"]["r2"],
        "confidence_level": 0.90,
        "feature_names": list(chosen["feature_names"]),
        "n_features": len(chosen["feature_names"]),
        "cdd_threshold_kelvin": CDD_BASE_KELVIN,
        "extreme_heat_threshold_kelvin": float(extreme_heat_threshold),
        "state_rmse": state_rmse,
        "state_rmse_summary": {
            "total_states": len(state_rmse), "successful": len(state_rmse), "failed": 0,
            "mean_rmse": float(np.mean(list(state_rmse.values()))),
            "min_rmse": float(np.min(list(state_rmse.values()))),
            "max_rmse": float(np.max(list(state_rmse.values()))),
            "failed_states": [],
        },
        "baselines": baselines,
        "horizon_rmse": best_bt["horizon_rmse"],
        "state_horizon_rmse": best_bt["state_horizon_rmse"],
        "national_horizon_rmse": best_bt["national_horizon_rmse"],
        "baseline_horizon_rmse": best_bt["baseline_horizon_rmse"],
        "backtest_config": best_bt["backtest_config"],
        "candidate_mean_horizon_rmse": {n: r["score"] for n, r in results.items()},
        "reference_horizon_rmse": results[REFERENCE_MODEL]["backtest"]["horizon_rmse"],
        "ridge_alpha": chosen["alpha"],
        "promotion_gate": {"passed": True, "beat_persistence_at_h": 1,
                           "within_baseline_at_h": [7, 14, 30], "tolerance": PROMOTION_TOLERANCE},
        "evaluation": "rmse/mae/r2/state_rmse: one-step-ahead on the rolling test window using actual lagged demand "
                      "(train-only fit). horizon_rmse*: recursive 1..30-day backtest on the same window "
                      "(see utils/backtest.py), train-only fit. The deployed model is refit on all data.",
        "target_scale": "mu",
    }


if __name__ == "__main__":
    run_feature_engineering()
    run_model_training()
