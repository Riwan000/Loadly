"""
Shared training-time helpers: rolling train/test split and the multi-horizon
recursive backtest (used by utils/train_pipeline.py and utils/model_retrainer.py).

The production model is one-step-ahead, so a 1-day test RMSE says nothing about the
7/14/30-day forecasts the app serves. The backtest forecasts 1..MAX_HORIZON days
ahead from many origins using the same recursion as inference
(features.recursive_forecast_batch), actual weather, and demand history only up to
the origin, and scores it against two naive baselines.
"""

import numpy as np
import pandas as pd

from features import DEMAND_BUFFER_LEN, DEMAND_FEATURE_NAMES, recursive_forecast_batch

# Rolling split: test = the last TEST_WINDOW_MONTHS of available data, train = before it.
TEST_WINDOW_MONTHS = 12
MAX_HORIZON = 30
# Origins every 5 days (not 7): a stride of 7 would pin every horizon to a single
# weekday of the target and alias the weekly cycle into the per-horizon RMSE.
ORIGIN_STRIDE_DAYS = 5
TARGET_COL = "energy met (mu)"
STATE_COL = "states"


def rolling_split_cutoff(dates, months=TEST_WINDOW_MONTHS):
    """First test date: the last `months` months of `dates` are test, everything before is train."""
    last = pd.to_datetime(pd.Series(dates)).max()
    return (last - pd.DateOffset(months=months)).normalize() + pd.Timedelta(days=1)


def _rmse_by_horizon(err_sq, horizon):
    """err_sq: (..., H) squared errors with NaN where unavailable -> {"1": rmse, ...}."""
    flat = err_sq.reshape(-1, horizon)
    out = {}
    for h in range(horizon):
        col = flat[:, h]
        col = col[~np.isnan(col)]
        if len(col):
            out[str(h + 1)] = float(np.sqrt(col.mean()))
    return out


def run_horizon_backtest(model, df, feature_names, test_start, max_horizon=MAX_HORIZON,
                         origin_stride=ORIGIN_STRIDE_DAYS, is_log_scale=False):
    """
    Recursive multi-horizon backtest on the test period.

    Parameters:
        model: fitted model (predict(DataFrame) over feature_names)
        df (pd.DataFrame): Total-source rows with date, STATE_COL, TARGET_COL and all
            feature_names columns (NaN/inf already cleaned)
        feature_names (list[str])
        test_start: first date a forecast target may fall on (rolling-split cutoff)
        origin_stride: days between forecast origins (common calendar dates for all states)

    An origin date o means y(o) is the last known value; targets are o+1..o+H. Static
    (weather/calendar) features are the actual ones for the target day; demand features
    come only from history <= o plus fed-back predictions.

    Returns dict with horizon_rmse, state_horizon_rmse, national_horizon_rmse,
    baseline_horizon_rmse {persistence, seasonal_naive} and backtest_config.
    Baselines: persistence = y(o); seasonal naive = latest observed value on the same
    weekday as the target (y(o+h-7*ceil(h/7))).
    """
    static_names = [f for f in feature_names if f not in DEMAND_FEATURE_NAMES]
    df = df.sort_values([STATE_COL, "date"])
    test_start = pd.Timestamp(test_start)
    all_dates = pd.DatetimeIndex(sorted(df["date"].unique()))
    last_date = all_dates.max()
    first_origin = test_start - pd.Timedelta(days=1)
    origin_dates = pd.date_range(first_origin, last_date - pd.Timedelta(days=1), freq=f"{origin_stride}D")

    run_state, run_origin, run_buf, run_static, run_actual, run_valid = [], [], [], [], [], []
    run_persist, run_snaive = [], []
    horizons = np.arange(1, max_horizon + 1)
    snaive_back = 7 * np.ceil(horizons / 7).astype(int)

    for state, g in df.groupby(STATE_COL, sort=True):
        dates = g["date"].to_numpy(dtype="datetime64[D]")
        y = g[TARGET_COL].to_numpy(dtype=float)
        static_all = g[static_names].to_numpy(dtype=float)
        index_of = {d: i for i, d in enumerate(dates)}
        n = len(g)
        for o_date in origin_dates:
            o = index_of.get(np.datetime64(o_date.date(), "D"))
            if o is None or o + 1 >= n:
                continue
            tgt = o + horizons
            in_range = tgt < n
            tgt_c = np.minimum(tgt, n - 1)
            contiguous = in_range & (dates[tgt_c] == dates[o] + horizons.astype("timedelta64[D]"))
            if not contiguous[0]:
                continue
            seed = y[max(0, o - DEMAND_BUFFER_LEN + 1): o + 1]
            buf = np.full(DEMAND_BUFFER_LEN, np.nan)
            buf[-len(seed):] = seed

            sn_idx = tgt - snaive_back
            sn_ok = contiguous & (sn_idx >= 0)
            sn_idx_c = np.clip(sn_idx, 0, n - 1)
            sn_ok &= dates[sn_idx_c] == dates[tgt_c] - snaive_back.astype("timedelta64[D]")

            run_state.append(state)
            run_origin.append(o_date)
            run_buf.append(buf)
            run_static.append(static_all[tgt_c])
            run_actual.append(np.where(contiguous, y[tgt_c], np.nan))
            run_valid.append(contiguous)
            run_persist.append(np.where(contiguous, y[o], np.nan))
            run_snaive.append(np.where(sn_ok, y[sn_idx_c], np.nan))

    if not run_state:
        raise ValueError("Backtest produced no forecast origins; test window too short or data missing.")

    n_runs = len(run_state)
    actual = np.array(run_actual)
    valid = np.array(run_valid)
    preds, _ = recursive_forecast_batch(
        model, feature_names, np.array(run_static), static_names, np.array(run_buf),
        is_log_scale=is_log_scale)
    preds = np.maximum(0.0, np.expm1(preds) if is_log_scale else preds)
    preds = np.where(valid, preds, np.nan)

    err = {
        "model": preds - actual,
        "persistence": np.array(run_persist) - actual,
        "seasonal_naive": np.array(run_snaive) - actual,
    }
    err_sq = {k: v ** 2 for k, v in err.items()}

    states_arr = np.array(run_state)
    state_horizon = {}
    for state in sorted(set(run_state)):
        rows = states_arr == state
        state_horizon[state] = _rmse_by_horizon(err_sq["model"][rows], max_horizon)

    # National aggregate: sum over states for the same origin date; only origins where
    # every state is available at that horizon count (errors are correlated across states).
    origin_arr = np.array(run_origin, dtype="datetime64[D]")
    n_states = len(set(run_state))
    nat_sq = []
    for o_date in np.unique(origin_arr):
        rows = origin_arr == o_date
        if rows.sum() != n_states:
            continue
        p, a = preds[rows], actual[rows]
        complete = ~(np.isnan(p).any(axis=0) | np.isnan(a).any(axis=0))
        nat = (p.sum(axis=0) - a.sum(axis=0)) ** 2
        nat_sq.append(np.where(complete, nat, np.nan))

    return {
        "horizon_rmse": _rmse_by_horizon(err_sq["model"], max_horizon),
        "state_horizon_rmse": state_horizon,
        "national_horizon_rmse": _rmse_by_horizon(np.array(nat_sq), max_horizon) if nat_sq else {},
        "baseline_horizon_rmse": {
            "persistence": _rmse_by_horizon(err_sq["persistence"], max_horizon),
            "seasonal_naive": _rmse_by_horizon(err_sq["seasonal_naive"], max_horizon),
        },
        "backtest_config": {
            "test_start": str(test_start.date()),
            "max_horizon": max_horizon,
            "origin_stride_days": origin_stride,
            "n_origin_runs": int(n_runs),
            "n_states": int(n_states),
        },
    }


def mean_horizon_rmse(result):
    """Single selection score: mean of the per-horizon pooled RMSE over horizons 1..H."""
    values = list(result["horizon_rmse"].values())
    return float(np.mean(values)) if values else float("inf")


# ---------------------------------------------------------------------------
# Promotion gate (shared by utils/train_pipeline.py and utils/model_retrainer.py)
# ---------------------------------------------------------------------------
# The candidate must strictly beat persistence at this horizon ...
GATE_BEAT_PERSISTENCE_HORIZON = 1
# ... and, at these horizons, be no worse than the BETTER of persistence / seasonal-naive
# by more than PROMOTION_TOLERANCE (a fraction: 0.02 = 2%).
GATE_BASELINE_HORIZONS = (7, 14, 30)
PROMOTION_TOLERANCE = 0.02


def evaluate_promotion_gate(result, beat_horizon=GATE_BEAT_PERSISTENCE_HORIZON,
                            baseline_horizons=GATE_BASELINE_HORIZONS,
                            tolerance=PROMOTION_TOLERANCE):
    """
    Judge a run_horizon_backtest() result against the naive baselines (pooled RMSE).

    Returns (passed, reasons): reasons lists every failed criterion (empty when passed).
    A horizon missing from the result fails its criterion, never passes silently.
    """
    model = result["horizon_rmse"]
    persistence = result["baseline_horizon_rmse"]["persistence"]
    seasonal = result["baseline_horizon_rmse"]["seasonal_naive"]
    reasons = []

    key = str(beat_horizon)
    if key not in model or key not in persistence:
        reasons.append(f"h={beat_horizon}: no backtest value to compare against persistence.")
    elif not model[key] < persistence[key]:
        reasons.append(f"h={beat_horizon}: model RMSE {model[key]:.3f} does not beat persistence {persistence[key]:.3f}.")

    for horizon in baseline_horizons:
        key = str(horizon)
        baselines = [b[key] for b in (persistence, seasonal) if key in b]
        if key not in model or not baselines:
            reasons.append(f"h={horizon}: no backtest value to compare against the baselines.")
            continue
        best_baseline = min(baselines)
        if model[key] > best_baseline * (1 + tolerance):
            reasons.append(
                f"h={horizon}: model RMSE {model[key]:.3f} is more than {tolerance:.0%} worse than "
                f"the better baseline {best_baseline:.3f}.")
    return (not reasons), reasons
