# ============================================================================
# TESTS: direct demand-lag features, shared feature list, promotion gate,
# refit-on-all-data, legacy (15-feature) model compatibility
# ============================================================================
import json
import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

import features
from features import (
    DEMAND_BUFFER_LEN,
    DEMAND_LAG_FEATURE_NAMES,
    DEMAND_LAGS,
    LEGACY_FEATURE_NAMES,
    PRODUCTION_FEATURE_NAMES,
    demand_features_from_buffer,
    predict_recursive,
    recursive_forecast_batch,
)
from utils import model_retrainer, train_pipeline
from utils.backtest import (
    GATE_BASELINE_HORIZONS,
    PROMOTION_TOLERANCE,
    evaluate_promotion_gate,
    run_horizon_backtest,
)
from utils.model_retrainer import retrain_model
from utils.train_pipeline import add_demand_lag_features, run_model_training, select_candidate
from tests.test_drift_detector import _write_tiny_dataset, temp_db  # noqa: F401

TARGET = "energy met (mu)"


def _series(n=60, seed=3):
    rng = np.random.default_rng(seed)
    return 100 + np.cumsum(rng.normal(0, 3, n))


def _buffer(known):
    buf = np.full((1, DEMAND_BUFFER_LEN), np.nan)
    seed = known[-DEMAND_BUFFER_LEN:]
    buf[0, -len(seed):] = seed
    return buf


class TestLagDefinition:
    """PURPOSE: demand_lag_k = y(t-k), identical in training and in the serving buffer.
    CATCHES: off-by-one lags (lag_1 == y_t is target leakage) and train/serve drift."""

    @pytest.mark.parametrize("t", [14, 15, 30, 59])
    def test_training_lag_equals_buffer_lag_and_is_y_t_minus_k(self, t):
        y = _series()
        train = add_demand_lag_features(pd.DataFrame({"g": "a", "y": y}), group_cols=["g"], target_col="y")
        feats = demand_features_from_buffer(_buffer(y[:t]))
        for k in DEMAND_LAGS:
            assert train.loc[t, f"demand_lag_{k}"] == pytest.approx(y[t - k])
            assert feats[f"demand_lag_{k}"][0] == pytest.approx(y[t - k])

    def test_lag_at_row_t_never_sees_y_t_or_later(self):
        n = 40
        dates = pd.date_range("2023-01-01", periods=n)
        df = pd.DataFrame({"states": "A", "source": "Total", "date": dates, TARGET: _series(n)})
        base = add_demand_lag_features(df, ["states", "source"], TARGET)
        t = 25
        perturbed = df.copy()
        perturbed.loc[t:, TARGET] += 1000.0  # y_t and the future change
        after = add_demand_lag_features(perturbed, ["states", "source"], TARGET)
        for name in DEMAND_LAG_FEATURE_NAMES:
            assert after.loc[t, name] == base.loc[t, name]  # unchanged at t
        perturbed_past = df.copy()
        perturbed_past.loc[t - 1, TARGET] += 1000.0
        assert add_demand_lag_features(perturbed_past, ["states", "source"], TARGET).loc[t, "demand_lag_1"] \
            == pytest.approx(base.loc[t, "demand_lag_1"] + 1000.0)

    def test_warmup_rows_are_nan_not_filled(self):
        train = add_demand_lag_features(pd.DataFrame({"g": "a", "y": _series(30)}), ["g"], "y")
        assert train.loc[:6, "demand_lag_7"].isna().all() and train.loc[7:, "demand_lag_7"].notna().all()
        assert train.loc[:13, "demand_lag_14"].isna().all()

    def test_date_aware_hole_only_invalidates_rows_that_point_at_it(self):
        dates = pd.date_range("2023-01-01", periods=30)
        y = np.arange(30, dtype=float) + 10
        df = pd.DataFrame({"states": "A", "date": dates, "y": y}).drop(index=12).reset_index(drop=True)
        out = add_demand_lag_features(df, ["states"], "y")
        by_date = out.set_index("date")
        hole = dates[12]
        assert np.isnan(by_date.loc[hole + pd.Timedelta(days=1), "demand_lag_1"])
        assert np.isnan(by_date.loc[hole + pd.Timedelta(days=7), "demand_lag_7"])
        # Rows between the hole and their t-7 are still exact (a positional shift would be wrong here).
        assert by_date.loc[hole + pd.Timedelta(days=3), "demand_lag_7"] == pytest.approx(
            y[12 + 3 - 7])
        assert by_date.loc[hole + pd.Timedelta(days=2), "demand_lag_1"] == pytest.approx(y[12 + 2 - 1])

    def test_short_buffer_gives_finite_fallback(self):
        buf = _buffer(np.array([10.0, 12.0, 14.0]))
        feats = demand_features_from_buffer(buf)
        assert feats["demand_lag_1"][0] == 14.0
        assert feats["demand_lag_7"][0] == pytest.approx(12.0)  # mean of what exists
        assert all(np.isfinite(feats[n][0]) for n in DEMAND_LAG_FEATURE_NAMES)


class TestRecursionFeedsLagBack:
    def test_lag_1_on_day_k_plus_1_is_the_prediction_for_day_k(self):
        class LagModel:
            feature_names_in_ = np.array(["demand_lag_1", "demand_lag_7"])

            def predict(self, X):
                return X["demand_lag_1"].to_numpy() + 1.0  # random-walk with drift

        buf = _buffer(np.arange(1.0, 21.0))  # y(t-1) = 20
        preds, fed = recursive_forecast_batch(LagModel(), ["demand_lag_1", "demand_lag_7"],
                                              np.zeros((1, 9, 1)), ["unused"], buf)
        assert preds[0].tolist() == pytest.approx([21.0 + i for i in range(9)])

        class Lag7(LagModel):
            def predict(self, X):
                return X["demand_lag_7"].to_numpy()

        p7, _ = recursive_forecast_batch(Lag7(), ["demand_lag_1", "demand_lag_7"],
                                         np.zeros((1, 8, 1)), ["unused"], buf)
        # y(t-7) = 14 on day 1, then 15..20 from history; on day 8 lag_7 is the value fed back for day 1.
        assert p7[0].tolist() == [14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 14.0]

    def test_predict_recursive_reports_lags_the_model_saw(self, sample_weather_df):
        class LagModel:
            feature_names_in_ = np.array(["demand_lag_1"])

            def predict(self, X):
                return X["demand_lag_1"].to_numpy() * 2.0

        hist = pd.DataFrame({"date": pd.date_range(end="2022-12-31", periods=30),
                             "actual_demand_MU": np.full(30, 5.0)})
        preds, df_feat = predict_recursive(LagModel(), sample_weather_df.head(3), "Kerala",
                                           historical_demand=hist)
        assert preds.tolist() == [10.0, 20.0, 40.0]
        assert df_feat["demand_lag_1"].tolist() == [5.0, 10.0, 20.0]


class TestSharedFeatureList:
    def test_retrainer_and_pipeline_use_the_single_shared_list(self):
        assert model_retrainer.FEATURE_NAMES == PRODUCTION_FEATURE_NAMES
        specs = train_pipeline._candidate_specs()
        for name in ("LR_lags", "Ridge_lags", "XGBoost_lags"):
            assert specs[name][1] == PRODUCTION_FEATURE_NAMES
        assert specs[train_pipeline.REFERENCE_MODEL][1] == LEGACY_FEATURE_NAMES
        assert len(LEGACY_FEATURE_NAMES) == 15 and set(LEGACY_FEATURE_NAMES) < set(PRODUCTION_FEATURE_NAMES)

    def test_every_demand_feature_is_computable_at_serving(self):
        # Features that are neither demand-derived nor static would make the backtest raise.
        demand = set(features.DEMAND_FEATURE_NAMES)
        for name in PRODUCTION_FEATURE_NAMES:
            assert name in demand or name in features.STATIC_FEATURE_NAMES


def _backtest(model, persistence, seasonal):
    return {
        "horizon_rmse": {str(h): v for h, v in model.items()},
        "baseline_horizon_rmse": {
            "persistence": {str(h): v for h, v in persistence.items()},
            "seasonal_naive": {str(h): v for h, v in seasonal.items()},
        },
    }


class TestPromotionGate:
    BASE = {1: 10.0, 7: 20.0, 14: 25.0, 30: 30.0}

    def test_passes_when_beating_persistence_and_within_tolerance(self):
        bt = _backtest({1: 9.0, 7: 20.0, 14: 25.0, 30: 30.0}, self.BASE, {**self.BASE, 7: 25.0})
        assert evaluate_promotion_gate(bt) == (True, [])

    def test_fails_when_not_strictly_better_than_persistence_at_h1(self):
        bt = _backtest({1: 10.0, 7: 20.0, 14: 25.0, 30: 30.0}, self.BASE, self.BASE)
        passed, reasons = evaluate_promotion_gate(bt)
        assert not passed and len(reasons) == 1 and "h=1" in reasons[0]

    @pytest.mark.parametrize("horizon", GATE_BASELINE_HORIZONS)
    def test_tolerance_boundary_is_two_percent_vs_better_baseline(self, horizon):
        ok = {**{1: 9.0}, **{h: self.BASE[h] for h in GATE_BASELINE_HORIZONS}}
        ok[horizon] = self.BASE[horizon] * (1 + PROMOTION_TOLERANCE)  # exactly at the limit
        assert evaluate_promotion_gate(_backtest(ok, self.BASE, self.BASE))[0]
        bad = {**ok, horizon: self.BASE[horizon] * (1 + PROMOTION_TOLERANCE) + 0.01}
        passed, reasons = evaluate_promotion_gate(_backtest(bad, self.BASE, self.BASE))
        assert not passed and f"h={horizon}" in reasons[0]

    def test_uses_the_better_of_the_two_baselines(self):
        seasonal_better = {**self.BASE, 7: 15.0}
        bt = _backtest({1: 9.0, 7: 20.0, 14: 25.0, 30: 30.0}, self.BASE, seasonal_better)
        passed, reasons = evaluate_promotion_gate(bt)
        assert not passed and "h=7" in reasons[0]

    def test_missing_horizon_fails_instead_of_passing(self):
        bt = _backtest({1: 9.0, 7: 20.0}, self.BASE, self.BASE)
        passed, reasons = evaluate_promotion_gate(bt)
        assert not passed and any("h=14" in r for r in reasons)


class TestSelectCandidate:
    def _res(self, **kw):
        return {n: {"score": s, "gate_passed": g, "kind": k} for n, (s, g, k) in kw.items()}

    def test_none_when_nothing_passes(self):
        assert select_candidate(self._res(LR=(10, False, "lr"), XG=(9, False, "xgb"))) is None

    def test_failing_candidate_is_never_selected_even_if_best(self):
        assert select_candidate(self._res(LR=(10, True, "lr"), XG=(5, False, "xgb"))) == "LR"

    def test_xgb_needs_to_beat_linear_by_margin(self):
        assert select_candidate(self._res(LR=(10, True, "lr"), XG=(9.9, True, "xgb"))) == "LR"
        assert select_candidate(self._res(LR=(10, True, "lr"), XG=(9.7, True, "xgb"))) == "XG"

    def test_best_linear_among_linear(self):
        assert select_candidate(self._res(LR=(10, True, "lr"), RG=(9.5, True, "ridge"))) == "RG"


# ---------------------------------------------------------------------------
# End-to-end pipeline on a small synthetic panel (never touches data/ or artifacts/)
# ---------------------------------------------------------------------------
def _write_panel(path, days=("2022-01-01", "2024-03-01")):
    rng = np.random.default_rng(11)
    frames = []
    dates = pd.date_range(*days)
    for state, base in (("Kerala", 60.0), ("Goa", 20.0)):
        dow = np.asarray(dates.dayofweek)
        y = base + 3 * np.sin(np.arange(len(dates)) / 20) + 2 * (dow >= 5) + rng.normal(0, 1.0, len(dates))
        temp = 295 + 6 * np.sin(np.arange(len(dates)) / 58) + rng.normal(0, 1, len(dates))
        df = pd.DataFrame({"date": dates, "states": state, "source": "Total", TARGET: y,
                           "2m_temperature_mean": temp, "2m_temperature_max": temp + 5,
                           "2m_temperature_min": temp - 5})
        df["temperature_lag_1"] = df["2m_temperature_mean"].shift(1).bfill()
        df["temp_range"] = 10.0
        df["CDD"] = np.maximum(0, temp - 297.15)
        df["month"], df["day_of_week"] = dates.month, dow
        df["is_weekend"] = (dow >= 5).astype(int)
        frames.append(add_demand_lag_features(df, ["states"], TARGET))
    pd.concat(frames).dropna(subset=["daily_change"]).to_csv(path, index=False)


@pytest.fixture
def panel(tmp_path):
    path = tmp_path / "panel.csv"
    _write_panel(path)
    return path


def test_refit_on_all_data_happens_after_evaluation(panel, tmp_path, monkeypatch):
    calls = []
    real_fit, real_refit = train_pipeline.fit_candidate, train_pipeline.refit_on_all_rows
    monkeypatch.setattr(train_pipeline, "fit_candidate",
                        lambda *a, **k: (calls.append("evaluate"), real_fit(*a, **k))[1])
    monkeypatch.setattr(train_pipeline, "refit_on_all_rows",
                        lambda *a, **k: (calls.append("refit"), real_refit(*a, **k))[1])
    monkeypatch.setattr(train_pipeline, "evaluate_promotion_gate", lambda r: (True, []))
    model_path, meta_path = tmp_path / "m.pkl", tmp_path / "m.json"

    summary = run_model_training(panel, model_path, meta_path)

    assert summary["deployed"]
    assert calls == ["evaluate"] * 4 + ["refit"]  # reference + 3 candidates, then exactly one refit
    meta = json.loads(meta_path.read_text())
    assert meta["train_cutoff_date"] == "2023-03-02"
    assert meta["trained_on"].endswith("2024-03-01")  # deployed model saw the test window
    assert "train-only fit" in meta["error_estimates_from"] and "2024-03-01" in meta["error_estimates_from"]
    assert meta["feature_names"] == PRODUCTION_FEATURE_NAMES
    deployed = pickle.loads(model_path.read_bytes())
    train_only = summary["results"][summary["selected"]]["model"]
    assert not np.allclose(getattr(deployed, "coef_", 0), getattr(train_only, "coef_", 1)) \
        or summary["selected"] == "XGBoost_lags"


def test_no_artifacts_written_when_gate_fails(panel, tmp_path, monkeypatch):
    monkeypatch.setattr(train_pipeline, "evaluate_promotion_gate", lambda r: (False, ["h=1: nope"]))
    model_path, meta_path = tmp_path / "m.pkl", tmp_path / "m.json"

    summary = run_model_training(panel, model_path, meta_path)

    assert summary["selected"] is None and not summary["deployed"]
    assert not model_path.exists() and not meta_path.exists()
    assert "Reference_LR_15" in summary["table"] and "persistence" in summary["table"]


# ---------------------------------------------------------------------------
# Retrainer: refit-on-all after evaluation, baseline gate, legacy champion
# ---------------------------------------------------------------------------
def test_retrainer_refits_on_all_rows_after_backtest(tmp_path, monkeypatch, temp_db):  # noqa: F811
    ds = tmp_path / "ds.csv"
    _write_tiny_dataset(ds)
    monkeypatch.setattr(model_retrainer, "DATASET_PATH", ds)
    model_retrainer.METADATA_PATH.write_text(json.dumps({"cdd_threshold_kelvin": 297.15}))
    seen = {}
    real = model_retrainer.run_horizon_backtest

    def spy(model, *a, **k):
        seen["model_file_at_backtest"] = model_retrainer.MODEL_PATH.read_bytes()
        seen["backtest_cutoff"] = a[2]
        return real(model, *a, **k)

    monkeypatch.setattr(model_retrainer, "run_horizon_backtest", spy)
    result = retrain_model(trigger_reason="t", force=True, db_path=temp_db)

    assert result["decision"] == "PROMOTED"
    assert model_retrainer.MODEL_PATH.read_bytes() != seen["model_file_at_backtest"]  # swap came after
    meta = json.loads(model_retrainer.METADATA_PATH.read_text())
    assert meta["model_name"] == model_retrainer.RETRAINED_MODEL_NAME
    assert meta["trained_on"].endswith("2024-03-01") and meta["train_cutoff_date"] == "2023-03-02"
    assert "train-only fit" in meta["error_estimates_from"]
    assert meta["feature_names"] == PRODUCTION_FEATURE_NAMES
    deployed = pickle.loads(model_retrainer.MODEL_PATH.read_bytes())
    assert list(deployed.feature_names_in_) == PRODUCTION_FEATURE_NAMES


def test_retrainer_rejects_candidate_failing_baseline_gate(tmp_path, monkeypatch, temp_db):  # noqa: F811
    ds = tmp_path / "ds.csv"
    _write_tiny_dataset(ds)
    monkeypatch.setattr(model_retrainer, "DATASET_PATH", ds)
    monkeypatch.setattr(model_retrainer, "RMSE_TOLERANCE", 1e9)
    monkeypatch.setattr(model_retrainer, "MIN_CHALLENGER_R2", -1e9)
    monkeypatch.setattr(model_retrainer, "MIN_CHALLENGER_COVERAGE_PCT", 0.0)
    monkeypatch.setattr(model_retrainer, "evaluate_promotion_gate", lambda r: (False, ["h=1: worse"]))
    before = model_retrainer.MODEL_PATH.read_bytes()

    result = retrain_model(trigger_reason="t", force=False, db_path=temp_db)

    assert result["decision"] == "REJECTED" and "baseline gate" in result["notes"]
    assert model_retrainer.MODEL_PATH.read_bytes() == before
    assert not model_retrainer.MODEL_PREV_PATH.exists()


# ---------------------------------------------------------------------------
# Legacy 15-feature models keep working
# ---------------------------------------------------------------------------
def _legacy_model():
    rng = np.random.default_rng(5)
    X = pd.DataFrame(rng.normal(10, 1, (200, 15)), columns=LEGACY_FEATURE_NAMES)
    return LinearRegression().fit(X, X["demand_roll_mean_7"] * 1.0 + 1.0)


def test_legacy_model_predicts_recursively_without_lag_features(sample_weather_df):
    model = _legacy_model()
    hist = pd.DataFrame({"date": pd.date_range(end="2022-12-31", periods=40),
                         "actual_demand_MU": np.linspace(50, 60, 40)})
    preds, df_feat = predict_recursive(model, sample_weather_df.head(5), "Kerala", historical_demand=hist)
    assert preds.shape == (5,) and np.isfinite(preds).all()
    assert not np.allclose(preds, preds[0])  # still recursive


def test_legacy_model_works_in_horizon_backtest():
    model = _legacy_model()
    dates = pd.date_range("2023-01-01", periods=120)
    rows = []
    for state in ("A", "B"):
        df = pd.DataFrame({f: 10.0 for f in LEGACY_FEATURE_NAMES}, index=range(len(dates)))
        df["date"], df["states"], df[TARGET] = dates, state, np.linspace(10, 20, len(dates))
        rows.append(df)
    res = run_horizon_backtest(model, pd.concat(rows), LEGACY_FEATURE_NAMES, "2023-03-15", max_horizon=5)
    assert set(res["horizon_rmse"]) == {"1", "2", "3", "4", "5"}


def test_retrainer_evaluates_a_legacy_champion_on_its_own_features(tmp_path, monkeypatch, temp_db):  # noqa: F811
    ds = tmp_path / "ds.csv"
    _write_tiny_dataset(ds)
    monkeypatch.setattr(model_retrainer, "DATASET_PATH", ds)
    model_retrainer.MODEL_PATH.write_bytes(pickle.dumps(_legacy_model()))

    result = retrain_model(trigger_reason="t", force=True, db_path=temp_db)

    assert result["champion"]["rmse"] is not None  # was evaluated, not skipped on a feature mismatch
