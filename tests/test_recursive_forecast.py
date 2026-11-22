# ============================================================================
# TESTS: recursive multi-step prediction, horizon-aware RMSE lookup, rolling split
# ============================================================================
import numpy as np
import pandas as pd
import pytest

from features import (
    DEMAND_BUFFER_LEN,
    DEMAND_FEATURE_NAMES,
    demand_features_from_buffer,
    predict_recursive,
    recursive_forecast_batch,
)
from models import get_horizon_rmse, get_horizon_rmse_series
from utils.backtest import rolling_split_cutoff, run_horizon_backtest
from utils.train_pipeline import add_demand_lag_features


class StubLinearModel:
    """pred = roll_mean_7 + 0.5 * daily_change  (no intercept, no weather)."""
    feature_names_in_ = np.array(["demand_roll_mean_7", "daily_change"])

    def predict(self, X):
        return X["demand_roll_mean_7"].to_numpy() + 0.5 * X["daily_change"].to_numpy()


def _history(values, end="2022-12-31"):
    dates = pd.date_range(end=end, periods=len(values), freq="D")
    return pd.DataFrame({"date": dates, "actual_demand_MU": np.asarray(values, dtype=float)})


class TestPredictRecursive:
    """PURPOSE: prediction k must feed the demand features of day k+1.
    CATCHES: a regression to the old behaviour where every horizon day saw identical
    demand features computed from history only."""

    def test_three_step_example_matches_hand_computation(self, sample_weather_df):
        history = _history([10, 12, 14, 16, 18, 20, 22])
        weather = sample_weather_df.head(3)

        preds, df_features = predict_recursive(
            StubLinearModel(), weather, "Maharashtra", historical_demand=history)

        # step 1: mean7=16, dc=22-20=2           -> 16 + 1       = 17
        # step 2: [12..22,17] mean7=119/7, dc=-5 -> 17 - 2.5     = 14.5
        # step 3: [14..17,14.5] mean7=121.5/7, dc=-2.5 -> 121.5/7 - 1.25
        assert preds == pytest.approx([17.0, 14.5, 121.5 / 7 - 1.25])
        # df_features reports the demand features the model actually saw each day
        assert df_features["demand_roll_mean_7"].tolist() == pytest.approx([16.0, 17.0, 121.5 / 7])
        assert df_features["daily_change"].tolist() == pytest.approx([2.0, -5.0, -2.5])

    def test_non_recursive_would_differ(self, sample_weather_df):
        history = _history([10, 12, 14, 16, 18, 20, 22])
        preds, _ = predict_recursive(
            StubLinearModel(), sample_weather_df.head(3), "Maharashtra", historical_demand=history)
        assert len(set(np.round(preds, 6))) == 3  # old code gave three identical values

    def test_teacher_forcing_uses_observed_demand_for_buffer(self, sample_weather_df):
        history = _history([10, 12, 14, 16, 18, 20, 22, 100, 100, 100], end="2023-01-03")
        # observed 2023-01-01..03 are 100; only dates < 2023-01-01 seed the buffer
        preds, _ = predict_recursive(
            StubLinearModel(), sample_weather_df.head(3), "Maharashtra",
            historical_demand=history, teacher_forcing=True)
        assert preds[0] == pytest.approx(17.0)
        # step 2 buffer ends with the OBSERVED 100, not the prediction 17
        assert preds[1] == pytest.approx(np.mean([12, 14, 16, 18, 20, 22, 100]) + 0.5 * 78)

    def test_model_without_demand_features_falls_back_to_batch_predict(self, sample_weather_df):
        class WeatherOnly:
            feature_names_in_ = np.array(["month"])

            def predict(self, X):
                return X["month"].to_numpy(dtype=float) * 2.0

        preds, _ = predict_recursive(WeatherOnly(), sample_weather_df.head(3), "Kerala",
                                     historical_demand=_history([1.0] * 40))
        assert preds.tolist() == [2.0, 2.0, 2.0]


class TestDemandFeaturesMatchTraining:
    """PURPOSE: inference demand features must equal train_pipeline.add_demand_lag_features.
    CATCHES: drift in window/ddof/pct-change definitions between training and serving."""

    @pytest.mark.parametrize("t", [2, 3, 8, 20, 35, 60])
    def test_row_t_features_equal_training_definition(self, t):
        rng = np.random.default_rng(7)
        y = 100 + np.cumsum(rng.normal(0, 3, 80))
        train = add_demand_lag_features(
            pd.DataFrame({"g": "a", "y": y}), group_cols=["g"], target_col="y")

        known = y[:t]  # y(0..t-1): everything observable when forecasting row t
        buf = np.full((1, DEMAND_BUFFER_LEN), np.nan)
        seed = known[-DEMAND_BUFFER_LEN:]
        buf[0, -len(seed):] = seed
        feats = demand_features_from_buffer(buf)

        for name in DEMAND_FEATURE_NAMES:
            if np.isnan(train.loc[t, name]):
                # Training leaves lag_k undefined (row dropped) when < k prior days exist;
                # serving falls back to a finite value instead (see test_demand_lags.py).
                assert name.startswith("demand_lag_") and np.isfinite(feats[name][0])
                continue
            assert feats[name][0] == pytest.approx(train.loc[t, name]), name

    def test_zero_previous_value_gives_zero_pct_change(self):
        buf = np.full((1, DEMAND_BUFFER_LEN), np.nan)
        buf[0, -3:] = [5.0, 0.0, 4.0]
        assert demand_features_from_buffer(buf)["daily_pct_change"][0] == 0.0


class TestBatchCore:
    def test_runs_are_independent(self):
        model = StubLinearModel()
        buf = np.full((2, DEMAND_BUFFER_LEN), np.nan)
        buf[0, -7:] = [10, 12, 14, 16, 18, 20, 22]
        buf[1, -7:] = [50] * 7
        static = np.zeros((2, 2, 1))
        preds, fed = recursive_forecast_batch(model, ["demand_roll_mean_7", "daily_change"],
                                              static, ["unused"], buf)
        assert preds[0, 0] == pytest.approx(17.0)
        assert preds[1].tolist() == pytest.approx([50.0, 50.0])
        assert fed.shape == (2, 2)

    def test_unknown_feature_is_rejected(self):
        with pytest.raises(ValueError, match="mystery"):
            recursive_forecast_batch(StubLinearModel(), ["mystery"], np.zeros((1, 1, 1)),
                                     ["other"], np.ones((1, DEMAND_BUFFER_LEN)))


class TestGetHorizonRmse:
    """PURPOSE: one lookup with a documented fallback chain.
    CATCHES: CIs silently reverting to the 1-day RMSE, or narrowing as horizon grows."""

    META = {
        "state_rmse": {"Kerala": 2.0, "Goa": 1.0},
        "horizon_rmse": {"1": 4.0, "2": 6.0, "3": 5.0, "30": 20.0},
        "state_horizon_rmse": {"Kerala": {"1": 2.0, "2": 3.0, "3": 2.5}},
        "national_horizon_rmse": {"1": 100.0, "2": 150.0},
    }

    def test_state_specific_table_wins_and_is_monotone(self):
        assert get_horizon_rmse("Kerala", 1, self.META) == 2.0
        assert get_horizon_rmse("Kerala", 3, self.META) == 3.0  # running max, 2.5 < 3.0 ignored

    def test_horizon_beyond_table_reuses_last_value(self):
        assert get_horizon_rmse("Kerala", 30, self.META) == 3.0

    def test_scaled_overall_when_state_table_missing(self):
        # Goa: overall(h=2)=6 * state_rmse(1.0) / overall(h=1)(4.0) = 1.5
        assert get_horizon_rmse("Goa", 2, self.META) == pytest.approx(1.5)
        assert get_horizon_rmse("Goa", 3, self.META) == pytest.approx(1.5)  # running max of 6,5

    def test_flat_state_rmse_when_no_horizon_data(self):
        assert get_horizon_rmse("Goa", 7, {"state_rmse": {"Goa": 1.0}}) == 1.0

    def test_none_when_nothing_known(self):
        assert get_horizon_rmse("Goa", 1, {}) is None
        assert get_horizon_rmse("Goa", 1, None) is None
        assert get_horizon_rmse("Nowhere", 1, self.META) is None

    def test_jammu_and_kashmir_alias(self):
        meta = {"state_horizon_rmse": {"J&K": {"1": 7.0}}}
        assert get_horizon_rmse("Jammu and Kashmir", 1, meta) == 7.0

    def test_national_uses_national_table(self):
        assert get_horizon_rmse("All-India", 2, self.META) == 150.0
        assert get_horizon_rmse("All-India", 1, {"state_rmse": {}}) is None

    def test_series_is_non_decreasing_and_none_if_unavailable(self):
        series = get_horizon_rmse_series("Kerala", 5, self.META)
        assert all(b >= a for a, b in zip(series, series[1:]))
        assert get_horizon_rmse_series("Nowhere", 5, self.META) is None


class TestRollingSplit:
    def test_last_twelve_months_are_test(self):
        dates = pd.date_range("2013-04-02", "2026-08-29")
        cutoff = rolling_split_cutoff(dates)
        assert cutoff == pd.Timestamp("2025-08-30")
        test = dates[dates >= cutoff]
        assert test.min() == pd.Timestamp("2025-08-30") and test.max() == pd.Timestamp("2026-08-29")
        assert len(test) == 365

    def test_cutoff_moves_with_new_data(self):
        base = pd.date_range("2020-01-01", "2024-06-30")
        extended = pd.date_range("2020-01-01", "2024-12-31")
        assert rolling_split_cutoff(extended) > rolling_split_cutoff(base)


class TestBacktestHonesty:
    """PURPOSE: backtest errors must be exactly computable on synthetic series.
    CATCHES: horizon misalignment, origin leakage, wrong baseline definitions."""

    FEATS = ["demand_roll_mean_7", "daily_change", "month"]

    class MeanModel:
        feature_names_in_ = np.array(["demand_roll_mean_7", "daily_change", "month"])

        def predict(self, X):
            return X["demand_roll_mean_7"].to_numpy()

    @staticmethod
    def _frame(fn, n=150):
        dates = pd.date_range("2024-01-01", periods=n)
        rows = [pd.DataFrame({"date": dates, "states": s, "energy met (mu)": fn(np.arange(n), lvl),
                              "month": dates.month}) for s, lvl in (("A", 100.0), ("B", 50.0))]
        return pd.concat(rows, ignore_index=True)

    def test_constant_series_has_zero_error_everywhere(self):
        df = self._frame(lambda t, lvl: np.full(len(t), lvl))
        res = run_horizon_backtest(self.MeanModel(), df, self.FEATS, "2024-03-01", max_horizon=10)
        assert all(v == pytest.approx(0.0) for v in res["horizon_rmse"].values())
        assert all(v == pytest.approx(0.0) for v in res["baseline_horizon_rmse"]["persistence"].values())
        assert all(v == pytest.approx(0.0) for v in res["national_horizon_rmse"].values())

    def test_ramp_baselines_have_known_errors(self):
        df = self._frame(lambda t, lvl: lvl + 2.0 * t)  # slope 2 MU/day
        res = run_horizon_backtest(self.MeanModel(), df, self.FEATS, "2024-03-01", max_horizon=15)
        persistence = res["baseline_horizon_rmse"]["persistence"]
        seasonal = res["baseline_horizon_rmse"]["seasonal_naive"]
        for h in (1, 5, 15):  # y(o+h) - y(o) = 2h
            assert persistence[str(h)] == pytest.approx(2.0 * h)
        assert seasonal["3"] == pytest.approx(14.0)    # 7 days back  -> 2*7
        assert seasonal["7"] == pytest.approx(14.0)
        assert seasonal["8"] == pytest.approx(28.0)    # 14 days back -> 2*14
        assert set(res["state_horizon_rmse"]) == {"A", "B"}

    def test_model_forecast_ignores_demand_after_the_origin(self):
        """Raising demand on every day AFTER an origin must not move that origin forecast."""
        df = self._frame(lambda t, lvl: np.full(len(t), lvl))
        origin = pd.Timestamp("2024-03-01")  # first origin of the backtest
        shifted = df.copy()
        shifted.loc[shifted["date"] > origin, "energy met (mu)"] += 1000.0
        res = run_horizon_backtest(self.MeanModel(), shifted, self.FEATS, "2024-03-02",
                                   max_horizon=3, origin_stride=10_000)
        # Single origin (2024-03-01); history up to it is constant, so the recursive forecast
        # is the constant level and the error is exactly the +1000 shift at every horizon.
        assert res["backtest_config"]["n_origin_runs"] == 2
        assert all(v == pytest.approx(1000.0) for v in res["horizon_rmse"].values())
