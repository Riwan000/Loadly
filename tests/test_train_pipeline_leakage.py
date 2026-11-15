# ============================================================================
# TARGET LEAKAGE GUARD TESTS for utils/train_pipeline.py demand features
# ============================================================================
import numpy as np
import pandas as pd
import pytest

from utils.train_pipeline import add_demand_lag_features

DEMAND_FEATURE_NAMES = [
    "demand_roll_mean_7",
    "demand_roll_mean_14",
    "demand_roll_mean_30",
    "demand_roll_std_7",
    "demand_roll_std_30",
    "daily_change",
    "daily_pct_change",
]


def _make_panel():
    """Two-group synthetic panel (states x source), 20 rows each, so all
    rolling windows (up to 14) have plenty of history."""
    n = 20
    dates = pd.date_range("2023-01-01", periods=n)
    frames = []
    for state, source, base in [("Maharashtra", "Total", 100.0), ("Kerala", "Total", 50.0)]:
        demand = base + np.arange(n, dtype=float) * 2.0
        frames.append(pd.DataFrame({
            "states": state,
            "source": source,
            "date": dates,
            "energy met (mu)": demand,
        }))
    return pd.concat(frames, ignore_index=True).sort_values(["states", "source", "date"]).reset_index(drop=True)


class TestNoTargetLeakage:
    """PURPOSE: Guard against reintroducing the leak where daily_change /
    daily_pct_change (and other demand-derived features) at row t are
    computed from y_t instead of strictly from y_{t-1} and earlier.
    CATCHES: Any future edit that removes the shift and lets the target
    leak into a feature, inflating offline metrics with an artifact that
    disappears at inference (features.py never has access to y_t)."""

    def test_perturbing_target_at_row_t_does_not_change_features_at_row_t(self):
        df = _make_panel()
        result_before = add_demand_lag_features(df.copy(), group_cols=["states", "source"], target_col="energy met (mu)")

        perturbed = df.copy()
        # Pick a row with enough history on both sides (row 10 of the first group).
        target_row = 10
        perturbed.loc[target_row, "energy met (mu)"] = perturbed.loc[target_row, "energy met (mu)"] + 10_000.0

        result_after = add_demand_lag_features(perturbed, group_cols=["states", "source"], target_col="energy met (mu)")

        for col in DEMAND_FEATURE_NAMES:
            before_val = result_before.loc[target_row, col]
            after_val = result_after.loc[target_row, col]
            assert before_val == pytest.approx(after_val, nan_ok=True), (
                f"{col} at row {target_row} changed after perturbing y_t at that same row "
                f"({before_val} -> {after_val}) — target is leaking into the feature."
            )

    def test_perturbing_target_at_row_t_does_not_change_any_row_before_t(self):
        """A leak could also show up as an earlier row picking up a later
        perturbation through an unshifted rolling/groupby operation."""
        df = _make_panel()
        result_before = add_demand_lag_features(df.copy(), group_cols=["states", "source"], target_col="energy met (mu)")

        perturbed = df.copy()
        target_row = 10
        perturbed.loc[target_row, "energy met (mu)"] = perturbed.loc[target_row, "energy met (mu)"] + 10_000.0
        result_after = add_demand_lag_features(perturbed, group_cols=["states", "source"], target_col="energy met (mu)")

        earlier_rows = result_before.index[result_before.index < target_row]
        for col in DEMAND_FEATURE_NAMES:
            pd.testing.assert_series_equal(
                result_before.loc[earlier_rows, col],
                result_after.loc[earlier_rows, col],
                check_names=False,
            )

    def test_daily_change_equals_y_t_minus_1_minus_y_t_minus_2(self):
        df = _make_panel()
        result = add_demand_lag_features(df, group_cols=["states", "source"], target_col="energy met (mu)")

        group = df[(df["states"] == "Maharashtra") & (df["source"] == "Total")].reset_index(drop=True)
        group_result = result[(result["states"] == "Maharashtra") & (result["source"] == "Total")].reset_index(drop=True)

        for t in range(2, len(group)):
            expected = group.loc[t - 1, "energy met (mu)"] - group.loc[t - 2, "energy met (mu)"]
            assert group_result.loc[t, "daily_change"] == pytest.approx(expected)

    def test_first_two_rows_per_group_are_nan_warmup(self):
        """Rows 0 and 1 of each group lack two known prior days and must be
        left as NaN (for the caller to drop), never silently filled."""
        df = _make_panel()
        result = add_demand_lag_features(df, group_cols=["states", "source"], target_col="energy met (mu)")

        for (_, _), group in result.groupby(["states", "source"]):
            group = group.reset_index(drop=True)
            assert pd.isna(group.loc[0, "daily_change"])
            assert pd.isna(group.loc[1, "daily_change"])
            assert pd.isna(group.loc[0, "demand_roll_mean_7"])
            # Row 1 has one known prior day (y0), so the rolling mean is defined.
            assert not pd.isna(group.loc[1, "demand_roll_mean_7"])
            assert not pd.isna(group.loc[2, "daily_change"])

    def test_daily_pct_change_handles_zero_division_without_nan_leak(self):
        """A lag2 == 0 should map to 0 (defined), not propagate as NaN and
        get mistaken for a warm-up row."""
        n = 10
        dates = pd.date_range("2023-01-01", periods=n)
        demand = [0.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0]
        df = pd.DataFrame({
            "states": "Goa",
            "source": "Total",
            "date": dates,
            "energy met (mu)": demand,
        })
        result = add_demand_lag_features(df, group_cols=["states", "source"], target_col="energy met (mu)")
        # Row 2: lag1=0.0 (row1), lag2=0.0 (row0) -> division by zero -> mapped to 0, not NaN.
        assert result.loc[2, "daily_pct_change"] == 0.0
